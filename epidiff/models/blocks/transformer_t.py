import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from epidiff.utils.ops import default, exists, zero_module


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads

        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = AdaLayerNorm(dim)
        self.norm2 = AdaLayerNorm(dim)
        self.norm3 = AdaLayerNorm(dim)

    def forward(self, x, t, context=None, mask=None):
        x = self.attn1(self.norm1(x, t)) + x
        x = self.attn2(self.norm2(x, t), context=context, mask=mask) + x
        x = self.ff(self.norm3(x, t)) + x
        return x


class Transformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        out_channels=None,
        depth=1,
        dropout=0.0,
        context_dim=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads=n_heads,
                    d_head=d_head,
                    context_dim=context_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        out_channels = default(out_channels, in_channels)
        self.proj_out = zero_module(nn.Linear(inner_dim, out_channels, bias=False))
        self.t_proj = nn.Linear(1280, inner_dim)

    def forward(self, x, t, context=None, attention_mask=None):
        x = self.proj_in(x)
        t = self.t_proj(t).unsqueeze(1)
        for block in self.transformer_blocks:
            x = block(x, t, context=context, mask=attention_mask)

        x = self.proj_out(x)
        return x


class FuseTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = AdaLayerNorm(dim)
        self.norm3 = AdaLayerNorm(dim)

    def forward(self, x, t, context=None, mask=None):
        # (b f n) k s c
        x = self.attn1(self.norm1(x, t), mask=mask) + x
        x = self.ff(self.norm3(x, t)) + x
        return x


class FuseTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        out_channels=None,
        depth=1,
        dropout=0.0,
        context_dim=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                FuseTransformerBlock(
                    inner_dim,
                    n_heads=n_heads,
                    d_head=d_head,
                    context_dim=context_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        out_channels = default(out_channels, in_channels)
        self.proj_out = zero_module(nn.Linear(inner_dim, out_channels, bias=False))
        self.t_proj = nn.Linear(1280, inner_dim)

    def forward(self, x, t, context=None, attention_mask=None):
        x = self.proj_in(x)
        t = self.t_proj(t)
        for block in self.transformer_blocks:
            x = block(x, t, context=context, mask=attention_mask)
        x = self.proj_out(x)
        return x
