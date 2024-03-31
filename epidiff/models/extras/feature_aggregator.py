# https://github.com/zhizdev/sparsefusion/blob/main/sparsefusion/eft.py
import os
from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn

from epidiff.models.blocks.resnet import ResBlock
from epidiff.models.blocks.transformer_t import (
    AdaLayerNorm,
    CrossAttention,
    Transformer,
)


class FeatureAggregator(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        embed_dim: int = 0,
        use_resnet: bool = True,
        use_fhw_attn: bool = False,
        pos_encs: List[str] = ["abs", "ref"],
    ):
        super().__init__()

        self.use_resnet = use_resnet
        self.use_fhw_attn = use_fhw_attn
        self.pos_encs = pos_encs

        # prepare model
        self.t1_point_fc = nn.Linear(dim, 1, bias=False)
        d_head = 64
        n_heads = dim // d_head

        # feature extract -> t1 -> t2 -> t3 (n s) (n k) (n)
        self.T1 = Transformer(
            in_channels=dim + len(pos_encs) * (embed_dim + 13),
            context_dim=dim + len(pos_encs) * (embed_dim + 13),
            # in_channels=dim,
            out_channels=dim,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
        )

        if use_resnet:
            self.resnet1 = ResBlock(dim, 1280, use_scale_shift_norm=True)
        if use_fhw_attn:
            self.HW = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head)
            self.norm = AdaLayerNorm(dim)
            self.t_proj = nn.Linear(1280, dim)

    def forward(
        self,
        images,
        projected_points,
        mask,
        query_plucker,
        reference_plucker,
        f_views,
        t_emb,
    ):
        """
        images: (B, F, C, H, W)
        projected_points: (B, F, K, N, PTS_PER_RAY, 2)
        return:
            images: (B, F, C, H, W)
            T1: (NUM_INPUT_CAM, N, PTS_PER_RAY, FEATURE_DIM) -> (NUM_INPUT_CAM, (N * PTS_PER_RAY), FEATURE_DIM)
            T2: (NUM_INPUT_CAM, N, PTS_PER_RAY, FEATURE_DIM) -> (PT_PER_RAY, (NUM_INPUT_CAM * N), FEATURE_DIM)
            T2_W:  (NUM_INPUT_CAMERAS, N, PTS_PER_RAY, FEATURE_DIM) -> (PT_PER_RAY, N, 1, FEATURE_DIM)
        """
        bf, c, h, w = images.shape
        if t_emb.shape[0] != bf:
            t_emb = repeat(t_emb, "b c -> (b f) c", f=f_views)  #
        if self.use_resnet:
            images = self.resnet1(images, t_emb)
        images = rearrange(images, "(b f) c h w -> (b f) h w c", f=f_views)

        n_sample = projected_points.shape[-2]
        k_views = projected_points.shape[2]
        # save projected_points
        projected_points = rearrange(projected_points, "b f k n s j -> (b f k) n s j")
        feature_map = repeat(
            images, "(b f) h w c-> (b f k) c h w", k=k_views, f=f_views
        )

        feature_sampled = F.grid_sample(
            feature_map,
            projected_points[..., [1, 0]],
            align_corners=True,
            padding_mode="border",
        )  # (B x F x K, C, H x W, N_samples)

        feature_sampled = rearrange(
            feature_sampled,
            "(b f k) c n s -> b f k n s c",
            f=f_views,
            k=k_views,
        )
        # check nan in feature_sampled
        # assert torch.isnan(feature_sampled).sum() == 0
        # exchange information among sampled points which in the same view
        t1 = feature_sampled

        if "abs" in self.pos_encs:
            t1 = torch.cat((t1, query_plucker), dim=-1)
        if "ref" in self.pos_encs:
            t1 = torch.cat((t1, reference_plucker), dim=-1)

        t1 = rearrange(t1, "b f k n s c -> (b f n) k s c")
        t1 = t1.to(images.dtype)
        query, context = t1.split([1, k_views - 1], dim=1)  # bfn 1 s c bfn k-1 s c
        mask = rearrange(mask, "b f k n s -> (b f n) k s")
        mask = mask[:, 1:]

        query = rearrange(query, "bfn 1 s c -> bfn s c")
        context = rearrange(context, "bfn k s c -> bfn (k s) c")
        t_emb = repeat(t_emb, "bf c -> (bf n) c", n=h * w)
        out = self.T1(query, t=t_emb, context=context, attention_mask=mask)

        mask = (
            reduce(
                mask,
                "bfn k s -> bfn s 1",
                "sum",
            )
            > 0
        )
        weight = self.t1_point_fc(out)
        weight = weight.masked_fill(~mask, torch.finfo(weight.dtype).min)
        weight = F.softmax(weight, dim=1)
        t1 = torch.sum(out * weight, dim=1)
        images = rearrange(t1, "(bf h w) c -> bf c h w", bf=bf, h=h, w=w)

        if self.use_fhw_attn:
            hw = rearrange(images, "(b f) c h w -> b (f h w) c", f=f_views)
            t_emb = rearrange(t_emb, "(b f n) c -> b (f n) c", f=f_views, n=h * w)
            t_emb = self.t_proj(t_emb)
            hw = self.HW(self.norm(hw, t_emb)) + hw

            images = rearrange(hw, "b (f h w) c -> (b f) c h w", f=f_views, h=h, w=w)

        images = rearrange(images, "bf c h w -> (bf h w) c")
        images = rearrange(
            # hw,
            t1,
            "(b f h w) c -> (b f) c h w",
            h=h,
            w=w,
            f=f_views,
        )
        return images
