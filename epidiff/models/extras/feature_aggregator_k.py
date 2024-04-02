import os

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn

from epidiff.models.blocks.resnet import ResBlock
from epidiff.models.blocks.transformer_t import FuseTransformer as Transformer


class FeatureAggregator(nn.Module):
    def __init__(
        self,
        dim,
        depth=1,
        embed_dim=0,
        use_resnet=True,
    ):
        super().__init__()

        d_head = 64
        n_heads = dim // d_head
        # feature extract -> t1 -> t2 -> t3 (n s) (n k) (n)
        self.T1 = Transformer(
            in_channels=dim + 2 * (embed_dim + 13),
            out_channels=dim,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
        )
        self.view_fuse = nn.Linear(dim, 1, bias=False)
        self.T2 = Transformer(
            in_channels=dim,
            out_channels=dim,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
        )
        self.ray_fuse = nn.Linear(dim, 1, bias=False)

        if use_resnet:
            self.resnet = ResBlock(dim, 1280, use_scale_shift_norm=True)

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
        b, f, k, n, s, j = projected_points.shape
        bf, c, h, w = images.shape
        # images = self.resnet(images, t_emb)

        images = rearrange(images, "(b f) c h w -> (b f) h w c", f=f_views)

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
            feature_sampled, "(b f k) c n s -> b f k n s c", f=f_views, k=k_views
        )

        t1 = torch.concat((feature_sampled, query_plucker, reference_plucker), dim=-1)

        # fuse k nearest view
        t1 = rearrange(t1, "b f k n s c -> (b f s n) k c").to(images.dtype)
        mask = rearrange(mask, "b f k n s -> (b f s n) k")
        emb = repeat(t_emb, "bf c -> (bf s n) 1 c", n=n, s=s)
        out = self.T1(t1, t=emb, attention_mask=mask)

        mask = rearrange(mask, "... -> ... 1")
        weight = self.view_fuse(out)
        weight = weight.masked_fill(~mask, torch.finfo(weight.dtype).min)
        weight = F.softmax(weight, dim=1)
        t1 = torch.sum(out * weight, dim=1)  # (bf s n) c

        t2 = rearrange(t1, "(bf s n) c -> (bf n) s c", s=s, n=n)
        emb = repeat(t_emb, "bf c -> (bf n) 1 c", n=n)
        t2 = self.T2(t2, t=emb)
        weight = self.ray_fuse(t2)
        weight = F.softmax(weight, dim=1)
        t2 = torch.sum(t2 * weight, dim=1)  # (b f n) c

        images = rearrange(t2, "(bf h w) c -> bf c h w", bf=bf, h=h, w=w)
        if self.resnet is not None:
            images = self.resnet(images, t_emb)
        images = rearrange(images, "bf c h w -> (bf h w) c")
        images = rearrange(images, "(b f h w) c -> (b f) c h w", h=h, w=w, f=f_views)

        return images
