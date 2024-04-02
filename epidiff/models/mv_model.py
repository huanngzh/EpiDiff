from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from einops import rearrange

from epidiff.utils.project import HarmonicEmbedding, process_cameras


@dataclass
class RenderConfig:
    ray_start: float = 0.5
    ray_end: float = 2.8
    resolution: int = 32
    n_samples: int = 8
    disparity_space_sampling: bool = False
    box_warp: float = 1.0


class MVModel(nn.Module):
    def __init__(
        self,
        cp_block_model: nn.Module,
        unet: Optional[UNet2DConditionModel] = None,
        base_model_id: str = "bennyguo/zero123-xl-diffusers",
        variant: str = "fp16_ema",
        train_unet: bool = False,
        insert_stages: List[int] = ["mid", "up"],
        insert_up_layers: List[int] = [0, 1, 2, 3],
        render_options: RenderConfig = RenderConfig(),
        use_residual: bool = False,
    ):
        super().__init__()

        self.trainable_parameters = []

        # prepare unet
        self.unet = unet
        if unet is None:
            self.unet = UNet2DConditionModel.from_pretrained(
                base_model_id, subfolder="unet", variant=variant
            )

        if train_unet:
            self.trainable_parameters += [(self.unet.parameters(), 0.01)]
        else:
            self.unet.requires_grad_(False)

        self.insert_stages = insert_stages
        self.insert_up_layers = insert_up_layers
        self.use_residual = use_residual

        render_options = vars(render_options)
        self.render_options = vars(RenderConfig())
        self.render_options.update(render_options)

        # camera embedding
        self.harmonic_embedding = HarmonicEmbedding()
        embed_dim = HarmonicEmbedding.get_output_dim_static(
            6, n_harmonic_functions=6, append_input=True
        )

        self.cp_blocks_encoder = nn.ModuleList()
        self.cp_blocks_mid = nn.ModuleList()
        self.cp_blocks_decoder = nn.ModuleList()

        if "down" in self.insert_stages:
            for i in range(len(self.unet.down_blocks)):
                self.cp_blocks_encoder.append(
                    cp_block_model(
                        dim=self.unet.down_blocks[i].resnets[-1].out_channels,
                        embed_dim=embed_dim,
                    )
                )
        if "mid" in self.insert_stages:
            self.cp_blocks_mid.append(
                cp_block_model(
                    self.unet.mid_block.resnets[-1].out_channels,
                    embed_dim=embed_dim,
                )
            )
        if "up" in self.insert_stages:
            for i in range(len(self.unet.up_blocks)):
                if i in self.insert_up_layers:
                    self.cp_blocks_decoder.append(
                        cp_block_model(
                            self.unet.up_blocks[i].resnets[-1].out_channels,
                            embed_dim=embed_dim,
                        )
                    )

        self.trainable_parameters += [
            (
                list(self.cp_blocks_mid.parameters())
                + list(self.cp_blocks_decoder.parameters())
                + list(self.cp_blocks_encoder.parameters()),
                1.0,
            )
        ]

    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op=None
    ) -> None:
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def forward(self, latents, timestep, prompt_embd, meta):
        cameras = meta["cameras"]
        k_near_indices = meta["k_near_indices"]
        b, m, c, h, w = latents.shape

        # [0, 1] -> [-1, 1]
        # bs*m, 4, 64, 64
        hidden_states = rearrange(latents, "b m c h w -> (b m) c h w")
        prompt_embd = rearrange(prompt_embd, "b m l c -> (b m) l c")

        # 1. process timesteps
        timestep = timestep.reshape(-1)
        t_emb = self.unet.time_proj(timestep)  # (bs, 320)
        emb = self.unet.time_embedding(t_emb)  # (bs, 1280)

        hidden_states = self.unet.conv_in(hidden_states)  # bs*m, 320, 64, 64

        # unet
        # a. downsample
        down_block_res_samples = (hidden_states,)
        for i, downsample_block in enumerate(self.unet.down_blocks):
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                for resnet, attn in zip(
                    downsample_block.resnets, downsample_block.attentions
                ):
                    hidden_states = resnet(hidden_states, emb)
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample
                    down_block_res_samples += (hidden_states,)
            else:
                for resnet in downsample_block.resnets:
                    hidden_states = resnet(hidden_states, emb)
                    down_block_res_samples += (hidden_states,)

            if len(self.cp_blocks_encoder) > 0:
                (
                    projected_points,
                    mask,
                    query_plucker,
                    reference_plucker,
                    sample_depths,
                ) = process_cameras(
                    cameras,
                    k_near_indices,
                    harmonic_embedding=self.harmonic_embedding,
                    rendering_options=self.render_options,
                )
                cp_block_states = self.cp_blocks_encoder[i](
                    hidden_states,
                    projected_points,
                    mask,
                    query_plucker,
                    reference_plucker,
                    m,
                    emb,
                )
                hidden_states = (
                    hidden_states + cp_block_states
                    if self.use_residual
                    else cp_block_states
                )

            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)
                self.render_options["resolution"] = (
                    self.render_options["resolution"] // 2
                )

        # b. mid
        hidden_states = self.unet.mid_block.resnets[0](hidden_states, emb)
        if len(self.cp_blocks_mid) > 0:
            (
                projected_points,
                mask,
                query_plucker,
                reference_plucker,
                sample_depths,
            ) = process_cameras(
                cameras,
                k_near_indices,
                harmonic_embedding=self.harmonic_embedding,
                rendering_options=self.render_options,
            )
            assert self.render_options["resolution"] == hidden_states.shape[-1]
            cp_block_states = self.cp_blocks_mid[0](
                hidden_states,
                projected_points,
                mask,
                query_plucker,
                reference_plucker,
                m,
                emb,
            )
            hidden_states = (
                hidden_states + cp_block_states
                if self.use_residual
                else cp_block_states
            )

        for attn, resnet in zip(
            self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]
        ):
            hidden_states = attn(
                hidden_states, encoder_hidden_states=prompt_embd
            ).sample
            hidden_states = resnet(hidden_states, emb)

        h, w = hidden_states.shape[-2:]

        # c. upsample
        current_upblock_id = 0
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                for resnet, attn in zip(
                    upsample_block.resnets, upsample_block.attentions
                ):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample
            else:
                for resnet in upsample_block.resnets:
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)

            if len(self.cp_blocks_decoder) > 0 and i in self.insert_up_layers:
                (
                    projected_points,
                    mask,
                    query_plucker,
                    reference_plucker,
                    sample_depths,
                ) = process_cameras(
                    cameras,
                    k_near_indices,
                    harmonic_embedding=self.harmonic_embedding,
                    rendering_options=self.render_options,
                )
                cp_block_states = self.cp_blocks_decoder[current_upblock_id](
                    hidden_states,
                    projected_points,
                    mask,
                    query_plucker,
                    reference_plucker,
                    m,
                    emb,
                )
                current_upblock_id += 1
                hidden_states = (
                    hidden_states + cp_block_states
                    if self.use_residual
                    else cp_block_states
                )

            if upsample_block.upsamplers is not None:
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)
                self.render_options["resolution"] = (
                    self.render_options["resolution"] * 2
                )

        # 4.post-process
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, "(b m) c h w -> b m c h w", m=m)
        return sample
