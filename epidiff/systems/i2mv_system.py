import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from einops import rearrange, repeat
from lightning import LightningModule
from lightning_utilities.core.rank_zero import rank_zero_only
from PIL import Image
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from extern.zero123 import CLIPCameraProjection
from extern.zero123_official import CCProjection

from .base import BaseSystem

BASE_MODEL_OPTIONS = [
    "bennyguo/zero123-xl-diffusers",
    "kxic/zero123-xl",
    "ashawkey/stable-zero123-diffusers",
]


class I2MVSystem(BaseSystem):
    def __init__(
        self,
        mv_model: torch.nn.Module,
        lr: float,
        base_model_id: str = "bennyguo/zero123-xl-diffusers",
        variant: Optional[str] = None,
        cfg: float = 0.1,  # classifier free guidance
        diffusion_timesteps: int = 50,
        guidance_scale: float = 3.0,
        num_val_dataloaders: int = 2,
        num_test_dataloaders: int = 2,
        report_to: str = "wandb",
        compile: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["mv_model"])

        # check pretrained base model
        assert (
            base_model_id in BASE_MODEL_OPTIONS
        ), f"base_model_id must be one of {BASE_MODEL_OPTIONS}"

        # prepare pretrain image encoder, vae, unet, scheduler for diffusion
        pipe_kwargs = {}
        if variant is not None:
            pipe_kwargs["variant"] = variant
        unet = UNet2DConditionModel.from_pretrained(
            base_model_id, subfolder="unet", **pipe_kwargs
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_id, subfolder="image_encoder", **pipe_kwargs
        )
        self.vae = AutoencoderKL.from_pretrained(
            base_model_id, subfolder="vae", **pipe_kwargs
        )
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_id, subfolder="feature_extractor", **pipe_kwargs
        )
        self.scheduler = DDIMScheduler.from_pretrained(
            base_model_id, subfolder="scheduler"
        )
        self.image_encoder.eval()
        self.vae.eval()
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        # prepare clip camera projection for different base models
        if base_model_id.endswith("diffusers"):
            self.clip_camera_projection = CLIPCameraProjection.from_pretrained(
                base_model_id, subfolder="clip_camera_projection", **pipe_kwargs
            )
        else:
            self.clip_camera_projection = CCProjection.from_pretrained(
                base_model_id, subfolder="cc_projection", **pipe_kwargs
            )

        # prepare model
        self.mv_model = mv_model(unet=unet)
        self.mv_model.set_use_memory_efficient_attention_xformers(True)
        self.trainable_parameters = list(self.mv_model.trainable_parameters)

        # metric objects for calculating and averaging accuracy across batches
        self.val_psnr = nn.ModuleList(
            [PeakSignalNoiseRatio(data_range=1.0) for _ in range(num_val_dataloaders)]
        )
        self.val_ssim = nn.ModuleList(
            [StructuralSimilarityIndexMeasure() for _ in range(num_val_dataloaders)]
        )
        self.val_lpips = nn.ModuleList(
            [
                LearnedPerceptualImagePatchSimilarity(normalize=True)
                for _ in range(num_val_dataloaders)
            ]
        )
        self.val_psnr_best = nn.ModuleList(
            [MaxMetric() for _ in range(num_val_dataloaders)]
        )

        self.test_psnr = nn.ModuleList(
            [PeakSignalNoiseRatio(data_range=1.0) for _ in range(num_test_dataloaders)]
        )
        self.test_ssim = nn.ModuleList(
            [StructuralSimilarityIndexMeasure() for _ in range(num_test_dataloaders)]
        )
        self.test_lpips = nn.ModuleList(
            [
                LearnedPerceptualImagePatchSimilarity(normalize=True)
                for _ in range(num_test_dataloaders)
            ]
        )

    def setup(self, stage: str) -> None:
        super().setup(stage)

        if self.hparams.compile and stage == "fit":
            self.mv_model = torch.compile(self.mv_model)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers for training."""
        param_groups = []
        for params, lr_scale in self.trainable_parameters:
            param_groups.append({"params": params, "lr": self.hparams.lr * lr_scale})

        optimizer = torch.optim.AdamW(param_groups)
        return optimizer

    def forward(self, latents, timestep, prompt_embd, meta) -> torch.Tensor:
        return self.mv_model(latents, timestep, prompt_embd, meta)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for metrics in [
            self.val_psnr,
            self.val_ssim,
            self.val_lpips,
            self.val_psnr_best,
        ]:
            if isinstance(metrics, nn.ModuleList):
                for metric in metrics:
                    metric.reset()
            else:
                metrics.reset()

    def training_step(self, batch, batch_idx):
        meta = {"cameras": batch["cameras"], "k_near_indices": batch["k_near_indices"]}
        image_0 = batch["image_0"]  # cond image (b c h w)
        elevations = batch["elevations"]  # (b f 1)
        azimuths = batch["azimuths"]  # (b f 1)
        distances = batch["distances"]  # (b f 1)

        latent_image, image_embeddings = self._encode_cond(
            image_0, elevations, azimuths, distances
        )  # (b f 1 768)

        if torch.rand(1) < self.hparams.cfg:
            latent_image = torch.zeros_like(latent_image)

        latents = self._encode_image(batch["images"], self.vae)

        t = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()

        noise = torch.randn_like(latents)
        noise_z = self.scheduler.add_noise(latents, noise, t)
        noise_z = torch.cat([noise_z, latent_image], dim=2)  # b x f x 8
        t = t[:, None].repeat(1, latents.shape[1])

        denoise = self.forward(noise_z, t, image_embeddings, meta)
        target = noise

        # eps mode
        loss = F.mse_loss(denoise, target)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images_pred = self._generate_images(batch)
        images = ((batch["images"] / 2 + 0.5) * 255).cpu().numpy().astype(np.uint8)

        # compute image & save
        image_fp = self._save_image(
            images_pred,
            images,
            batch["caption"],
            f"{dataloader_idx}_{batch_idx}_{self.global_rank}",
            stage="validation",
        )

        # update and log metrics
        images_gt_ = rearrange(
            batch["images"] / 2 + 0.5, "b m c h w -> (b m) c h w"
        ).float()
        images_pred_ = torch.tensor(
            rearrange(images_pred, "b m h w c -> (b m) c h w") / 255.0,
            dtype=torch.float32,
        ).to(images_gt_.device)

        self.val_psnr[dataloader_idx](images_gt_, images_pred_)
        self.val_ssim[dataloader_idx](images_gt_, images_pred_)
        self.val_lpips[dataloader_idx](images_gt_, images_pred_)
        self.log(
            f"val_psnr_{dataloader_idx}",
            self.val_psnr[dataloader_idx],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"val_ssim_{dataloader_idx}",
            self.val_ssim[dataloader_idx],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"val_lpips_{dataloader_idx}",
            self.val_lpips[dataloader_idx],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

        return image_fp

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        for i in range(self.hparams.num_val_dataloaders):
            acc = self.val_psnr[i].compute()  # get current val acc
            self.val_psnr_best[i](acc)  # update best so far val acc
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log(
                f"val_psnr_best_{i}",
                self.val_psnr_best[i].compute(),
                sync_dist=True,
                prog_bar=True,
            )

        # log images
        if self.hparams.report_to == "wandb":
            self._log_to_wandb("validation")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images_pred = self._generate_images(batch)
        images = ((batch["images"] / 2 + 0.5) * 255).cpu().numpy().astype(np.uint8)

        # save images
        image_fp = self._save_image(
            images_pred,
            images,
            batch["caption"],
            f"{dataloader_idx}_{batch_idx}_{self.global_rank}",
            stage="test",
        )

        # update and log metrics
        images_gt_ = rearrange(
            batch["images"] / 2 + 0.5, "b m c h w -> (b m) c h w"
        ).float()
        images_pred_ = torch.tensor(
            rearrange(images_pred, "b m h w c -> (b m) c h w") / 255.0,
            dtype=torch.float32,
        ).to(images_gt_.device)

        self.test_psnr[dataloader_idx](images_gt_, images_pred_)
        self.test_ssim[dataloader_idx](images_gt_, images_pred_)
        self.test_lpips[dataloader_idx](images_gt_, images_pred_)
        self.log(
            f"test_psnr_{dataloader_idx}",
            self.test_psnr[dataloader_idx],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"test_ssim_{dataloader_idx}",
            self.test_ssim[dataloader_idx],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"test_lpips_{dataloader_idx}",
            self.test_lpips[dataloader_idx],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

        return image_fp

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        # log images
        if self.hparams.report_to == "wandb":
            self._log_to_wandb("test")

    @torch.no_grad()
    def _encode_cond(self, image, elevation, azimuth, distance):
        (b, f, _), device = elevation.shape, elevation.device
        latent_image = self._encode_image(image, self.vae, False)
        latent_image = repeat(latent_image, "b 1 c h w -> b f c h w", f=f)
        # image to PIL image
        image = (image + 1.0) / 2.0
        pil_images = [tv.transforms.ToPILImage()(img) for img in image]
        image = self.feature_extractor(
            images=pil_images, return_tensors="pt"
        ).pixel_values
        image = image.to(device)
        image_embedding = self.image_encoder(image).image_embeds
        image_embedding = image_embedding.unsqueeze(1)
        camera_embeddings = torch.stack(
            [
                elevation,
                torch.sin(azimuth),
                torch.cos(azimuth),
                distance,
            ],
            dim=-1,
        )

        image_embeddings = repeat(
            image_embedding, "b n c -> b f n c", f=f
        )  # (b f 1 768)
        image_embeddings = torch.cat([image_embeddings, camera_embeddings], dim=-1)

        image_embeddings = rearrange(image_embeddings, "b f n c -> (b f) n c")
        image_embeddings = self.clip_camera_projection(image_embeddings)

        image_embeddings = rearrange(image_embeddings, "(b f) n c -> b f n c", b=b)
        return latent_image, image_embeddings

    @torch.no_grad()
    def _encode_image(self, x_input, vae, scale=True):
        b = x_input.shape[0]

        x_input = x_input.reshape(
            -1, x_input.shape[-3], x_input.shape[-2], x_input.shape[-1]
        )
        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()
        z = z.reshape(
            b, -1, z.shape[-3], z.shape[-2], z.shape[-1]
        )  # (bs, 2, 4, 64, 64)

        # use the scaling factor from the vae config
        if scale:
            z = z * vae.config.scaling_factor
        z = z.float()
        return z

    @torch.no_grad()
    def _decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        latents = 1 / vae.config.scaling_factor * latents
        images = []
        for j in range(m):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
        image = (image * 255).round().astype("uint8")

        return image

    def _gen_cls_free_guide_pair(
        self, latents, latent_image, timestep, prompt_embd, batch
    ):
        latents = torch.cat([latents] * 2)
        latent_image = torch.cat([torch.zeros_like(latent_image), latent_image])

        latents = torch.cat([latents, latent_image], dim=2)
        timestep = torch.cat([timestep] * 2)

        cameras = torch.cat([batch["cameras"]] * 2)
        k_near_indices = torch.cat([batch["k_near_indices"]] * 2)
        meta = {"cameras": cameras, "k_near_indices": k_near_indices}

        return latents, timestep, prompt_embd, meta

    @torch.no_grad()
    def _forward_cls_free(
        self, latents_high_res, latent_image, _timestep, prompt_embd, batch, model
    ):
        latents, _timestep, _prompt_embd, meta = self._gen_cls_free_guide_pair(
            latents_high_res, latent_image, _timestep, prompt_embd, batch
        )
        noise_pred = model(latents, _timestep, _prompt_embd, meta)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.hparams.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        return noise_pred

    @torch.no_grad()
    def _generate_images(self, batch):
        image_0 = batch["image_0"]  # cond image b x c x h x w
        elevations = batch["elevations"]  # b x f x 1
        azimuths = batch["azimuths"]  # b x f x 1
        distances = batch["distances"]  # b x f x 1

        bs, m, _ = elevations.shape
        bs, _, h, w = image_0.shape
        device = image_0.device
        latents = torch.randn(bs, m, 4, h // 8, w // 8, device=device)

        latent_image, image_embeddings = self._encode_cond(
            image_0, elevations, azimuths, distances
        )  # b x f x 1 x 768
        negative_prompt_embeds = torch.zeros_like(image_embeddings)

        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        self.scheduler.set_timesteps(self.hparams.diffusion_timesteps, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]] * m, dim=1)

            noise_pred = self._forward_cls_free(
                latents,
                latent_image,
                _timestep,
                image_embeddings,
                batch,
                self.mv_model,
            )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        images_pred = self._decode_latent(latents, self.vae)
        return images_pred

    @torch.no_grad()
    @rank_zero_only
    def _save_image(self, images_pred, images, prompt, batch_idx, stage="validation"):
        save_dir = self.save_dir
        images = rearrange(images, "b m c h w -> (b h) (m w) c")
        images_pred = rearrange(images_pred, "b m h w c -> (b h) (m w) c")
        full_image = np.concatenate([images, images_pred], axis=0)

        with open(
            os.path.join(save_dir, f"{stage}_{self.global_step}_{batch_idx}.txt"), "w"
        ) as f:
            f.write("\n".join(prompt))

        im = Image.fromarray(full_image)
        im_fp = os.path.join(
            save_dir,
            f"{stage}_{self.global_step}_{batch_idx}--{prompt[0].replace(' ', '_').replace('/', '_')}.jpg",
        )
        im.save(im_fp)

        # add image to logger
        if self.hparams.report_to == "tensorboard":
            log_image = torch.tensor(full_image / 255.0).permute(2, 0, 1).float().cpu()
            self.logger.experiment.add_image(
                f"{stage}/{self.global_step}_{batch_idx}",
                log_image,
                global_step=self.global_step,
            )

        return im_fp

    @torch.no_grad()
    @rank_zero_only
    def _log_to_wandb(self, stage):
        import wandb

        num_dataloaders = (
            self.hparams.num_val_dataloaders
            if stage == "validation"
            else self.hparams.num_test_dataloaders
        )
        for i in range(num_dataloaders):
            captions, images = [], []
            # get images which start with {stage}_{self.global_step}_{self.dataloader_idx} from self.save_dir
            for f in os.listdir(self.save_dir):
                if f.startswith(f"{stage}_{self.global_step}_{i}") and f.endswith(
                    ".jpg"
                ):
                    captions.append(f)
                    images.append(os.path.join(self.save_dir, f))

            self.logger.experiment.log(
                {
                    f"{stage}_{i}": [
                        wandb.Image(im_fp, caption=caption)
                        for im_fp, caption in zip(images, captions)
                    ]
                },
                step=self.global_step,
            )
