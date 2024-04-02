import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import hydra
import lightning as L
import numpy as np
import torch
from einops import rearrange
from lightning import LightningModule
from PIL import Image

from epidiff.utils import load_config
from epidiff.utils.media import get_bg_color, load_image
from epidiff.utils.pose import get_k_near_views

WIDTH, HEIGHT = 256, 256
K_NEAR_VIEWS = 16


# TODO: simplify this with multiview dataset
def prepare_inputs(input_img: str, input_elevation: float, sample_views_mode: str):
    bg_color = get_bg_color("white")
    input_img = load_image(
        input_img, (WIDTH, HEIGHT), bg_color, return_type="pt"
    ).permute(2, 0, 1)

    meta_fp = f"meta_info/transforms_{sample_views_mode}.json"
    with open(meta_fp, "r") as f:
        meta = json.load(f)

    # Camera intrinsics
    fov = meta["camera_angle_x"]
    focal_length = 1 / (2 * np.tan(0.5 * fov))
    intrinsics_4x4 = torch.tensor(
        [
            [focal_length, 0, 0.5, 0],
            [0, focal_length, 0.5, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Camera extrinsics
    num_views = len(meta["frames"])
    elevations, azimuths, c2w_matrixs = [], [], []
    for frame in meta["frames"]:
        elevations.append(frame["elevation"])
        azimuths.append(frame["azimuth"])
        c2w_matrixs.append(frame["transform_matrix"])
    elevations = torch.tensor(elevations)  # (N,)
    azimuths = torch.tensor(azimuths)  # (N,)
    c2w_matrixs = torch.tensor(c2w_matrixs)  # (N, 4, 4)
    c2w_matrixs[:, :, 1:3] *= -1  # blender to opencv

    # concat intrinsics and extrinsics
    intrinsics_4x4 = (
        intrinsics_4x4.unsqueeze(0).repeat(num_views, 1, 1).reshape(num_views, 16)
    )
    _c2w_matrixs = c2w_matrixs.reshape(num_views, 16)
    camera_params = torch.cat([intrinsics_4x4, _c2w_matrixs], dim=1)

    # find nearest views
    k_near_indices = get_k_near_views(elevations, azimuths, K_NEAR_VIEWS, num_views)

    # normalize elevations and azimuths
    input_elevation = torch.tensor([input_elevation / 180 * math.pi])
    d_elevations = (elevations - input_elevation).reshape(-1, 1)
    d_azimuths = azimuths.reshape(-1, 1) % (2 * math.pi)
    distances = torch.zeros_like(d_elevations)

    return {
        "image_0": input_img,
        "elevations": d_elevations,
        "azimuths": d_azimuths,
        "distances": distances,
        "c2w_matrixs": c2w_matrixs,
        "cameras": camera_params,
        "k_near_indices": k_near_indices,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_img", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--elevation", type=float, required=True)
    parser.add_argument(
        "--sample_views_mode", type=str, choices=["ele30"], default="ele30"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args, extras = parser.parse_known_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    L.seed_everything(args.seed, workers=True)
    cfg = load_config(args.config, cli_args=extras)

    # prepare model
    model: LightningModule = hydra.utils.instantiate(cfg.system)
    model.load_weights(args.ckpt)
    model = model.to(args.device).eval()
    print(f"Loaded model from {args.ckpt}")

    # prepare data
    data = prepare_inputs(args.input_img, args.elevation, args.sample_views_mode)
    for k, v in data.items():
        data[k] = v.unsqueeze(0).to(args.device)
        if k not in ["k_near_indices"]:
            data[k] = data[k].float()

    # generate
    with torch.no_grad():
        images_pred = model._generate_images(data)

    # save
    image_base_name = os.path.basename(args.input_img).split(".")[0]
    image_list = []
    for image in images_pred[0]:
        image_list.append(Image.fromarray(image))
    image_list[0].save(
        output_dir / f"{image_base_name}.gif",
        save_all=True,
        append_images=image_list[1:],
        duration=100,
        loop=0,
    )

    full_image = rearrange(images_pred, "b m h w c -> (b h) (m w) c")
    Image.fromarray(full_image).save(output_dir / f"{image_base_name}.jpg")


if __name__ == "__main__":
    main()
