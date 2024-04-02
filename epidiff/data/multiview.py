import json
import math
import os
from os.path import join as osp
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class MultiViewDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        num_views: int,
        bg_color: str,
        img_wh: Tuple[int, int],
        caption_path: str,
        ref_view_mode: Union[str, List] = "random",
        sample_views_mode: Union[str, List] = "random",
        k_near_views: Optional[int] = None,
        num_samples: Optional[int] = None,
        use_abs_elevation: bool = False,
    ):
        data = []
        with open(caption_path, "r") as f:
            for line in f:
                obj_id, caption = line.strip().split("\t")
                obj_dir = osp(root_dir, obj_id)
                data.append({"obj_dir": obj_dir, "caption": caption})

        if num_samples is not None:
            data = data[:num_samples]

        self.data = data
        self.num_views = num_views
        self.bg_color = bg_color
        self.img_wh = img_wh
        self.ref_view_mode = ref_view_mode
        self.sample_views_mode = sample_views_mode
        self.k_near_views = k_near_views if k_near_views else num_views
        self.use_abs_elevation = use_abs_elevation

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            bg_color = np.random.rand(3)
        elif bg_color == "random_gray":
            bg_color = np.random.rand(1) * 0.5 + 0.3
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, float):
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(self, img_path, bg_color, rescale=True, return_type="np"):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 255.0  # [0, 1]

        if img.shape[-1] == 4:
            alpha = img[..., 3:4]
            img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if rescale:
            img = img * 2.0 - 1.0  # to -1 ~ 1

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def get_k_near_views(self, elevations, azimuths, k_near_views, num_views):
        views = torch.cat((elevations.unsqueeze(1), azimuths.unsqueeze(1)), dim=1)
        distances = torch.cdist(views, views)
        torch.fill_(distances.diagonal(), 0.0)
        k_nearest_indices = torch.topk(distances, k_near_views, largest=False).indices

        return k_nearest_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        obj_dir = sample["obj_dir"]
        with open(osp(obj_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        # no use
        caption = sample["caption"]

        # camera intrinsics
        fov = meta["camera_angle_x"]
        focal_length = 0.5 * 1 / np.tan(0.5 * fov)  # FIXME: hard-code
        intrinsics = np.array(
            [
                [focal_length, 0, 0.5],
                [0, focal_length, 0.5],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        intrinsics = torch.from_numpy(intrinsics)
        intrinsics_4x4 = torch.zeros(4, 4)
        intrinsics_4x4[:3, :3] = intrinsics
        intrinsics_4x4[3, 3] = 1.0

        # sample or select ref view ids
        num_views_all = len(meta["locations"])
        if self.ref_view_mode == "random":
            ref_view_id = np.random.randint(num_views_all)
        elif isinstance(self.ref_view_mode, int):
            ref_view_id = self.ref_view_mode
        else:
            raise NotImplementedError

        # sample or select view ids in a mode
        num_views_all = len(meta["locations"])
        if self.sample_views_mode == "random":
            view_ids = np.random.choice(
                num_views_all, self.num_views - 1, replace=False
            )
        elif isinstance(self.sample_views_mode, list) or isinstance(
            self.sample_views_mode, ListConfig
        ):
            assert len(self.sample_views_mode) == self.num_views - 1, "Invalid view_ids"
            view_ids = self.sample_views_mode
        else:
            raise NotImplementedError
        view_ids = np.insert(view_ids, 0, ref_view_id)
        locations = [meta["locations"][i] for i in view_ids]

        # load images, elevations, azimuths, c2w_matrixs
        bg_color = self.get_bg_color(self.bg_color)
        img_paths, img_tensors, elevations, azimuths, c2w_matrixs = [], [], [], [], []
        for loc in locations:
            img_path = osp(obj_dir, loc["frames"][0]["name"])
            img = self.load_image(img_path, bg_color, return_type="pt").permute(2, 0, 1)
            img_tensors.append(img)
            img_paths.append(img_path)
            elevations.append(loc["elevation"])
            azimuths.append(loc["azimuth"])
            c2w_matrixs.append(loc["transform_matrix"])

        # concat and stack
        img_tensors = torch.stack(img_tensors, dim=0).float()  # (Nv, 3, H, W)
        elevations = torch.tensor(elevations).float()  # (Nv,)
        azimuths = torch.tensor(azimuths).float()  # (Nv,)
        c2w_matrixs = torch.tensor(c2w_matrixs).float()  # (Nv, 4, 4)

        # blender to opencv
        c2w_matrixs[:, :, 1:3] *= -1

        intrinsics_matrixs = intrinsics.unsqueeze(0).repeat(
            self.num_views, 1, 1
        )  # (Nv, 3, 3)
        intrinsics_matrixs_4x4 = intrinsics_4x4.unsqueeze(0).repeat(
            self.num_views, 1, 1
        )  # (Nv, 4, 4)

        # flatten intrinsics_matrixs_4x4 and c2w_matrixs to (Nv, 16), and concat them
        intrinsics_matrixs_4x4 = intrinsics_matrixs_4x4.reshape(self.num_views, 16)
        c2w_matrixs = c2w_matrixs.reshape(self.num_views, 16)
        camera_params = torch.cat((intrinsics_matrixs_4x4, c2w_matrixs), dim=1)
        c2w_matrixs = c2w_matrixs.reshape(self.num_views, 4, 4)

        # find nearest views
        if self.k_near_views > self.num_views:
            raise ValueError("k_near_views should be no larger than num_views")
        else:
            k_near_indices = self.get_k_near_views(
                elevations[1:], azimuths[1:], self.k_near_views, self.num_views - 1
            )  # NOTE: we use the 1st view as the reference view
        d_elevations = (elevations - elevations[0:1]).reshape(-1, 1)
        d_azimuths = (azimuths - azimuths[0:1]).reshape(-1, 1) % (2 * math.pi)

        if self.use_abs_elevation:  # stable zero123 use abs_elevation as distance input
            distances = elevations[0:1].repeat(self.num_views).reshape(-1, 1)
        else:
            distances = torch.zeros_like(d_elevations)

        return {
            "view_ids": torch.tensor(view_ids),
            "images": img_tensors[1:],
            "image_0": img_tensors[0],
            "elevations": d_elevations[1:],
            "azimuths": d_azimuths[1:],
            "distances": distances[1:],
            "c2w_matrixs": c2w_matrixs[1:],
            "intrinsics_matrixs": intrinsics_matrixs[1:],
            "cameras": camera_params[1:],
            "caption": caption,
            "k_near_indices": k_near_indices,
        }


class MultiViewDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset[Any],
        val_dataset: Dataset[Any],
        test_dataset: Dataset[Any],
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train = train_dataset
        self.data_val = val_dataset
        self.data_test = test_dataset

        self.num_workers = num_workers if num_workers else train_batch_size * 2

    def prepare_data(self) -> None:
        # TODO: check if data is available
        pass

    def _dataloader(
        self, dataset: Dataset, batch_size: int, shuffle: bool
    ) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        # support multiple validation datasets
        if isinstance(self.data_val, ListConfig):
            return [
                self._dataloader(dataset, self.hparams.val_batch_size, False)
                for dataset in self.data_val
            ]
        elif isinstance(self.data_val, DictConfig):
            return [
                self._dataloader(dataset, self.hparams.val_batch_size, False)
                for _, dataset in self.data_val.items()
            ]
        else:
            return self._dataloader(self.data_val, self.hparams.val_batch_size, False)

    def test_dataloader(self) -> DataLoader[Any]:
        # support multiple test datasets
        if isinstance(self.data_test, ListConfig):
            return [
                self._dataloader(dataset, self.hparams.test_batch_size, False)
                for dataset in self.data_test
            ]
        elif isinstance(self.data_test, DictConfig):
            return [
                self._dataloader(dataset, self.hparams.test_batch_size, False)
                for _, dataset in self.data_test.items()
            ]
        else:
            return self._dataloader(self.data_test, self.hparams.test_batch_size, False)


if __name__ == "__main__":
    from torchvision.utils import save_image

    dataset = MultiViewDataset(
        root_dir="/mnt/pfs/data/render_lvis_hzh",
        num_views=17,
        k_near_views=4,
        bg_color="white",
        img_wh=(256, 256),
        ref_view_mode=2,
        sample_views_mode="random",
        caption_path="data/render_lvis_hzh/caption_train.txt",
        use_abs_elevation=True,
    )

    sample = dataset[0]
    print(sample["view_ids"])
    save_image((sample["image_0"] + 1.0) / 2, "temp_0.jpg", nrow=1, value_range=(0, 1))
    save_image((sample["images"] + 1.0) / 2, "temp.jpg", nrow=8, value_range=(0, 1))
