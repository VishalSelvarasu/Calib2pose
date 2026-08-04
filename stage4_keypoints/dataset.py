import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from model import INPUT_SIZE, HEATMAP_SIZE, make_target

IMAGE_SIZE = 640
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class DrillKeypointDataset(Dataset):
    def __init__(self, data_dir, split, augment=False):
        with open(os.path.join(data_dir, f"{split}.json")) as f:
            self.records = json.load(f)["records"]
        self.img_dir = os.path.join(data_dir, "images")
        self.augment = augment
        self.split = split

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = cv2.imread(os.path.join(self.img_dir, r["file"]))
        if img is None:
            raise FileNotFoundError(r["file"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE),
                         interpolation=cv2.INTER_AREA)

        if self.augment:
            img = self._photometric(img)

        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1))

        kps_img = np.array(r["keypoints"], np.float32)          # 640 space
        kps_hm = kps_img * (HEATMAP_SIZE / IMAGE_SIZE)          # 64 space
        target, weight = make_target(kps_hm)

        return {
            "image": x,
            "target": torch.from_numpy(target),
            "weight": torch.from_numpy(weight),
            "kps_img": torch.from_numpy(kps_img),
            "idx": i,
        }

    def _photometric(self, img):
        rng = np.random
        f = img.astype(np.float32)
        f *= rng.uniform(0.7, 1.3)                               # brightness
        m = f.mean()
        f = (f - m) * rng.uniform(0.7, 1.3) + m                  # contrast
        if rng.rand() < 0.3:                                     # colour cast
            f *= rng.uniform(0.9, 1.1, size=3)
        if rng.rand() < 0.3:
            f += rng.normal(0, rng.uniform(1, 6), f.shape)       # sensor noise
        f = np.clip(f, 0, 255).astype(np.uint8)
        if rng.rand() < 0.2:
            k = int(rng.choice([3, 5]))
            f = cv2.GaussianBlur(f, (k, k), 0)                   # defocus
        return f


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "weight": torch.stack([b["weight"] for b in batch]),
        "kps_img": torch.stack([b["kps_img"] for b in batch]),
        "idx": torch.tensor([b["idx"] for b in batch]),
    }
