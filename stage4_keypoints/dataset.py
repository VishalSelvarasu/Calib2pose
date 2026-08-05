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
    def __init__(self, data_dir, split, augment=False, heatmap_size=HEATMAP_SIZE,
                 aug_strength=1.0):
        with open(os.path.join(data_dir, f"{split}.json")) as f:
            self.records = json.load(f)["records"]
        self.img_dir = os.path.join(data_dir, "images")
        self.augment = augment
        self.split = split
        self.heatmap_size = heatmap_size
        self.aug_strength = aug_strength

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

        kps_img = np.array(r["keypoints"], np.float32)          # 640 space
        kps_in = kps_img * (INPUT_SIZE / IMAGE_SIZE)            # 256 space

        if self.augment:
            img, kps_in = self._affine(img, kps_in)
            img = self._photometric(img)
            img = self._cutout(img)

        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1))

        kps_img = kps_in * (IMAGE_SIZE / INPUT_SIZE)            # back to 640
        kps_hm = kps_in * (self.heatmap_size / INPUT_SIZE)
        target, weight = make_target(kps_hm, size=self.heatmap_size)

        return {
            "image": x,
            "target": torch.from_numpy(target),
            "weight": torch.from_numpy(weight),
            "kps_img": torch.from_numpy(kps_img),
            "idx": i,
        }

    def _affine(self, img, kps):
        """Scale, translate and in-plane rotate. Keypoints transform with the
        image, so the correspondence stays exact."""
        s = self.aug_strength
        rng = np.random
        angle = rng.uniform(-25, 25) * s
        scale = 1.0 + rng.uniform(-0.25, 0.25) * s
        tx = rng.uniform(-0.10, 0.10) * s * INPUT_SIZE
        ty = rng.uniform(-0.10, 0.10) * s * INPUT_SIZE

        c = INPUT_SIZE / 2.0
        M = cv2.getRotationMatrix2D((c, c), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        out = cv2.warpAffine(img, M, (INPUT_SIZE, INPUT_SIZE),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        kps_h = np.hstack([kps, np.ones((len(kps), 1), np.float32)])
        return out, (kps_h @ M.T).astype(np.float32)

    def _cutout(self, img):
        """Random occluding rectangles. Heavily occluded views were the worst
        performing group in the first evaluation, and the renderer's distractor
        objects alone did not produce enough of them."""
        rng = np.random
        n = rng.randint(0, int(3 * self.aug_strength) + 1)
        out = img.copy()
        for _ in range(n):
            w = rng.randint(INPUT_SIZE // 12, INPUT_SIZE // 3)
            h = rng.randint(INPUT_SIZE // 12, INPUT_SIZE // 3)
            x0 = rng.randint(0, INPUT_SIZE - w)
            y0 = rng.randint(0, INPUT_SIZE - h)
            out[y0:y0 + h, x0:x0 + w] = rng.randint(0, 256, 3)
        return out

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
