import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

N_KP = 8
INPUT_SIZE = 256
HEATMAP_SIZE = 64
SIGMA = 2.0          # gaussian std in heatmap pixels


class KeypointNet(nn.Module):
    def __init__(self, n_kp=N_KP, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        # drop avgpool and fc; keep the conv trunk
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # 8x8 -> 16 -> 32 -> 64
        self.deconv = nn.Sequential(
            self._deconv_block(512, 256),
            self._deconv_block(256, 256),
            self._deconv_block(256, 256),
        )
        self.head = nn.Conv2d(256, n_kp, kernel_size=1)

        for m in self.deconv.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, -2.19)   # start near sigmoid(0.1)

    @staticmethod
    def _deconv_block(cin, cout):
        return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # logits, (B, K, 64, 64)
        return self.head(self.deconv(self.backbone(x)))


def make_target(kps_hm, size=HEATMAP_SIZE, sigma=SIGMA):
    """Gaussian target heatmaps from keypoints already in heatmap coordinates.

    Returns (K, size, size) float32 and a (K,) weight mask. Keypoints outside
    the heatmap get weight 0 -- they have no peak to place, and forcing an
    all-zero target teaches the network to suppress evidence it may actually
    see near the border.
    """
    k = len(kps_hm)
    target = np.zeros((k, size, size), np.float32)
    weight = np.ones(k, np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for i, (x, y) in enumerate(kps_hm):
        if not (0 <= x < size and 0 <= y < size):
            weight[i] = 0.0
            continue
        target[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return target, weight


def decode_heatmaps(heatmaps):
    """Heatmaps -> sub-pixel keypoints in heatmap coordinates.

    Peak by argmax, then a quadratic fit on the log of the three samples along
    each axis. For a gaussian the log is exactly a parabola, so the vertex is
    the true centre; this recovers ~0.1 px rather than the 1 px the argmax alone
    would give.

    Args:  (B, K, H, W) tensor or array of logits
    Returns: coords (B, K, 2), confidence (B, K)
    """
    if isinstance(heatmaps, torch.Tensor):
        hm = torch.sigmoid(heatmaps).detach().cpu().numpy()
    else:
        hm = heatmaps
    B, K, H, W = hm.shape
    coords = np.zeros((B, K, 2), np.float32)
    conf = np.zeros((B, K), np.float32)

    for b in range(B):
        for k in range(K):
            h = hm[b, k]
            idx = int(h.argmax())
            py, px = divmod(idx, W)
            conf[b, k] = h[py, px]
            x, y = float(px), float(py)

            # log-quadratic refinement, guarded against the border and against
            # flat/zero neighbourhoods where the fit is meaningless
            eps = 1e-10
            if 0 < px < W - 1:
                l, c, r = (np.log(h[py, px - 1] + eps), np.log(h[py, px] + eps),
                           np.log(h[py, px + 1] + eps))
                d = l - 2 * c + r
                if d < -1e-6:
                    x += np.clip(0.5 * (l - r) / d, -0.5, 0.5)
            if 0 < py < H - 1:
                u, c, dn = (np.log(h[py - 1, px] + eps), np.log(h[py, px] + eps),
                            np.log(h[py + 1, px] + eps))
                d = u - 2 * c + dn
                if d < -1e-6:
                    y += np.clip(0.5 * (u - dn) / d, -0.5, 0.5)
            coords[b, k] = (x, y)
    return coords, conf


class HeatmapLoss(nn.Module):
    """MSE on sigmoid heatmaps, masked by keypoint validity.

    Plain MSE over a 64x64 map where the target is ~99% zeros lets the
    background dominate the gradient. `pos_weight` upweights the region near the
    target peak so the network is pushed to produce a peak rather than to
    produce nothing everywhere.
    """

    def __init__(self, pos_weight=8.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred_logits, target, weight):
        pred = torch.sigmoid(pred_logits)
        w = 1.0 + self.pos_weight * target          # (B,K,H,W)
        se = (pred - target) ** 2 * w
        per_kp = se.flatten(2).mean(2)              # (B,K)
        masked = per_kp * weight
        denom = weight.sum().clamp(min=1.0)
        return masked.sum() / denom
