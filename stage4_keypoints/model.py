import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

N_KP = 8
INPUT_SIZE = 256
HEATMAP_SIZE = 64    # default; --heatmap 128 adds a fourth deconv
SIGMA = 2.0          # gaussian std, in heatmap pixels, AT HEATMAP_SIZE=64


def sigma_for(heatmap_size):
    """Sigma must scale with heatmap resolution.

    Sigma is expressed in heatmap pixels, so holding it fixed while doubling
    the canvas shrinks the target's share of the image 4x: at 64 a sigma-2
    gaussian covers 0.51 % of the map, at 128 only 0.13 %. The background term
    in the loss then dominates 4x harder and the network collapses to
    predicting zero everywhere -- observed directly as a run where confidence
    never moved from its initialisation value of sigmoid(-2.19) = 0.10 and
    pixel error stayed at 226 px for ten epochs.

    Scaling sigma linearly keeps the target the same physical width in image
    pixels (20 px either way) and the positive fraction roughly constant, while
    still buying the finer grid that was the point of the higher resolution.
    """
    return SIGMA * (heatmap_size / 64.0)

# Precision budget, which is why resolution is worth testing:
#   at 64:  640/64 = 10 image px per heatmap px, sigma 2.0 = 20 image px
#   at 128: 640/128 = 5 image px per heatmap px, sigma 2.0 = 10 image px
# The first run measured 24.5 image px error = 2.45 heatmap px = 1.22 sigma.
# If the network localises to a fixed fraction of sigma, doubling resolution
# halves the error. If the limit is learning rather than representation, it
# buys nothing. One run distinguishes these.


class KeypointNet(nn.Module):
    def __init__(self, n_kp=N_KP, pretrained=True, heatmap_size=HEATMAP_SIZE):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        # drop avgpool and fc; keep the conv trunk
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.heatmap_size = heatmap_size

        # ResNet-18 at 256 input gives 8x8. Each deconv doubles.
        if heatmap_size == 64:
            chans = [(512, 256), (256, 256), (256, 256)]      # 8->16->32->64
        elif heatmap_size == 128:
            # Narrower late layers: a 128x128 map at 256 channels is ~1 GB of
            # activations at batch 32, which does not fit comfortably in 8 GB.
            chans = [(512, 256), (256, 256), (256, 128), (128, 128)]
        else:
            raise ValueError(
                f"heatmap_size must be 64 or 128, got {heatmap_size}")

        self.deconv = nn.Sequential(
            *[self._deconv_block(a, b) for a, b in chans])
        self.head = nn.Conv2d(chans[-1][1], n_kp, kernel_size=1)

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


def make_target(kps_hm, size=HEATMAP_SIZE, sigma=None):
    """Gaussian target heatmaps from keypoints already in heatmap coordinates.

    Returns (K, size, size) float32 and a (K,) weight mask. Keypoints outside
    the heatmap get weight 0 -- they have no peak to place, and forcing an
    all-zero target teaches the network to suppress evidence it may actually
    see near the border.
    """
    sigma = sigma_for(size) if sigma is None else sigma
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
