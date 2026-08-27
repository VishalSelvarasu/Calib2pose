import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DrillKeypointDataset, collate, IMAGE_SIZE
from model import KeypointNet, HeatmapLoss, decode_heatmaps, HEATMAP_SIZE

CKPT_DIR = "checkpoints"


def set_seed(seed, deterministic=False):
    """Seed every source of randomness the training loop touches.

    Without this the augmentation stream, the weight init and the DataLoader
    shuffle all differ between runs, so a reported number like 11.21 mm ADD
    cannot be reproduced by retraining -- only by trusting the checkpoint.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def evaluate(model, loader, device, loss_fn, heatmap_size, max_batches=None):
    model.eval()
    losses, errs, confs = [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            img = batch["image"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            wt = batch["weight"].to(device, non_blocking=True)
            logits = model(img)
            losses.append(loss_fn(logits, tgt, wt).item())

            coords, conf = decode_heatmaps(logits)
            pred_img = coords * (IMAGE_SIZE / heatmap_size)
            gt = batch["kps_img"].numpy()
            w = wt.cpu().numpy()
            d = np.linalg.norm(pred_img - gt, axis=2)          # (B,K)
            errs.append(d[w > 0])
            confs.append(conf[w > 0])
    return (float(np.mean(losses)),
            float(np.concatenate(errs).mean()),
            float(np.median(np.concatenate(errs))),
            float(np.concatenate(confs).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: cap dataset size")
    ap.add_argument("--val-batches", type=int, default=0,
                    help="cap validation batches per epoch (0 = all)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip ImageNet weights (offline testing only -- costs accuracy)")
    ap.add_argument("--heatmap", type=int, default=64, choices=[64, 128],
                    help="heatmap resolution. 128 halves the quantisation but "
                         "roughly doubles memory and epoch time.")
    ap.add_argument("--aug", type=float, default=1.0,
                    help="augmentation strength. 0 disables geometric and "
                         "cutout, reproducing the first run.")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after N epochs with no val improvement (0 = off)")
    ap.add_argument("--tag", default="",
                    help="suffix for the checkpoint dir, to keep runs separate")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds Python, NumPy, Torch, CUDA and the DataLoader "
                         "workers so a run can be reproduced")
    ap.add_argument("--deterministic", action="store_true",
                    help="also force deterministic cuDNN kernels. Slower, and "
                         "some ops have no deterministic implementation.")
    args = ap.parse_args()

    set_seed(args.seed, args.deterministic)

    ckpt_dir = CKPT_DIR + (f"_{args.tag}" if args.tag else "")
    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")

    train_ds = DrillKeypointDataset(args.data, "train", augment=args.aug > 0,
                                    heatmap_size=args.heatmap,
                                    aug_strength=args.aug)
    val_ds = DrillKeypointDataset(args.data, "val", augment=False,
                                  heatmap_size=args.heatmap)
    print(
        f"heatmap {args.heatmap}  aug {args.aug}  seed {args.seed}  ckpts -> {ckpt_dir}/")
    if args.limit:
        train_ds.records = train_ds.records[:args.limit]
        val_ds.records = val_ds.records[:max(args.limit // 4, 8)]
    print(f"train {len(train_ds)}  val {len(val_ds)}")

    pin = device.type == "cuda"
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=collate,
                          pin_memory=pin, drop_last=True,
                          persistent_workers=args.workers > 0,
                          generator=gen, worker_init_fn=worker_init)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate,
                        pin_memory=pin, persistent_workers=args.workers > 0)

    model = KeypointNet(pretrained=not args.no_pretrained,
                        heatmap_size=args.heatmap).to(device)
    loss_fn = HeatmapLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # T_max must match the FULL intended run length. Resuming with a different
    # --epochs restarts the cosine at a different point and the LR jumps.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best, stale = 0, float("inf"), 0
    hist = []
    last_path = os.path.join(ckpt_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("best", float("inf"))
        hist = ck.get("hist", [])
        print(f"resumed from epoch {ck['epoch']} (best val {best:.3f} px)")
        if ck.get("args", {}).get("epochs") != args.epochs:
            print(f"  WARNING: original run used --epochs "
                  f"{ck.get('args', {}).get('epochs')}, now {args.epochs}. "
                  f"The cosine LR schedule will jump. Keep --epochs constant "
                  f"across resumes.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), []
        for bi, batch in enumerate(train_ld):
            img = batch["image"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            wt = batch["weight"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(img), tgt, wt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running.append(loss.item())

            if bi % 50 == 0:
                print(f"  e{epoch} b{bi}/{len(train_ld)} loss {np.mean(running[-50:]):.5f}",
                      flush=True)
        sched.step()

        vl, vmean, vmed, vconf = evaluate(model, val_ld, device, loss_fn,
                                          args.heatmap,
                                          args.val_batches or None)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}  train {np.mean(running):.5f}  val {vl:.5f}  "
              f"px mean {vmean:6.2f}  median {vmed:6.2f}  conf {vconf:.3f}  "
              f"{dt:.0f}s  lr {sched.get_last_lr()[0]:.2e}", flush=True)

        hist.append({"epoch": epoch, "train_loss": float(np.mean(running)),
                     "val_loss": vl, "val_px_mean": vmean, "val_px_median": vmed,
                     "val_conf": vconf, "secs": dt})

        # Update `best` BEFORE writing last.pt. Writing it first stores the
        # pre-update value, so a resumed run starts with a stale (worse)
        # threshold and can overwrite best.pt with an inferior model.
        improved = vmean < best
        if improved:
            best = vmean
            stale = 0
        else:
            stale += 1

        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch, "best": best,
                 "hist": hist, "args": vars(args)}
        torch.save(state, last_path)
        if improved:
            torch.save(state, os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> new best {best:.3f} px")

        with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)

        if args.patience and stale >= args.patience:
            print(f"\nno val improvement for {stale} epochs -- stopping early "
                  f"at epoch {epoch}")
            break

    print(f"\ndone. best val keypoint error {best:.3f} px (image space, 640)")


if __name__ == "__main__":
    main()
