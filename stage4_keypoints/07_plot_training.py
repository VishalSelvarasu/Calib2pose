import matplotlib.pyplot as plt
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="tags to plot; default is every checkpoints_* found")
    ap.add_argument("--out", default="results/training_curves.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.runs:
        dirs = [f"checkpoints_{r}" if not r.startswith("checkpoints") else r
                for r in args.runs]
    else:
        dirs = sorted(glob.glob("checkpoints*"))

    runs = []
    for d in dirs:
        p = os.path.join(d, "history.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            h = json.load(f)
        if h:
            runs.append(
                (d.replace("checkpoints_", "").replace("checkpoints", "run1"), h))

    if not runs:
        raise SystemExit("no history.json found in any checkpoints* directory")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colours = plt.cm.tab10.colors

    for i, (name, h) in enumerate(runs):
        c = colours[i % len(colours)]
        ep = [r["epoch"] for r in h]
        ax1.plot(ep, [r["train_loss"] for r in h], color=c, lw=1.6,
                 label=f"{name} train")
        ax1.plot(ep, [r["val_loss"] for r in h], color=c, lw=1.6, ls="--",
                 label=f"{name} val")
        ax2.plot(ep, [r["val_px_mean"]
                 for r in h], color=c, lw=1.8, label=name)

    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("heatmap loss (log)")
    ax1.set_title("loss — solid train, dashed val")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7, ncol=2)

    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val keypoint error (px, image space)")
    ax2.set_title("validation keypoint error")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"plotted {len(runs)} runs: {', '.join(n for n, _ in runs)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
