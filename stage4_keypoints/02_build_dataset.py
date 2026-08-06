
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.data, "manifest_*.json")))
    if not shards:
        raise SystemExit(
            f"No manifest shards in {args.data}/. Run 01_render.py first.")

    records, seeds = [], []
    for s in shards:
        with open(s) as f:
            d = json.load(f)
        records.extend(d["records"])
        seeds.append(d["seed"])
        print(f"  {os.path.basename(s)}: {d['n']} images (seed {d['seed']})")

    if len(set(seeds)) != len(seeds):
        print("\n  WARNING: duplicate seeds across shards -- those chunks contain")
        print("  IDENTICAL poses. Re-render with distinct --start-idx values.")

    names = [r["file"] for r in records]
    if len(set(names)) != len(names):
        raise SystemExit("Duplicate filenames across shards. Chunks overlapped; "
                         "check your --start-idx values.")

    img_dir = os.path.join(args.data, "images")
    missing = [n for n in names if not os.path.exists(
        os.path.join(img_dir, n))]
    if missing:
        raise SystemExit(f"{len(missing)} manifest entries have no image file, "
                         f"e.g. {missing[:3]}")

    n = len(records)
    if args.val + args.test >= n:
        raise SystemExit(f"val+test ({args.val+args.test}) >= total ({n}).")

    vf = np.array([r["visible_frac"] for r in records])
    rng = np.random.default_rng(args.seed)

    # Stratify by occlusion bucket so every split sees the same difficulty mix.
    buckets = np.digitize(vf, [0.5, 0.75, 0.95])
    test_idx, val_idx = [], []
    for b in np.unique(buckets):
        idx = np.where(buckets == b)[0]
        rng.shuffle(idx)
        n_t = int(round(args.test * len(idx) / n))
        n_v = int(round(args.val * len(idx) / n))
        test_idx.extend(idx[:n_t])
        val_idx.extend(idx[n_t:n_t + n_v])

    # Shuffle after stratifying. The loop above appends bucket by bucket, so
    # without this the written file is ORDERED BY OCCLUSION -- and anything that
    # reads it sequentially (a figure script taking the first N, a quick
    # sanity-check on a subset) silently samples only the hardest images.
    rng.shuffle(test_idx)
    rng.shuffle(val_idx)

    test_set, val_set = set(test_idx), set(val_idx)
    train_idx = [i for i in range(n) if i not in test_set and i not in val_set]
    rng.shuffle(train_idx)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    print(f"\ntotal {n} images")
    print(f"{'split':<8}{'n':>7}{'mean vis':>11}{'occluded':>11}{'heavy':>9}")
    for name, idx in splits.items():
        v = vf[list(idx)]
        print(f"{name:<8}{len(idx):>7}{v.mean()*100:>10.1f}%"
              f"{(v < 0.95).mean()*100:>10.1f}%{(v < 0.7).mean()*100:>8.1f}%")
        out = {"split": name, "n": len(idx),
               "records": [records[i] for i in idx]}
        with open(os.path.join(args.data, f"{name}.json"), "w") as f:
            json.dump(out, f)

    print(f"\nwrote train.json, val.json, test.json to {args.data}/")
    print("The test split stays untouched until the final evaluation.")


if __name__ == "__main__":
    main()
