"""Estimate how likely each visitor is to be a kid (vs a parent walking up with them).

Two weak signals, combined:
  - CLIP: does the crop look more like "a child" or "an adult"?
  - Height for where they stand: the camera is low and the path runs away
    from it, so the same person looks shorter at the far end. We learn the
    typical adult height at each spot (the tall end of everyone seen there)
    and compare each person to it.

The combination is fit to hand-labeled examples in age_labels.csv (frame and
box of each person, labeled kid or adult). Adds a p_kid column to
output/labeled.csv, and writes crop sheets to output/review/ for checking.

Usage:
    uv run ages.py
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from residents import crop_sheet

KID_PROMPTS = ["a photo of a child", "a photo of a little kid in a Halloween costume", "a photo of a toddler"]
ADULT_PROMPTS = ["a photo of an adult", "a photo of a parent", "a photo of a grown man",
                 "a photo of a grown woman", "a photo of a teenager"]


def adult_height_model(df: pd.DataFrame) -> np.ndarray:
    """Fit typical adult box height as a plane over foot position: h = a + b*x + c*y."""
    fx = (df["x1"] + df["x2"]) / 2
    cell = list(zip((fx // 80).astype(int), (df["y2"] // 60).astype(int)))
    g = df.assign(h=df["y2"] - df["y1"], cell=cell).groupby("cell")["h"].agg(n="count", q=lambda s: s.quantile(0.85))
    g = g[g["n"] >= 50]
    cx = np.array([c[0] * 80 + 40 for c in g.index])
    cy = np.array([c[1] * 60 + 30 for c in g.index])
    w = np.sqrt(g["n"].to_numpy())
    A = np.c_[np.ones(len(g)), cx, cy]
    return np.linalg.lstsq(A * w[:, None], g["q"].to_numpy() * w, rcond=None)[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labeled", type=Path, default=Path("output/labeled.csv"))
    p.add_argument("--labels", type=Path, default=Path("age_labels.csv"))
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--clip", default="ViT-B-16:laion2b_s34b_b88k")
    args = p.parse_args()

    df = pd.read_csv(args.labeled, dtype={"frame": str})
    arch, weights = args.clip.split(":")
    emb = np.load(args.out / f"embeddings_{arch}.npy")
    assert len(emb) == len(df), "embeddings don't match labeled.csv; rerun residents.py"

    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=weights)
    tokenizer = open_clip.get_tokenizer(arch)
    with torch.no_grad():
        t = model.encode_text(tokenizer(KID_PROMPTS + ADULT_PROMPTS))
    t = torch.nn.functional.normalize(t, dim=-1).numpy()
    probs = torch.softmax(torch.from_numpy(100 * emb @ t.T), dim=1).numpy()
    clip_kid = np.clip(probs[:, : len(KID_PROMPTS)].sum(1), 1e-4, 1 - 1e-4)

    visitors = df["who"] == "visitor"
    a, b, c = adult_height_model(df[visitors])
    height_ratio = (df["y2"] - df["y1"]) / (a + b * (df["x1"] + df["x2"]) / 2 + c * df["y2"])
    X = np.c_[np.log(clip_kid / (1 - clip_kid)), height_ratio]

    labels = pd.read_csv(args.labels, dtype={"frame": str})
    key = ["frame", "x1", "y1", "x2", "y2"]
    rows = df.reset_index().merge(labels, on=key)
    clf = LogisticRegression().fit(X[rows["index"]], rows["age"] == "kid")
    print(f"fit on {len(rows)} labeled people ({(rows['age'] == 'kid').sum()} kids)")

    # Saved so count_tracks.py can estimate kids among its tracked people.
    (args.out / "age_model.pkl").write_bytes(pickle.dumps(
        {"clf": clf, "height_plane": (a, b, c), "text": t, "n_kid_prompts": len(KID_PROMPTS), "clip": args.clip}))
    df["p_kid"] = np.where(visitors, clf.predict_proba(X)[:, 1], np.nan).round(3)
    df.to_csv(args.labeled, index=False)

    v = df[visitors]
    print(f"average p_kid over visitor detections: {v['p_kid'].mean():.2f}")
    review = args.out / "review"
    crop_sheet(v[v["p_kid"] >= 0.7].assign(score=v["p_kid"]), args.frames, review / "ages_kids.jpg",
               "Most likely kids (p_kid >= 0.7)")
    crop_sheet(v[v["p_kid"] <= 0.3].assign(score=v["p_kid"]), args.frames, review / "ages_adults.jpg",
               "Most likely adults (p_kid <= 0.3)")


if __name__ == "__main__":
    main()
