"""Count trick-or-treaters from tracks: each person who walks into the yard once.

Reads output/tracks.csv and the crops from track.py. For every track that
spends at least --min-seconds in the yard:
  - skip it if it's on a prop (zone.json ignore_boxes) or is one of the hosts
    (the model residents.py saved, applied to the track's crops),
  - optionally (--join) join it to an earlier track that vanished a few seconds
    before, close by, and looks the same. Off by default: checked against hand
    counts it merged different people far more often than it fixed splits,
  - estimate how likely it is to be a kid (the model ages.py saved).

Writes output/people.csv (one row per visitor), output/groups.csv (people who
arrived together) and output/per_15_minutes_tracked.png.

Usage:
    uv run count_tracks.py
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from residents import on_props


def embed_tracks(crop_dir: Path, tracks: list[str], clip: str, device: str) -> dict[str, np.ndarray]:
    """Mean CLIP embedding of each track's saved crops."""
    arch, weights = clip.split(":")
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=weights)
    model = model.to(device).eval()
    wanted = set(tracks)
    flat = [(f.stem.rsplit("_", 1)[0], f) for f in sorted(crop_dir.glob("*.jpg"))]
    flat = [(t, f) for t, f in flat if t in wanted]
    out: dict[str, list] = {}
    for i in range(0, len(flat), 64):
        batch = flat[i : i + 64]
        x = torch.stack([preprocess(Image.open(f).convert("RGB")) for _, f in batch]).to(device)
        with torch.no_grad():
            e = torch.nn.functional.normalize(model.encode_image(x), dim=-1).cpu().numpy()
        for (t, _), v in zip(batch, e):
            out.setdefault(t, []).append(v)
    return {t: np.array(v) for t, v in out.items()}


def train_host_model(out: Path, frames_dir: Path, clip: str, names: list, device: str):
    """Learn host vs visitor on tightly cropped people, like the tracker's crops.

    Training examples are the people residents.py was surest about: confidently
    tagged hosts, and visitors. Cached in output/host_model_tracks.pkl.
    """
    cache = out / "host_model_tracks.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    df = pd.read_csv(out / "labeled.csv", dtype={"frame": str})
    rng = np.random.default_rng(0)
    parts = []
    for i, name in enumerate(names):
        h = df[(df["who"] == name) & (df["score"] >= 0.75)]
        parts.append(h.iloc[rng.choice(len(h), min(800, len(h)), replace=False)].assign(y=i))
    v = df[df["who"] == "visitor"]
    parts.append(v.iloc[rng.choice(len(v), min(2000, len(v)), replace=False)].assign(y=len(names)))
    train = pd.concat(parts)

    arch, weights = clip.split(":")
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=weights)
    model = model.to(device).eval()
    emb = []
    rows = list(train.itertuples())
    for i in range(0, len(rows), 64):
        imgs = [Image.open(frames_dir / f"{r.frame}.jpg").convert("RGB").crop((r.x1, r.y1, r.x2, r.y2)) for r in rows[i : i + 64]]
        with torch.no_grad():
            e = model.encode_image(torch.stack([preprocess(im) for im in imgs]).to(device))
        emb.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    clf = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced").fit(np.concatenate(emb), train["y"])
    cache.write_bytes(pickle.dumps(clf))
    return clf


def merge_fragments(s: pd.DataFrame, emb: dict, max_gap: float, max_dist: float, min_sim: float) -> pd.Series:
    """Map each track to a person id, joining tracks that are one person seen in pieces.

    A later track continues an earlier one if it starts within `max_gap` seconds of
    the earlier one ending, `max_dist` px away, and looks the same (CLIP similarity).
    """
    s = s.sort_values("first")
    ids = list(s.index)
    first = s["first"].astype("int64").to_numpy() / 1e9
    last = s["last"].astype("int64").to_numpy() / 1e9
    fx, fy = s["first_x"].to_numpy(), s["first_y"].to_numpy()
    lx, ly = s["last_x"].to_numpy(), s["last_y"].to_numpy()
    mean = {}
    for t in ids:
        if t in emb:
            m = emb[t].mean(0)
            mean[t] = m / np.linalg.norm(m)
    by_end = np.argsort(last)
    ends = last[by_end]
    person = {t: t for t in ids}
    continued = set()
    for j, b in enumerate(ids):
        if b not in mean:
            continue
        # Only tracks that ended in the `max_gap` seconds before this one started.
        lo, hi = np.searchsorted(ends, first[j] - max_gap), np.searchsorted(ends, first[j], side="left")
        best, best_sim = None, min_sim
        for i in by_end[lo:hi]:
            a = ids[i]
            if a == b or a in continued or a not in mean:
                continue
            if np.hypot(fx[j] - lx[i], fy[j] - ly[i]) > max_dist:
                continue
            sim = float(mean[a] @ mean[b])
            if sim >= best_sim:
                best, best_sim = a, sim
        if best is not None:
            person[b] = best
            continued.add(best)

    def root(t):
        seen = {t}
        while person[t] != t and person[t] not in seen:
            t = person[t]
            seen.add(t)
        return t
    return pd.Series({t: root(t) for t in ids})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--zone", type=Path, default=Path("zone.json"))
    p.add_argument("--min-seconds", type=float, default=2.0, help="time in the yard to count as a visit")
    p.add_argument("--host-prob", type=float, default=0.5)
    p.add_argument("--join", action="store_true",
                   help="join track pieces that look like one person (off: in testing it merged different people)")
    p.add_argument("--group-gap", type=float, default=20, help="seconds between arrivals that splits groups")
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    zone = json.loads(args.zone.read_text())
    t = pd.read_csv(args.out / "tracks.csv", parse_dates=["time"])
    t = t[t["in_zone"] == 1]
    t = t[~on_props(t, zone.get("ignore_boxes", []), zone.get("ignore_iou", 0.5))]
    t = t.sort_values("time")
    t["fx"] = (t["x1"] + t["x2"]) / 2

    g = t.groupby("track")
    s = pd.DataFrame({
        "first": g["time"].min(), "last": g["time"].max(), "sightings": g.size(),
        "first_x": g["fx"].first(), "first_y": g["y2"].first(),
        "last_x": g["fx"].last(), "last_y": g["y2"].last(),
    })
    s["seconds"] = (s["last"] - s["first"]).dt.total_seconds()

    print(f"{len(s)} tracks in the yard; embedding their crops...", flush=True)
    hosts = pickle.loads((args.out / "host_model.pkl").read_bytes())
    ages = pickle.loads((args.out / "age_model.pkl").read_bytes())
    emb = embed_tracks(args.out / "track_crops", list(s.index), hosts["clip"], args.device)

    print("recognizing hosts and kids...", flush=True)
    # Hosts: average host-likelihood over the track's crops.
    n_hosts = len(hosts["names"])
    host_clf = train_host_model(args.out, args.frames, hosts["clip"], hosts["names"], args.device)
    s["p_host"] = [host_clf.predict_proba(emb[k])[:, :n_hosts].sum(1).mean() if k in emb else 0 for k in s.index]

    # Kid-likelihood: CLIP's child-vs-adult score on the crops plus height for where they stand.
    a, b, c = ages["height_plane"]
    h = (t["y2"] - t["y1"]) / (a + b * t["fx"] + c * t["y2"])
    ratio = h.groupby(t["track"]).median()
    p_kid = []
    for k in s.index:
        if k not in emb:
            p_kid.append(np.nan)
            continue
        pr = torch.softmax(torch.from_numpy(100 * emb[k] @ ages["text"].T), dim=1).numpy()
        ck = np.clip(pr[:, : ages["n_kid_prompts"]].sum(1).mean(), 1e-4, 1 - 1e-4)
        p_kid.append(ages["clf"].predict_proba([[np.log(ck / (1 - ck)), ratio[k]]])[0, 1])
    s["p_kid"] = p_kid

    visitors = s[(s["p_host"] < args.host_prob)]
    print(f"joining pieces of {len(visitors)} visitor tracks...", flush=True)
    if args.join:
        s["person"] = merge_fragments(visitors, emb, max_gap=8, max_dist=150, min_sim=0.88).reindex(s.index)
    else:
        s["person"] = pd.Series(visitors.index, index=visitors.index).reindex(s.index)

    # One row per person: from their first sighting to their last, across joined tracks.
    v = s[s["person"].notna()].assign(track_id=lambda d: d.index)
    people = v.groupby("person").agg(
        track_ids=("track_id", " ".join),
        first=("first", "min"), last=("last", "max"), tracks=("sightings", "size"),
        sightings=("sightings", "sum"), p_kid=("p_kid", "mean"),
    )
    people["seconds"] = (people["last"] - people["first"]).dt.total_seconds()
    people = people[people["seconds"] >= args.min_seconds].sort_values("first").reset_index()
    people["kid"] = people["p_kid"] >= 0.5

    # Groups: people arriving within --group-gap seconds of the previous arrival.
    people["group"] = (people["first"].diff().dt.total_seconds() > args.group_gap).cumsum() + 1
    groups = people.groupby("group").agg(arrived=("first", "min"), left=("last", "max"),
                                         people=("person", "size"), kids=("p_kid", "sum"))
    groups["kids"] = groups["kids"].round().astype(int)

    people.to_csv(args.out / "people.csv", index=False)
    groups.to_csv(args.out / "groups.csv")

    per = people.set_index("first")["p_kid"].resample("15min").agg(["size", "sum"])
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 15 / 1440 * 0.85
    ax.bar(per.index, per["sum"], width=width, align="edge", color="#e8731a", label="kids")
    ax.bar(per.index, per["size"] - per["sum"], bottom=per["sum"], width=width, align="edge", color="#8a8f98", label="adults")
    ax.legend(frameon=False)
    ax.set_title(f"Trick-or-treaters per 15 minutes, tracked ({len(people)} people)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%-I:%M"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "per_15_minutes_tracked.png", dpi=150)

    n_tracks = len(s)
    print(f"{n_tracks} tracks in the yard: {int((s['p_host'] >= args.host_prob).sum())} hosts, "
          f"{len(v)} visitor tracks -> {v['person'].nunique()} after joining pieces -> "
          f"{len(people)} people in the yard {args.min_seconds:g}s or more")
    print(f"about {round(people['p_kid'].sum())} kids and {len(people) - round(people['p_kid'].sum())} adults, "
          f"in {len(groups)} groups")


if __name__ == "__main__":
    main()
