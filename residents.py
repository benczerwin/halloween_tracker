"""Tag the people who are there all afternoon (the hosts) so they aren't counted.

Each host is described in residents.json by a short text description. CLIP
finds the detections that best match each description, averages them into an
appearance profile, then in every frame marks the one detection most like
that profile (if it's close enough) as that host. Everyone else is a visitor.

Writes output/labeled.csv (detections in the zone with a "who" column) and a
crop sheet per host in output/review/ so you can check who was excluded.

Usage:
    uv run residents.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from matplotlib.path import Path as Polygon
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression

VISITOR_PROMPTS = [
    "a child in a Halloween costume",
    "a person in an orange costume",
    "a person in an orange jumpsuit",
    "a parent walking with kids",
    "a person",
    "a stack of white pumpkins",
]


def in_zone(df: pd.DataFrame, zone: dict) -> pd.DataFrame:
    df = df.dropna(subset=["x1"])
    df = df[df["conf"] >= zone.get("min_conf", 0.0)]
    df = df[(df["y2"] - df["y1"]) >= zone.get("min_height", 0)]
    feet = list(zip((df["x1"] + df["x2"]) / 2, df["y2"]))
    return df[Polygon(zone["polygon"]).contains_points(feet)].reset_index(drop=True)


def on_props(df: pd.DataFrame, props: list, min_iou: float) -> np.ndarray:
    """True for detections that are really a decoration: they mostly overlap a prop's box."""
    boxes = df[["x1", "y1", "x2", "y2"]].to_numpy()
    return np.array([any(iou(b, prop) >= min_iou for prop in props) for b in boxes], dtype=bool)


def crop(frames_dir: Path, r) -> Image.Image:
    img = Image.open(frames_dir / f"{r.frame}.jpg").convert("RGB")
    pad_x, pad_y = (r.x2 - r.x1) * 0.1, (r.y2 - r.y1) * 0.05
    return img.crop((r.x1 - pad_x, r.y1 - pad_y, r.x2 + pad_x, r.y2 + pad_y))


def embed_crops(df: pd.DataFrame, frames_dir: Path, model, preprocess, device: str) -> np.ndarray:
    out = []
    for i in range(0, len(df), 64):
        batch = torch.stack([preprocess(crop(frames_dir, r)) for r in df.iloc[i : i + 64].itertuples()])
        with torch.no_grad():
            e = model.encode_image(batch.to(device))
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        print(f"embedded {min(i + 64, len(df))}/{len(df)}", flush=True)
    return np.concatenate(out)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def link_tracks(df: pd.DataFrame, min_iou: float = 0.3, max_gap: int = 3) -> np.ndarray:
    """Link each person's boxes from second to second by overlap. Returns a track id per row."""
    times = pd.to_datetime(df["frame"], format="%Y%m%d_%H%M%S")
    track = np.full(len(df), -1)
    active: dict[int, tuple] = {}  # track id -> (last time, last box)
    next_id = 0
    for frame, idx in sorted(df.groupby("frame").indices.items()):
        t = times.iloc[idx[0]]
        active = {k: v for k, v in active.items() if (t - v[0]).total_seconds() <= max_gap}
        boxes = {r: tuple(df.iloc[r][["x1", "y1", "x2", "y2"]]) for r in idx}
        pairs = sorted(((iou(boxes[r], v[1]), r, k) for r in idx for k, v in active.items()), reverse=True)
        taken_r, taken_k = set(), set()
        for o, r, k in pairs:
            if o < min_iou:
                break
            if r in taken_r or k in taken_k:
                continue
            track[r] = k
            taken_r.add(r)
            taken_k.add(k)
        for r in idx:
            if track[r] < 0:
                track[r] = next_id
                next_id += 1
            active[track[r]] = (t, boxes[r])
    return track


def crop_sheet(df: pd.DataFrame, frames_dir: Path, out: Path, title: str, n: int = 60) -> None:
    if df.empty:
        return
    sample = df.iloc[np.linspace(0, len(df) - 1, min(n, len(df))).astype(int)]
    tw, th, cols = 100, 200, 12
    rows = -(-len(sample) // cols)
    sheet = Image.new("RGB", (cols * tw, rows * th + 30), (20, 20, 20))
    ImageDraw.Draw(sheet).text((6, 8), title, fill=(255, 255, 255))
    for i, r in enumerate(sample.itertuples()):
        tile = crop(frames_dir, r).resize((tw, th))
        d = ImageDraw.Draw(tile)
        d.rectangle([0, 0, tw, 14], fill=(0, 0, 0))
        d.text((2, 1), f"{r.frame[9:11]}:{r.frame[11:13]}:{r.frame[13:15]} {r.score:.2f}", fill=(255, 255, 0))
        sheet.paste(tile, ((i % cols) * tw, 30 + (i // cols) * th))
    sheet.save(out, quality=85)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detections", type=Path, default=Path("output/detections.csv"))
    p.add_argument("--zone", type=Path, default=Path("zone.json"))
    p.add_argument("--residents", type=Path, default=Path("residents.json"))
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--device", default="mps")
    p.add_argument("--clip", default="ViT-B-16:laion2b_s34b_b88k", help="open_clip model:pretrained")
    args = p.parse_args()

    zone = json.loads(args.zone.read_text())
    residents = json.loads(args.residents.read_text())
    df = in_zone(pd.read_csv(args.detections, dtype={"frame": str}), zone)
    ignored = on_props(df, zone.get("ignore_boxes", []), zone.get("ignore_iou", 0.5))

    arch, weights = args.clip.split(":")
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=weights)
    model = model.to(args.device).eval()
    tokenizer = open_clip.get_tokenizer(arch)

    cache = args.out / f"embeddings_{arch}.npy"
    emb = np.load(cache) if cache.exists() and len(np.load(cache)) == len(df) else None
    if emb is None:
        emb = embed_crops(df, args.frames, model, preprocess, args.device)
        np.save(cache, emb)

    # Zero-shot: which description does each crop match best?
    names = list(residents["hosts"])
    prompts = [residents["hosts"][n]["description"] for n in names] + VISITOR_PROMPTS
    with torch.no_grad():
        t = model.encode_text(tokenizer(prompts).to(args.device))
    t = torch.nn.functional.normalize(t, dim=-1).cpu().numpy()
    probs = torch.softmax(torch.from_numpy(100 * emb @ t.T), dim=1).numpy()

    review = args.out / "review"
    review.mkdir(parents=True, exist_ok=True)

    # Training examples: the crops CLIP is surest match each host's description,
    # and for visitors the crops that look least like any host.
    labels = np.full(len(df), -1)
    host_score = probs[:, : len(names)].max(1)
    labels[np.argsort(host_score)[: residents.get("visitor_seeds", 400)]] = len(names)
    for i, name in enumerate(names):
        seeds = np.argsort(-probs[:, i])[: residents["hosts"][name].get("seeds", 60)]
        labels[seeds] = i
        crop_sheet(df.iloc[seeds].assign(score=probs[seeds, i]).sort_values("frame"), args.frames,
                   review / f"seeds_{name}.jpg", f"Examples used to learn what {name} looks like")

    # Learn from the seeds, then once more adding everything the first pass was sure of.
    clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced")
    for _ in range(2):
        clf.fit(emb[labels >= 0], labels[labels >= 0])
        p = clf.predict_proba(emb)
        sure = (p.max(1) > 0.95) & (labels < 0)
        labels[sure] = p[sure].argmax(1)
    p_single = clf.predict_proba(emb)

    # One frame on its own is a noisy guess. Average each person's guesses over
    # their whole track: the hosts stand around for hours, visitors pass through.
    df["track"] = link_tracks(df)
    p = pd.DataFrame(p_single).groupby(df["track"]).transform("mean").to_numpy()
    weight = residents.get("track_weight", 0.7)
    p = weight * p + (1 - weight) * p_single

    # Which host is which doesn't matter for counting, only host vs visitor. In each
    # frame, the (up to) N most host-like people are the hosts.
    n_hosts = len(names)
    host_p = p[:, :n_hosts].sum(1)
    host_p[ignored] = 0
    min_prob = residents.get("min_prob", 0.5)
    df["who"], df["score"] = "visitor", 0.0
    frames = df.groupby("frame").indices
    tagged = {}

    def tag(rows, bar):
        rows = [r for r in sorted(rows, key=lambda r: -host_p[r]) if host_p[r] >= bar]
        for r in rows:
            df.iat[r, df.columns.get_loc("who")] = names[int(p[r, :n_hosts].argmax())]
            df.iat[r, df.columns.get_loc("score")] = host_p[r]
        return len(rows)

    for frame, idx in frames.items():
        tagged[frame] = tag(sorted(idx, key=lambda r: -host_p[r])[:n_hosts], min_prob)

    # The hosts are there all afternoon. If more hosts were recognized within a few
    # seconds both before and after a frame, the missing ones are probably still
    # there, just seen from an odd angle: tag the most host-like people at a lower bar.
    fill_prob, window = residents.get("fill_prob", 0.3), residents.get("fill_seconds", 10)
    order = sorted(frames)
    times = pd.to_datetime(pd.Series(order), format="%Y%m%d_%H%M%S").to_numpy()
    counts = np.array([tagged[f] for f in order])
    for k, frame in enumerate(order):
        near = np.abs(times - times[k]) <= np.timedelta64(window, "s")
        before = counts[near & (times < times[k])]
        after = counts[near & (times > times[k])]
        if len(before) == 0 or len(after) == 0:
            continue
        missing = min(before.max(), after.max()) - counts[k]
        if missing > 0:
            free = [r for r in frames[frame] if df["who"].iat[r] == "visitor"]
            tag(sorted(free, key=lambda r: -host_p[r])[:missing], fill_prob)

    # Decorations: the detector sometimes sees a prop (the pumpkin stack) as a
    # person. Props never move, so zone.json lists their boxes to ignore.
    df.loc[ignored, "who"] = "decoration"
    crop_sheet(df[ignored], args.frames, review / "decorations.jpg", "Ignored as decorations (not counted)")
    print(f"decorations: {ignored.sum()} detections")

    for i, name in enumerate(names):
        tagged = df[df["who"] == name]
        print(f"{name}: tagged in {len(tagged)} of {df['frame'].nunique()} frames with people")
        crop_sheet(tagged, args.frames, review / f"host_{name}.jpg", f"Tagged as {name} (not counted)")
        # Near misses: counted people who look somewhat like this host. Check none of these are the host.
        near = df.assign(score=p[:, i])[(df["who"] == "visitor") & (p[:, i] > 0.15)]
        crop_sheet(near, args.frames, review / f"near_{name}.jpg", f"Counted as visitors, but somewhat like {name}")

    df.to_csv(args.out / "labeled.csv", index=False)
    print(df["who"].value_counts().to_string())


if __name__ == "__main__":
    main()
