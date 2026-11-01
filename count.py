"""Turn labeled detections into a list of trick-or-treater visits.

A visit is a run of frames with visitors in the yard, allowing short gaps
(people step behind the tree, or the detector misses a frame). A visit's size
is the most visitors seen at once, counted only if that number shows up in at
least --confirm frames, so one bad frame can't inflate a group.

Writes to output/:
  visits.csv           one row per visit
  per_15_minutes.png   chart of trick-or-treaters over time
  review.html          every visit with boxes drawn, to check the counts by eye

Usage:
    uv run count.py
"""

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

COLORS = {"visitor": (255, 40, 40), "decoration": (150, 150, 150)}  # hosts are blue
HOST_COLOR = (40, 140, 255)


def arrivals(c: pd.Series, hold: int) -> int:
    """Sum of upward steps in the visitor count, ignoring blips shorter than `hold` frames."""
    smooth = c.rolling(5, center=True, min_periods=1).median().round().astype(int).to_numpy()
    level, total, i = 0, 0, 0
    while i < len(smooth):
        if smooth[i] != level:
            run = smooth[i : i + hold]
            # Only move to a new level if it holds (same side of the old level) for `hold` frames.
            if len(run) == hold and ((run > level).all() or (run < level).all()):
                new = int(np.median(run))
                total += max(0, new - level)
                level = new
                i += hold
                continue
        i += 1
    return total


def find_visits(counts: pd.Series, max_gap: int, min_frames: int, confirm: int) -> list[dict]:
    """Group frames with visitors into visits. `counts` is visitors per frame, indexed by time."""
    active = counts[counts > 0]
    runs: list[list] = []
    for t in active.index:
        if runs and (t - runs[-1][-1]).total_seconds() <= max_gap:
            runs[-1].append(t)
        else:
            runs.append([t])

    out = []
    for times in runs:
        c = counts.loc[times[0] : times[-1]]
        seen = int((c > 0).sum())
        if seen < min_frames:
            continue
        # Largest k such that at least `confirm` frames show k or more visitors.
        need = min(confirm, seen)
        peak = max(k for k in range(1, int(c.max()) + 1) if (c >= k).sum() >= need)
        # Groups overlap when it's busy, so count arrivals rather than the peak:
        # every time the number in the yard steps up and stays up, people arrived.
        people = max(peak, arrivals(c, hold=3))
        # Best frame to show: one at the peak, from the middle of them.
        at_size = c[c == peak]
        best = at_size.index[len(at_size) // 2] if len(at_size) else c.idxmax()
        out.append(
            {
                "start": times[0],
                "end": times[-1],
                "seconds": int((times[-1] - times[0]).total_seconds()) + 1,
                "people": people,
                "frames_seen": seen,
                "best_frame": best,
            }
        )
    return out


def annotate(frame_path: Path, boxes: pd.DataFrame, zone: list, label: str | None = None) -> Image.Image:
    img = Image.open(frame_path).convert("RGB")
    d = ImageDraw.Draw(img)
    d.polygon([tuple(p) for p in zone], outline=(255, 200, 0), width=2)
    for b in boxes.itertuples():
        color = COLORS.get(b.who, HOST_COLOR)
        d.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=4 if b.who == "visitor" else 3)
        if b.who != "visitor":
            d.text((b.x1 + 3, b.y1 + 2), b.who, fill=color, font=ImageFont.load_default(size=16))
    if label:
        d.rectangle([0, img.height - 34, img.width, img.height], fill=(0, 0, 0))
        d.text((8, img.height - 30), label, fill=(255, 255, 255), font=ImageFont.load_default(size=24))
    return img


def strip(frames: list[Path], dets: pd.DataFrame, zone: list, counts: pd.Series, n: int = 8) -> Image.Image:
    """A row of small frames spread across the visit, each labeled with its visitor count."""
    pick = [frames[round(i * (len(frames) - 1) / max(1, n - 1))] for i in range(min(n, len(frames)))]
    pick = list(dict.fromkeys(pick))
    tw = 320
    tiles = []
    for f in pick:
        t = pd.to_datetime(f.stem, format="%Y%m%d_%H%M%S")
        img = annotate(f, dets[dets["frame"] == f.stem], zone)
        img = img.resize((tw, round(img.height * tw / img.width)))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, tw, 22], fill=(0, 0, 0))
        d.text((4, 2), f"{t:%H:%M:%S}  {counts.get(t, 0)} visitors", fill=(255, 255, 255), font=ImageFont.load_default(size=16))
        tiles.append(img)
    out = Image.new("RGB", (tw * len(tiles), tiles[0].height), (20, 20, 20))
    for i, t in enumerate(tiles):
        out.paste(t, (i * tw, 0))
    return out


def chart(visits: pd.DataFrame, out: Path, bin_minutes: int) -> None:
    cols = ["kids", "adults"] if "kids" in visits else ["people"]
    per_bin = visits.set_index("start")[cols].resample(f"{bin_minutes}min").sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    width = bin_minutes / 1440 * 0.85
    bottom = None
    for col, color in zip(cols, ["#e8731a", "#8a8f98"]):
        ax.bar(per_bin.index, per_bin[col], width=width, align="edge", color=color, bottom=bottom, label=col)
        bottom = per_bin[col] if bottom is None else bottom + per_bin[col]
    if len(cols) > 1:
        ax.legend(frameon=False)
    ax.set_title(f"Trick-or-treaters per {bin_minutes} minutes ({visits['people'].sum()} total, {len(visits)} groups)")
    ax.set_ylabel("people")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%-I:%M"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def review_page(visits: pd.DataFrame, out: Path, chart_name: str) -> None:
    rows = []
    for v in visits.itertuples():
        n = v.Index + 1
        rows.append(
            f"""<section id="v{n}">
  <h2>#{n} &middot; {v.start:%-I:%M:%S %p} &middot; <b>{v.people}</b> {"person" if v.people == 1 else "people"}{f" (~{v.kids} kids)" if hasattr(v, "kids") else ""}
  <small>{v.seconds}s in the yard, seen in {v.frames_seen} frames</small></h2>
  <img class="best" src="visits/visit_{n:03d}.jpg" loading="lazy">
  <img class="strip" src="visits/strip_{n:03d}.jpg" loading="lazy">
</section>"""
        )
    total = int(visits["people"].sum())
    out.write_text(
        f"""<!doctype html><meta charset="utf-8"><title>Trick-or-treater review</title>
<style>
  body {{ font: 15px system-ui, sans-serif; background: #111; color: #eee; max-width: 1300px; margin: 0 auto; padding: 16px; }}
  img {{ max-width: 100%; display: block; }}
  .best {{ margin: 6px 0; }}
  section {{ border-top: 1px solid #333; padding: 12px 0 20px; }}
  h2 {{ font-size: 18px; margin: 4px 0; }} small {{ color: #999; font-weight: normal; margin-left: 8px; }}
  .key span {{ display: inline-block; margin-right: 16px; }}
</style>
<h1>{total} trick-or-treaters in {len(visits)} groups</h1>
<p class="key"><span style="color:#ff2828">&#9632; counted visitor</span><span style="color:#288cff">&#9632; host (not counted)</span>
<span style="color:#969696">&#9632; decoration (not counted)</span><span style="color:#ffc800">&#9633; yard zone</span></p>
<p>Each group's size is the most visitors seen at once in at least two frames. The big image is a frame at that size;
the strip below shows frames from across the visit. Wrong count? Note the # and time.</p>
<img src="{html.escape(chart_name)}">
{"".join(rows)}
"""
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detections", type=Path, default=Path("output/detections.csv"))
    p.add_argument("--labeled", type=Path, default=Path("output/labeled.csv"))
    p.add_argument("--zone", type=Path, default=Path("zone.json"))
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--max-gap", type=int, default=20, help="seconds with no visitors that end a visit")
    p.add_argument("--min-frames", type=int, default=3, help="ignore visits seen in fewer frames")
    p.add_argument("--confirm", type=int, default=2, help="frames a group size must appear in")
    p.add_argument("--bin", type=int, default=15, help="chart bin size in minutes")
    args = p.parse_args()

    zone = json.loads(args.zone.read_text())["polygon"]
    all_frames = pd.read_csv(args.detections, dtype={"frame": str}, usecols=["frame"])["frame"].unique()
    times = pd.to_datetime(pd.Series(sorted(all_frames)), format="%Y%m%d_%H%M%S")
    dets = pd.read_csv(args.labeled, dtype={"frame": str})
    visitors = dets[dets["who"] == "visitor"]
    counts = visitors.groupby("frame").size().reindex(sorted(all_frames), fill_value=0)
    counts.index = times.values

    visits = pd.DataFrame(find_visits(counts, args.max_gap, args.min_frames, args.confirm))
    if visits.empty:
        print("No visits found.")
        return
    if "p_kid" in visitors:
        # Kids in a visit: its size times the average kid-likelihood of everyone seen in it.
        vt = pd.to_datetime(visitors["frame"], format="%Y%m%d_%H%M%S")
        share = [visitors.loc[(vt >= v.start) & (vt <= v.end), "p_kid"].mean() for v in visits.itertuples()]
        visits["kids"] = (visits["people"] * pd.Series(share)).round().astype(int)
        visits["adults"] = visits["people"] - visits["kids"]

    args.out.mkdir(parents=True, exist_ok=True)
    visits.to_csv(args.out / "visits.csv", index_label="visit")
    chart_name = f"per_{args.bin}_minutes.png"
    chart(visits, args.out / chart_name, args.bin)

    vdir = args.out / "visits"
    vdir.mkdir(exist_ok=True)
    for v in visits.itertuples():
        n = v.Index + 1
        best = v.best_frame.strftime("%Y%m%d_%H%M%S")
        label = f"#{n}  {v.start:%-I:%M:%S %p}  {v.people} counted"
        annotate(args.frames / f"{best}.jpg", dets[dets["frame"] == best], zone, label).save(vdir / f"visit_{n:03d}.jpg", quality=85)
        span = [args.frames / f"{t:%Y%m%d_%H%M%S}.jpg" for t in counts.loc[v.start : v.end].index]
        strip([f for f in span if f.exists()], dets, zone, counts).save(vdir / f"strip_{n:03d}.jpg", quality=80)
    review_page(visits, args.out / "review.html", chart_name)

    print(f"{len(visits)} groups, {visits['people'].sum()} people")
    if "kids" in visits:
        print(f"about {visits['kids'].sum()} kids and {visits['adults'].sum()} adults")
    print(visits["people"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
