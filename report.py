"""Build the Haunted Pumpkin Graveyard one-pager from the tracked counts.

Reads output/people.csv, output/groups.csv and output/tracks.csv (from
count_tracks.py and track.py) and writes output/report.html: one
self-contained page with the headline numbers, findings, an arrivals chart
and one example frame with the detection boxes drawn.

The example frame shows real people, so the report stays local (output/ is
gitignored).

Usage:
    uv run report.py
"""

import argparse
import base64
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from residents import on_props

KID, ADULT, HOST, PROP = "#d9731c", "#8c78d8", "#5fd0c0", "#8a8494"  # chart colors validated on the dark surface


def fmt_time(t: pd.Timestamp) -> str:
    return t.strftime("%-I:%M %p").lower()


def example_frame(out: Path, frames: Path, zone: dict, people: pd.DataFrame) -> tuple[str, str, int]:
    """The busiest moment, drawn with boxes: kids, adults, hosts and the ignored prop."""
    t = pd.read_csv(out / "tracks.csv", parse_dates=["time"])
    t = t[t["time"].dt.microsecond == 0]  # whole seconds line up with the saved frames
    visitors = t[(t["in_zone"] == 1) & t["track"].isin(people["track_ids"].explode())]
    busiest = visitors.groupby("time").size().sort_values(ascending=False)
    for when in busiest.index:
        frame = frames / f"{when - pd.Timedelta(seconds=0):%Y%m%d_%H%M%S}.jpg"
        if frame.exists():
            break
    rows = t[t["time"] == when]
    p_kid = people.explode("track_ids").set_index("track_ids")["p_kid"]
    img = Image.open(frame).convert("RGB")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=18)
    d.polygon([tuple(p) for p in zone["polygon"]], outline=(255, 200, 80), width=2)
    props = on_props(rows, zone.get("ignore_boxes", []), zone.get("ignore_iou", 0.5))
    counted = 0
    for r, is_prop in zip(rows.itertuples(), props):
        if not r.in_zone:
            continue
        if is_prop:
            color, label = PROP, "pumpkins (ignored)"
        elif r.track in p_kid.index:
            kid = p_kid[r.track] >= 0.5
            color, label = (KID, "kid") if kid else (ADULT, "adult")
            counted += 1
        else:
            color, label = HOST, "host"
        d.rectangle([r.x1, r.y1, r.x2, r.y2], outline=color, width=4)
        tw = d.textlength(label, font=font)
        d.rectangle([r.x1, r.y1 - 24, r.x1 + tw + 10, r.y1], fill=color)
        d.text((r.x1 + 5, r.y1 - 22), label, fill=(20, 16, 28), font=font)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode(), fmt_time(when), counted


def chart_svg(people: pd.DataFrame) -> tuple[str, str]:
    """Stacked columns: kids and adults arriving per 15 minutes. Returns (svg, table rows)."""
    per = people.set_index("first")["p_kid"].resample("15min").agg(["size", "sum"])
    per["kids"] = per["sum"].round().astype(int)
    per["adults"] = per["size"] - per["kids"]
    W, H, left, bottom, top = 720, 260, 40, 34, 16
    plot_h = H - bottom - top
    vmax = max(10, int(np.ceil(per["size"].max() / 10.0) * 10))
    slot = (W - left - 8) / len(per)
    bw = min(24, slot * 0.6)
    parts = []
    for v in range(0, vmax + 1, vmax // 4 if vmax >= 40 else 10):
        y = top + plot_h * (1 - v / vmax)
        parts.append(f'<line x1="{left}" x2="{W - 8}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="{left - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{v}</text>')
    rows = []
    for i, (when, r) in enumerate(per.iterrows()):
        x = left + i * slot + (slot - bw) / 2
        hk = plot_h * r["kids"] / vmax
        ha = plot_h * r["adults"] / vmax
        base = top + plot_h
        tip = f"{fmt_time(when)}: {int(r['size'])} arrived ({r['kids']} kids, {r['adults']} adults)"
        gap = 2 if hk and ha else 0
        parts.append(f'<g class="col" tabindex="0" data-tip="{html.escape(tip)}">'
                     f'<rect x="{x - 4:.1f}" y="{top}" width="{bw + 8:.1f}" height="{plot_h}" fill="transparent"/>')
        if hk:
            parts.append(f'<path d="{bar_path(x, base, bw, hk, round_top=not ha)}" fill="{KID}"/>')
        if ha:
            parts.append(f'<path d="{bar_path(x, base - hk - gap, bw, ha - (gap if hk else 0), round_top=True)}" fill="{ADULT}"/>')
        parts.append("</g>")
        if when.minute == 0:
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{H - 12}" class="tick" text-anchor="middle">{when:%-I %p}</text>'.lower())
        rows.append(f"<tr><td>{fmt_time(when)}</td><td>{r['kids']}</td><td>{r['adults']}</td><td>{int(r['size'])}</td></tr>")
    parts.append(f'<line x1="{left}" x2="{W - 8}" y1="{top + plot_h}" y2="{top + plot_h}" class="axis"/>')
    svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Arrivals per 15 minutes, kids and adults">'
           + "".join(parts) + "</svg>")
    return svg, "".join(rows)


def bar_path(x: float, base: float, w: float, h: float, round_top: bool) -> str:
    """A column from `base` up `h` px, square at the bottom, 4px rounded top if it's the data end."""
    r = min(4, h / 2, w / 2) if round_top else 0
    y = base - h
    return (f"M{x:.1f},{base:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{base:.1f} Z")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--zone", type=Path, default=Path("zone.json"))
    args = p.parse_args()

    people = pd.read_csv(args.out / "people.csv", parse_dates=["first", "last"])
    people["track_ids"] = people["track_ids"].str.split(" ")
    groups = pd.read_csv(args.out / "groups.csv", parse_dates=["arrived", "left"])
    zone = json.loads(args.zone.read_text())

    total = len(people)
    kids = int(round(people["p_kid"].sum()))
    adults = total - kids
    per15 = people.set_index("first").resample("15min").size()
    peak_at, peak_n = per15.idxmax(), int(per15.max())
    per_min = people.set_index("first").resample("1min").size()
    busiest_min, busiest_n = per_min.idxmax(), int(per_min.max())
    arrivals = people["first"].sort_values()
    gaps = arrivals.diff().dt.total_seconds()
    lull_i = gaps.idxmax()
    lull_min = gaps.max() / 60
    lull_from = arrivals.loc[:lull_i].iloc[-2]
    big = groups.loc[groups["people"].idxmax()]
    linger = people.loc[people["seconds"].idxmax()]
    hours = (arrivals.iloc[-1] - arrivals.iloc[0]).total_seconds() / 3600

    img64, img_time, img_count = example_frame(args.out, args.frames, zone, people)
    svg, table = chart_svg(people)

    findings = [
        ("The witching hour", f"{peak_n} souls crossed the gate between {fmt_time(peak_at)} and "
                              f"{fmt_time(peak_at + pd.Timedelta(minutes=15))}, the busiest quarter-hour of the night."),
        ("Stampede of the undead", f"At {fmt_time(busiest_min)} {busiest_n} visitors arrived in a single minute."),
        ("The endless procession", f"From {fmt_time(big['arrived'])}, {int(big['people'])} visitors filed in without "
                                   f"a single 20-second break."),
        ("Ghost town", f"The longest lull: {lull_min:.0f} minutes without a single visitor, from {fmt_time(lull_from)}."),
        ("A lingering spirit", f"One visitor haunted the yard for {linger['seconds'] / 60:.1f} minutes "
                               f"(arrived {fmt_time(linger['first'])}). Most were gone in "
                               f"{people['seconds'].median():.0f} seconds."),
        ("Chaperone ratio", f"{kids / max(adults, 1):.1f} little goblins for every grown-up spirit walking them up."),
    ]
    finding_html = "".join(f"<li><h3>{html.escape(h)}</h3><p>{html.escape(b)}</p></li>" for h, b in findings)

    page = TEMPLATE.format(
        total=total, kids=kids, adults=adults, groups=len(groups),
        avg_group=total / len(groups), hours=hours,
        first=fmt_time(arrivals.iloc[0]), last=fmt_time(arrivals.iloc[-1]),
        findings=finding_html, chart=svg, table=table, img=img64, img_time=img_time, img_count=img_count,
        KID=KID, ADULT=ADULT, HOST=HOST, PROP=PROP,
    )
    (args.out / "report.html").write_text(page)
    print(f"wrote {args.out / 'report.html'}: {total} people, {kids} kids, {adults} adults")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Haunted Pumpkin Graveyard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Creepster&family=Nunito:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #17121f; --panel: #211a2c; --line: #352b44;
    --ink: #f3ece2; --ink-2: #c4b8cf; --ink-3: #8f839d;
    --pumpkin: #f08a24; --kid: {KID}; --adult: {ADULT};
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.5 Nunito, system-ui, sans-serif; }}
  .sky {{ background: radial-gradient(circle at 82% 110px, #f6e7c1 0 34px, rgba(246,231,193,.18) 36px, transparent 90px),
                       radial-gradient(ellipse at 80% -10%, #3b2a55 0%, transparent 55%), var(--bg); }}
  main {{ max-width: 960px; margin: 0 auto; padding: 32px 16px 48px; }}
  header {{ position: relative; padding: 24px 0 8px; }}
  .kicker {{ color: var(--pumpkin); font-weight: 800; letter-spacing: .14em; text-transform: uppercase; font-size: 13px; }}
  h1 {{ font-family: Creepster, cursive; font-weight: 400; font-size: clamp(44px, 9vw, 84px); line-height: .95;
        margin: 8px 0 12px; color: var(--pumpkin); overflow-wrap: anywhere; text-shadow: 0 0 24px rgba(240,138,36,.35); }}
  .sub {{ color: var(--ink-2); max-width: 620px; margin: 0; }}
  .graves {{ display: block; width: 100%; height: 70px; margin-top: 12px; }}
  .stats {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 8px 0 28px; }}
  .stat {{ min-width: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 14px 14px 6px 6px; padding: 14px 16px; }}
  .stat .n {{ font-size: 40px; font-weight: 800; line-height: 1.1; }}
  .stat .l {{ color: var(--ink-2); font-size: 14px; }}
  .stat .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }}
  .stat.hero {{ grid-column: span 2; border-color: #5a3a1c; background: linear-gradient(160deg, #2c1d16, var(--panel)); }}
  .stat.hero .n {{ font-size: 64px; color: var(--pumpkin); }}
  section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 20px; margin-bottom: 20px; }}
  h2 {{ font-family: Creepster, cursive; font-weight: 400; font-size: 30px; margin: 0 0 4px; color: var(--ink); letter-spacing: .02em; }}
  .note {{ color: var(--ink-3); font-size: 14px; margin: 0 0 12px; }}
  .legend {{ display: flex; gap: 16px; color: var(--ink-2); font-size: 14px; margin-bottom: 6px; flex-wrap: wrap; }}
  .legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
  .chart {{ position: relative; }}
  svg {{ width: 100%; height: auto; display: block; }}
  .grid {{ stroke: var(--line); stroke-width: 1; }}
  .axis {{ stroke: #4a3e5a; stroke-width: 1; }}
  .tick {{ fill: var(--ink-3); font-size: 11px; font-family: Nunito, sans-serif; }}
  .col {{ cursor: default; outline: none; }}
  .col:hover path, .col:focus path {{ filter: brightness(1.2); }}
  .tip {{ position: absolute; pointer-events: none; background: #0f0b15; border: 1px solid var(--line); color: var(--ink);
          padding: 6px 10px; border-radius: 8px; font-size: 13px; white-space: nowrap; opacity: 0; transition: opacity .1s; }}
  details {{ margin-top: 10px; color: var(--ink-2); font-size: 14px; }}
  table {{ border-collapse: collapse; margin-top: 8px; }}
  td, th {{ padding: 2px 14px 2px 0; text-align: right; }} td:first-child, th:first-child {{ text-align: left; }}
  .findings {{ list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
  .findings li {{ border: 1px solid var(--line); border-radius: 40px 40px 8px 8px; padding: 18px 18px 14px; background: #1b1525; }}
  .findings h3 {{ margin: 0 0 4px; font-size: 16px; color: var(--pumpkin); }}
  .findings p {{ margin: 0; color: var(--ink-2); }}
  figure {{ margin: 0; }}
  figure img {{ width: 100%; border-radius: 10px; display: block; }}
  figcaption {{ color: var(--ink-3); font-size: 14px; margin-top: 8px; }}
  .key {{ display: flex; gap: 14px; flex-wrap: wrap; font-size: 14px; color: var(--ink-2); margin: 0 0 10px; }}
  .key i {{ display: inline-block; width: 14px; height: 14px; border: 3px solid; border-radius: 3px; margin-right: 6px; vertical-align: -2px; }}
  footer {{ color: var(--ink-3); font-size: 13px; line-height: 1.6; }}
  @media (max-width: 720px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} .stat.hero .n {{ font-size: 52px; }} }}
</style></head>
<body><div class="sky"><main>
<header>
  <div class="kicker">Halloween 2026 &middot; Visitor census</div>
  <h1>The Haunted Pumpkin Graveyard</h1>
  <p class="sub">A camera watched the front gate from {first} to {last}. Here is everyone who dared to enter,
  counted one by one by a computer that does not believe in ghosts.</p>
  <svg class="graves" viewBox="0 0 960 70" preserveAspectRatio="none" aria-hidden="true">
    <path d="M0 62 Q240 48 480 58 T960 54 V70 H0Z" fill="#2a2138"/>
    <g fill="#3d3150">
      <path d="M90 62 V30 Q90 14 108 14 Q126 14 126 30 V62Z"/><path d="M300 60 V36 Q300 24 314 24 Q328 24 328 36 V60Z"/>
      <path d="M640 58 V22 Q640 6 660 6 Q680 6 680 22 V58Z"/><rect x="858" y="18" width="8" height="40"/><rect x="848" y="28" width="28" height="7"/>
    </g>
    <g fill="#f08a24"><ellipse cx="190" cy="56" rx="16" ry="11"/><ellipse cx="520" cy="58" rx="12" ry="9"/><ellipse cx="770" cy="55" rx="18" ry="12"/></g>
    <g fill="#4d7a3a"><rect x="188" y="42" width="4" height="6" rx="1"/><rect x="518" y="47" width="3" height="5" rx="1"/><rect x="768" y="40" width="4" height="6" rx="1"/></g>
  </svg>
</header>

<div class="stats">
  <div class="stat hero"><div class="n">{total}</div><div class="l">souls crossed the gate</div></div>
  <div class="stat"><div class="n">{kids}</div><div class="l"><span class="swatch" style="background:var(--kid)"></span>little goblins (kids)</div></div>
  <div class="stat"><div class="n">{adults}</div><div class="l"><span class="swatch" style="background:var(--adult)"></span>grown-up spirits</div></div>
  <div class="stat"><div class="n">{groups}</div><div class="l">covens &middot; {avg_group:.1f} per group</div></div>
  <div class="stat"><div class="n">{hours:.1f}h</div><div class="l">of haunting, {first}&ndash;{last}</div></div>
</div>

<section>
  <h2>When the dead walked</h2>
  <p class="note">Visitors arriving per 15 minutes. Hover a column for the numbers.</p>
  <div class="legend"><span><i style="background:var(--kid)"></i>Kids</span><span><i style="background:var(--adult)"></i>Adults</span></div>
  <div class="chart">{chart}<div class="tip" id="tip"></div></div>
  <details><summary>Show as a table</summary><table><tr><th>Time</th><th>Kids</th><th>Adults</th><th>Total</th></tr>{table}</table></details>
</section>

<section>
  <h2>Tales from the crypt</h2>
  <ul class="findings">{findings}</ul>
</section>

<section>
  <h2>Through the camera's eye</h2>
  <p class="note">The busiest moment, {img_time}: {img_count} visitors counted at once. Each person the computer saw gets a box.</p>
  <div class="key"><span><i style="border-color:{KID}"></i>Kid</span><span><i style="border-color:{ADULT}"></i>Adult</span>
    <span><i style="border-color:{HOST}"></i>Host (not counted)</span><span><i style="border-color:{PROP}"></i>Pumpkin stack (ignored)</span>
    <span><i style="border-color:#ffc850"></i>The yard</span></div>
  <figure><img src="data:image/jpeg;base64,{img}" alt="Camera frame with boxes drawn around each visitor"></figure>
</section>

<footer>
  <p><b>How it was counted.</b> A Tapo camera recorded the yard. A person detector (YOLO) and a tracker followed each person
  through the video at 2 frames per second, and everyone who spent at least 2 seconds in the yard counts once. People on the
  sidewalk, the hosts (recognized by their costumes) and the pumpkin stack are left out. Kids and adults are told apart by
  an image model plus how tall each person is for where they stand.</p>
  <p><b>How sure is it?</b> Four 3-minute stretches were counted by hand and compared: 63 people by hand, 62 by the computer
  (quiet 7 vs 12, empty 0 vs 0, peak 34 vs 34, busy 22 vs 16). A separate frame-by-frame method, corrected for the crowds it
  misses, lands near 730. Call the total roughly 650&ndash;850. The kid/adult split is rougher: about &plusmn;10%.</p>
</footer>
</main></div>
<script>
  const tip = document.getElementById("tip"), chart = tip.parentElement;
  chart.querySelectorAll(".col").forEach(c => {{
    const show = e => {{
      tip.textContent = c.dataset.tip; tip.style.opacity = 1;
      const b = chart.getBoundingClientRect(), r = c.getBoundingClientRect();
      const x = Math.min(Math.max(r.left - b.left + r.width / 2 - tip.offsetWidth / 2, 0), b.width - tip.offsetWidth);
      tip.style.left = x + "px"; tip.style.top = Math.max(r.top - b.top - 36, 0) + "px";
    }};
    c.addEventListener("mouseenter", show); c.addEventListener("focus", show);
    c.addEventListener("mouseleave", () => tip.style.opacity = 0); c.addEventListener("blur", () => tip.style.opacity = 0);
  }});
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
