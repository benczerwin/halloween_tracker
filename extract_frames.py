"""Extract still frames from Tapo SD-card recordings.

Tapo files are named like 20261031_142701_tp00002.mp4, where the first two
fields are the recording start time. Each extracted frame is named after the
wall-clock time it was taken, e.g. frames/20261031_142731.jpg.

Usage:
    uv run extract_frames.py --date 20261031 --every 1
"""

import argparse
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

NAME_RE = re.compile(r"(\d{8}_\d{6})_tp\d+\.mp4$")


def recording_start(path: Path) -> datetime | None:
    m = NAME_RE.search(path.name)
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else None


def extract(video: Path, start: datetime, out_dir: Path, every: float, width: int) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        # Sample at the middle of each interval so frame n sits at (n - 0.5) * every.
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(video),
                "-vf", f"fps=1/{every},scale={width}:-2",
                "-q:v", "3",
                f"{tmp}/%06d.jpg",
            ],
            check=True,
        )
        count = 0
        for f in sorted(Path(tmp).glob("*.jpg")):
            n = int(f.stem)
            ts = start + timedelta(seconds=(n - 0.5) * every)
            f.rename(out_dir / f"{ts:%Y%m%d_%H%M%S}.jpg")
            count += 1
    return count


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--footage", type=Path, default=Path("footage"))
    p.add_argument("--out", type=Path, default=Path("frames"))
    p.add_argument("--date", help="only use recordings starting on this day (YYYYMMDD)")
    p.add_argument("--every", type=float, default=1.0, help="seconds between frames")
    p.add_argument("--width", type=int, default=1280)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    videos = sorted(args.footage.glob("*.mp4"))
    for video in videos:
        start = recording_start(video)
        if start is None:
            print(f"skip {video.name}: unrecognized name")
            continue
        if args.date and f"{start:%Y%m%d}" != args.date:
            print(f"skip {video.name}: not on {args.date}")
            continue
        n = extract(video, start, args.out, args.every, args.width)
        print(f"{video.name}: {n} frames from {start:%H:%M:%S}")


if __name__ == "__main__":
    main()
