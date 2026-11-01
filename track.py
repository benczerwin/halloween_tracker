"""Detect and track people through the video, so each person can be counted once.

Decodes each video with ffmpeg at --fps frames per second, runs YOLO + BoT-SORT
on each frame and writes one row per tracked person per frame to output/tracks.csv, in the
same 1280x720 coordinates as zone.json. For each track it also saves a few
small crops (output/track_crops/) so hosts and kids can be recognized later.

Usage:
    uv run track.py --date 20261031
"""

import argparse
import csv
import json
import subprocess
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from matplotlib.path import Path as Polygon
from ultralytics import YOLO

from extract_frames import recording_start

SAVE_AT = {1, 3, 6, 12, 24, 48}  # which in-zone sighting of a track to save a crop of
W, H = 1280, 720


def frames(video: Path, fps: float):
    """Yield decoded frames at `fps`, scaled to W x H (ffmpeg decodes much faster than OpenCV)."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "quiet", "-i", str(video), "-vf", f"fps={fps},scale={W}:{H}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    size = W * H * 3
    while len(buf := proc.stdout.read(size)) == size:
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    proc.wait()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--footage", type=Path, default=Path("footage"))
    p.add_argument("--date", help="only use recordings starting on this day (YYYYMMDD)")
    p.add_argument("--zone", type=Path, default=Path("zone.json"))
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--model", default="yolo11m.pt")
    p.add_argument("--tracker", default="tracker.yaml")
    p.add_argument("--fps", type=float, default=2.0, help="frames per second to track at")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    zone = Polygon(json.loads(args.zone.read_text())["polygon"])
    crops = args.out / "track_crops"
    crops.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    with (args.out / "tracks.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "track", "x1", "y1", "x2", "y2", "conf", "in_zone"])
        for vi, video in enumerate(sorted(args.footage.glob("*.mp4"))):
            start = recording_start(video)
            if start is None or (args.date and f"{start:%Y%m%d}" != args.date):
                continue
            sightings: dict[int, int] = {}
            model.predictor = None  # fresh tracker for each recording
            for k, img in enumerate(frames(video, args.fps)):
                t = start + timedelta(seconds=k / args.fps)
                r = model.track(img, persist=True, tracker=args.tracker, classes=[0], imgsz=args.imgsz,
                                device=args.device, verbose=False)[0]
                if r.boxes.id is None:
                    continue
                for (x1, y1, x2, y2), tid, conf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.conf.tolist()):
                    track = f"{vi:02d}-{tid}"
                    b = [round(x1), round(y1), round(x2), round(y2)]
                    inside = bool(zone.contains_point(((b[0] + b[2]) / 2, b[3])))
                    w.writerow([f"{t:%Y-%m-%d %H:%M:%S.%f}"[:-4], track, *b, round(conf, 3), int(inside)])
                    if inside:
                        n = sightings[tid] = sightings.get(tid, 0) + 1
                        if n in SAVE_AT:
                            crop = img[max(0, b[1]):b[3], max(0, b[0]):b[2]]
                            if crop.size:
                                crop = cv2.resize(crop, (max(1, round(crop.shape[1] * 224 / crop.shape[0])), 224))
                                cv2.imwrite(str(crops / f"{track}_{n:02d}.jpg"), crop)
                if k % 600 == 0:
                    print(f"{video.name} {t:%H:%M:%S}", flush=True)
            print(f"{video.name}: done, {len(sightings)} tracks entered the yard", flush=True)


if __name__ == "__main__":
    main()
