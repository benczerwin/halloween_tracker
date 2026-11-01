"""Run a person detector over every extracted frame.

Writes one row per detected person to output/detections.csv, plus a row with
no box for frames with nobody in them, so later steps know the frame exists.

Usage:
    uv run detect.py
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--out", type=Path, default=Path("output/detections.csv"))
    p.add_argument("--model", default="yolo11m.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    frames = sorted(args.frames.glob("*.jpg"))

    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "x1", "y1", "x2", "y2", "conf"])
        for i in range(0, len(frames), args.batch):
            batch = frames[i : i + args.batch]
            results = model.predict(
                [str(f) for f in batch],
                classes=[0],  # COCO "person"
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            for f, r in zip(batch, results):
                boxes = r.boxes.xyxy.tolist()
                confs = r.boxes.conf.tolist()
                if not boxes:
                    w.writerow([f.stem, "", "", "", "", ""])
                for (x1, y1, x2, y2), c in zip(boxes, confs):
                    w.writerow([f.stem, round(x1), round(y1), round(x2), round(y2), round(c, 3)])
            print(f"{min(i + args.batch, len(frames))}/{len(frames)}", flush=True)


if __name__ == "__main__":
    main()
