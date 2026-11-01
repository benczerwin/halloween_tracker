# halloween_tracker

Count trick-or-treaters from a front-yard camera (a Tapo C120 recording continuously to a microSD card).

## Pipeline

```bash
# Per-second frames: learn who the hosts are and what kids look like, plus a quick frame-based count
uv run extract_frames.py --date 20261031   # footage/*.mp4 -> frames/, 1 frame per second, named by clock time
uv run detect.py                           # YOLO person detection -> output/detections.csv
uv run residents.py                        # tag hosts and decorations -> output/labeled.csv, output/review/
uv run ages.py                             # kid vs adult, fit to age_labels.csv -> p_kid in labeled.csv
uv run count.py                            # frame-based visits, chart and review page -> output/

# Tracking: follow each person through the video and count them once
uv run track.py --date 20261031            # YOLO + BoT-SORT at 2 fps -> output/tracks.csv, output/track_crops/
uv run count_tracks.py                     # one row per visitor -> output/people.csv, output/groups.csv
uv run report.py                           # the Haunted Pumpkin Graveyard one-pager -> output/report.html
```

Requires `ffmpeg` (`brew install ffmpeg`) and [uv](https://docs.astral.sh/uv/). `track.py` takes about 1.5 hours
for 3.5 hours of footage on an M1 Pro.

## How it counts

1. **Yard zone** (`zone.json`): only people whose feet are inside the yard count, so people walking by on the
   sidewalk don't.
2. **Decorations** (`zone.json` `ignore_boxes`): props the detector mistakes for people never move, so their boxes are
   listed once and any detection mostly overlapping one is ignored.
3. **Hosts** (`residents.json`): the people handing out candy are in view all afternoon. Each is described in a
   sentence; CLIP picks out clear examples and a small classifier learns their look.
4. **Kids vs adults** (`age_labels.csv`): CLIP's "child or adult?" plus how tall someone is for where they stand
   (the camera is low, so people look shorter further away). Fit to ~90 hand-labeled people; about 72% right per
   person, and per-group kid counts use the average likelihood so errors mostly cancel.
5. **Counting**: `track.py` follows each person across frames. `count_tracks.py` drops hosts, joins pieces of one
   person who stepped behind the tree, and counts everyone who spent 2+ seconds in the yard. People arriving within
   20 seconds of each other form a group.

`count.py` is the older frame-based count (no tracking): visits split by 20+ seconds of empty yard, sized by
arrivals. It undercounts when groups overlap but needs no video re-processing.

## Checking the results

- `output/report.html`: the one-pager, including a frame with every person boxed (kid, adult, host, prop).
- `output/review.html`: every frame-based visit with red boxes on counted visitors, blue on hosts, gray on decorations.
- `output/review/host_*.jpg` / `near_*.jpg`: who was tagged as a host, and visitors who look a bit like one.
- `output/review/ages_kids.jpg` / `ages_adults.jpg`: the most confident kid and adult calls.
- `output/review/decorations.jpg`: what was ignored as a decoration.

## Privacy

Footage, frames, crops and anything showing people (including the report) stay local: `.gitignore` excludes them.
`age_labels.csv` holds box coordinates only. This repo is public.
