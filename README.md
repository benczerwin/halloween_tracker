# halloween_tracker

Count trick-or-treaters from a front-yard camera (a Tapo C120 recording continuously to a microSD card).

## Pipeline

```bash
uv run extract_frames.py --date 20261031   # footage/*.mp4 -> frames/, 1 frame per second, named by clock time
uv run detect.py                           # YOLO person detection -> output/detections.csv
uv run residents.py                        # tag hosts and decorations -> output/labeled.csv, output/review/
uv run count.py                            # visits, chart and review page -> output/
open output/review.html
```

Requires `ffmpeg` (`brew install ffmpeg`) and [uv](https://docs.astral.sh/uv/).

## How it counts

1. **Yard zone** (`zone.json`): only people whose feet are inside the yard count, so people walking by on the
   sidewalk don't.
2. **Hosts** (`residents.json`): the people handing out candy are in view all afternoon. Each is described in a
   sentence; CLIP picks out clear examples, a small classifier learns their look, and the person tags are averaged
   over each short track. If a host was recognized just before and after a frame, the closest match in that frame
   is tagged too. Anything that stays in exactly the same spot for minutes (a pumpkin stack) is a decoration.
3. **Visits**: frames with visitors are grouped into visits, split by 20+ seconds of empty yard. A visit's size is
   the number of arrivals (the count stepping up and staying up), and at least the most people seen at once.

## Checking the results

- `output/review.html`: every visit with red boxes on counted visitors, blue on hosts, gray on decorations.
- `output/review/host_*.jpg`: who was tagged as each host. Visitors in here were missed.
- `output/review/near_*.jpg`: visitors that look a bit like a host. A host in here was counted by mistake.
- `output/review/decorations.jpg`: what was ignored as a decoration.

Adjust the descriptions or `min_prob` in `residents.json`, or the polygon in `zone.json`, then rerun
`residents.py` and `count.py`.

## Privacy

Footage, frames and anything showing people stay local: `.gitignore` excludes them. This repo is public.
