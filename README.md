# fp_cvat_visualization

Standalone visualization tool for FOOTPASS play-by-play events on video. Renders prediction and ground-truth event panels over broadcast footage, with optional CVAT player-track overlays and a tactical (HDF5) variant that uses normalised pitch coordinates.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

## Scripts

All three scripts live in `visualization/` and are run via `uv run`:

- `visualize_events.py` — predictions (left) vs. ground truth (right). Optional CVAT XML marks the acting player on the pitch.
- `visualize_events_post_processing.py` — same layout, but additionally highlights events that were removed by post-processing (magenta).
- `visualize_events_tactical.py` — tactical variant that reads player positions from an HDF5 file and draws them on a minimap.

### Action classes

`0=Background, 1=Drive, 2=Pass, 3=Cross, 4=Shot, 5=Header, 6=Throw-in, 7=Tackle, 8=Block`

### Event JSON formats

- Predictions — 5 fields: `[frame, team, jersey, class, confidence]`
- Ground truth — 6 fields: `[frame, team, jersey, class, visible, split]`

## Example

```bash
uv run python visualization/visualize_events.py \
    --video  data/rbk_videos/game_103.mp4 \
    --json   data/play_by_play/fold_103/playbyplay_TAAD_val.json \
    --gt     data/playbyplay_GT_rbk/playbyplay_train.json \
    --key    game_103_H1 \
    --output output/game_103_events.mp4 \
    --min_confidence 0.35
```

Run any script with `--help` to see all options.

## Layout

```
visualization/        # the three scripts
data/                 # local input data (gitignored)
output/               # rendered videos (gitignored)
```
