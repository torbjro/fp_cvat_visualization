"""
Visualize FOOTPASS events on video using tactical pitch data (HDF5).

Companion to visualization/visualize_events.py. The panels, legend, colours and
match/misclass logic are reused from that script. The only difference is
the player-position source: instead of CVAT image-space bounding boxes, we
read normalised pitch coordinates from an HDF5 tactical dataset.

Because pitch coordinates are not directly projectable onto the broadcast
footage (no homography is supplied), the acting player is highlighted on a
minimap overlay rather than drawn as a bounding box on the pitch itself.

HDF5 layout: one dataset per half (e.g. 'game_10_H1') with 14 columns
    0: frame index (absolute; shared with the playbyplay JSON)
    1: internal player id
    2: team (0 or 1)
    3: jersey number
    4: position code
    5: pitch x, normalised to [0, 1]
    6: pitch y, normalised to [0, 1]
    7–13: velocity / reserved / status — unused here

Usage:
    uv run python visualization/visualize_events_tactical.py \\
        --video  data/game_10.mp4 \\
        --gt     data/FP/playbyplay_train.json \\
        --h5     data/FP/train_tactical_data.h5 \\
        --key    game_10_H1 \\
        --output output/game_10_vis.mp4

    # Also overlay predictions (same JSON format as visualize_events.py):
    uv run python visualization/visualize_events_tactical.py \\
        --video  data/game_10.mp4 \\
        --json   path/to/predictions.json \\
        --gt     data/FP/playbyplay_train.json \\
        --h5     data/FP/train_tactical_data.h5 \\
        --key    game_10_H1 \\
        --output output/game_10_vis.mp4
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

# Reuse all shared drawing / loading logic from the sibling script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_events import (  # noqa: E402
    ACTION_LABELS,
    MARKER_GT_COLOR,
    MARKER_MATCH_COLOR,
    MARKER_MISCLASS_COLOR,
    MARKER_PRED_COLOR,
    PANEL_TEXT_COLOR,
    TEAM_LABELS,
    Event,
    assign_slots,
    build_frame_lookup,
    draw_dashed_rect,
    draw_rounded_rect,
    filter_by_confidence,
    load_events,
)

# ---------------------------------------------------------------------------
# Compact layout — sized for small videos (e.g. 640×352)
# ---------------------------------------------------------------------------

PANEL_WIDTH   = 180
PANEL_HEIGHT  = 40
PANEL_Y_START = 28
PANEL_GAP     = 4
CORNER_RADIUS = 5
ALPHA         = 0.78

# ---------------------------------------------------------------------------
# Compact panel / legend drawing (for small-resolution videos)
# ---------------------------------------------------------------------------

def draw_header(img: np.ndarray, text: str, x: int,
                color: Tuple[int, int, int]) -> np.ndarray:
    cv2.putText(img, text, (x, PANEL_Y_START - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return img


def draw_panels(
    frame_img: np.ndarray,
    active_events: List[Event],
    event_slots: Dict[Tuple[int, int, int, int], int],
    current_frame: int,
    display_frames: int,
    anchor_x: int,
    is_gt: bool,
) -> np.ndarray:
    img = frame_img
    accent_color = MARKER_GT_COLOR if is_gt else MARKER_PRED_COLOR
    header_text = "GROUND TRUTH" if is_gt else "PREDICTED"
    img = draw_header(img, header_text, anchor_x, accent_color)

    img_h = img.shape[0]
    for evt in active_events:
        evt_frame, team, jersey, cls, conf = evt
        key = (evt_frame, team, jersey, cls)
        slot = event_slots.get(key)
        if slot is None:
            continue
        y1 = PANEL_Y_START + slot * (PANEL_HEIGHT + PANEL_GAP)
        y2 = y1 + PANEL_HEIGHT
        if y2 > img_h - 6:
            continue
        x1 = anchor_x
        x2 = anchor_x + PANEL_WIDTH

        img = draw_rounded_rect(img, x1, y1, x2, y2, CORNER_RADIUS,
                                (20, 20, 20), ALPHA)

        # Accent bar
        if is_gt:
            img = draw_rounded_rect(img, x2 - 5, y1, x2, y2, CORNER_RADIUS,
                                    accent_color, 0.9)
        else:
            img = draw_rounded_rect(img, x1, y1, x1 + 5, y2, CORNER_RADIUS,
                                    accent_color, 0.9)

        action_name = ACTION_LABELS.get(cls, f"Class {cls}")
        cv2.putText(img, action_name, (x1 + 10, y1 + 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45,
                    PANEL_TEXT_COLOR, 1, cv2.LINE_AA)

        team_str = f"{TEAM_LABELS.get(team, str(team))} #{jersey}"
        cv2.putText(img, team_str, (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (200, 200, 200), 1, cv2.LINE_AA)

        if not is_gt:
            conf_str = f"{conf * 100:.0f}%"
            cv2.putText(img, conf_str, (x2 - 34, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (160, 160, 160), 1, cv2.LINE_AA)

        # Progress bar
        frames_remaining = max(0, (evt_frame + display_frames) - current_frame)
        progress  = frames_remaining / display_frames
        bar_max_w = PANEL_WIDTH - 10
        bar_w     = int(bar_max_w * progress)
        bar_y     = y2 - 4
        cv2.rectangle(img, (x1 + 5, bar_y), (x1 + 5 + bar_max_w, bar_y + 2),
                      (60, 60, 60), -1)
        cv2.rectangle(img, (x1 + 5, bar_y), (x1 + 5 + bar_w, bar_y + 2),
                      accent_color, -1)

    return img


def draw_legend(img: np.ndarray) -> None:
    """Compact legend for small videos."""
    entries = [
        ("Pred (dashed)",       MARKER_PRED_COLOR,     False),
        ("GT (dashed)",         MARKER_GT_COLOR,       False),
        ("Match (solid)",       MARKER_MATCH_COLOR,    True),
        ("Wrong cls (solid)",   MARKER_MISCLASS_COLOR, True),
    ]
    h, w = img.shape[:2]
    pad = 6
    row_h = 14
    box_w = 145
    box_h = pad * 2 + row_h * len(entries) + 14
    x1 = 10
    y2 = h - 10
    y1 = y2 - box_h
    x2 = x1 + box_w
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, dst=img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), 1)
    cv2.putText(img, "LEGEND", (x1 + pad, y1 + pad + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 230, 230), 1, cv2.LINE_AA)
    for i, (label, color, solid) in enumerate(entries):
        ry = y1 + pad + 16 + i * row_h
        sx1, sy1 = x1 + pad, ry
        sx2, sy2 = sx1 + 18, ry + 9
        if solid:
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), color, 1)
        else:
            draw_dashed_rect(img, (sx1, sy1), (sx2, sy2), color,
                             thickness=1, dash=3, gap=2)
        cv2.putText(img, label, (sx2 + 5, sy2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 220, 220),
                    1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Tactical minimap
# ---------------------------------------------------------------------------

# BGR team colours for the minimap. Deliberately distinct from the
# pred/GT/match/misclass palette so team identity never clashes with the
# event-status ring colours drawn on top.
TEAM_MINIMAP_COLORS = {
    0: (230, 230, 230),  # team 0 — light grey / white
    1: (60,  60,  200),  # team 1 — dark red
}

MINIMAP_PAD       = 4      # padding between pitch border and overlay edge
MINIMAP_BG_ALPHA  = 0.78
MINIMAP_LINE_CLR  = (100, 100, 100)
MINIMAP_DOT_R     = 2


def load_tactical(h5_path: str, key: str) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    """
    Load one half of the tactical dataset into memory and build a
    frame -> (start, end) row-slice index.
    """
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            available = list(f.keys())[:20]
            raise KeyError(f"Key '{key}' not in h5. First 20 available: {available}")
        arr = f[key][:]

    frames = arr[:, 0].astype(np.int64)
    order = np.argsort(frames, kind="stable")
    arr = arr[order]
    frames = frames[order]

    uniq, starts = np.unique(frames, return_index=True)
    ends = np.append(starts[1:], len(frames))
    slices = {int(u): (int(s), int(e)) for u, s, e in zip(uniq, starts, ends)}
    return arr, slices


def pitch_to_minimap(
    x_norm: float, y_norm: float,
    mx1: int, my1: int, mx2: int, my2: int,
) -> Tuple[int, int]:
    """Map pitch coordinates (0..1) to pixel coordinates inside the minimap."""
    inner_x1 = mx1 + MINIMAP_PAD
    inner_y1 = my1 + MINIMAP_PAD
    inner_w  = (mx2 - mx1) - 2 * MINIMAP_PAD
    inner_h  = (my2 - my1) - 2 * MINIMAP_PAD
    x_clamped = max(0.0, min(1.0, float(x_norm)))
    y_clamped = max(0.0, min(1.0, float(y_norm)))
    return (
        inner_x1 + int(round(x_clamped * inner_w)),
        inner_y1 + int(round(y_clamped * inner_h)),
    )


def draw_pitch_lines(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """Draw a thin rectangle + halfway line + centre circle inside the minimap."""
    ix1 = x1 + MINIMAP_PAD
    iy1 = y1 + MINIMAP_PAD
    ix2 = x2 - MINIMAP_PAD
    iy2 = y2 - MINIMAP_PAD
    cv2.rectangle(img, (ix1, iy1), (ix2, iy2), MINIMAP_LINE_CLR, 1)
    cx = (ix1 + ix2) // 2
    cy = (iy1 + iy2) // 2
    cv2.line(img, (cx, iy1), (cx, iy2), MINIMAP_LINE_CLR, 1)
    radius = max(6, min(ix2 - ix1, iy2 - iy1) // 10)
    cv2.circle(img, (cx, cy), radius, MINIMAP_LINE_CLR, 1)


def draw_minimap(
    img: np.ndarray,
    players: np.ndarray,
    pred_by_player: Dict[Tuple[int, int], Event],
    gt_by_player: Dict[Tuple[int, int], Event],
    matched_players: set,
    misclass_players: set,
    anchor: Tuple[int, int, int, int],
) -> None:
    """
    Draw the minimap in the rectangle given by `anchor` (x1, y1, x2, y2):
      * all 22 players as small team-coloured dots
      * players that have an active event get a coloured ring:
          gold   — match (same player, same class)
          red    — misclass (same player, wrong class)
          blue   — pred only
          green  — GT only
    """
    x1, y1, x2, y2 = anchor
    draw_rounded_rect(img, x1, y1, x2, y2, 6, (20, 20, 20), MINIMAP_BG_ALPHA)
    draw_pitch_lines(img, x1, y1, x2, y2)

    cv2.putText(img, "PITCH", (x1 + MINIMAP_PAD + 1, y1 + MINIMAP_PAD + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1, cv2.LINE_AA)

    for row in players:
        team   = int(row[2])
        jersey = int(row[3])
        px, py = pitch_to_minimap(row[5], row[6], x1, y1, x2, y2)

        base_color = TEAM_MINIMAP_COLORS.get(team, (200, 200, 200))
        cv2.circle(img, (px, py), MINIMAP_DOT_R, base_color, -1)

        key = (team, jersey)
        ring_color: Optional[Tuple[int, int, int]] = None
        ring_solid = True
        if key in matched_players:
            ring_color = MARKER_MATCH_COLOR
        elif key in misclass_players:
            ring_color = MARKER_MISCLASS_COLOR
        elif key in pred_by_player:
            ring_color = MARKER_PRED_COLOR
            ring_solid = False
        elif key in gt_by_player:
            ring_color = MARKER_GT_COLOR
            ring_solid = False

        if ring_color is not None:
            r = MINIMAP_DOT_R + 2
            if ring_solid:
                cv2.circle(img, (px, py), r, ring_color, 1)
                cv2.circle(img, (px, py), r + 1, ring_color, 1)
            else:
                for start_deg in range(0, 360, 45):
                    cv2.ellipse(img, (px, py), (r, r), 0,
                                start_deg, start_deg + 25, ring_color, 1)

            cv2.putText(img, f"#{jersey}", (px + r + 1, py + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, ring_color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    h5_path: str,
    key: str,
    output_path: str,
    gt_path: Optional[str],
    pred_path: Optional[str],
    display_seconds: float,
    fps_override: Optional[float],
    min_confidence: float,
    show: str,
) -> None:
    if not gt_path and not pred_path:
        raise ValueError("At least one of --gt or --json must be provided.")

    # --- Load events ---
    pred_events: List[Event] = []
    pred_lookup: Dict[int, List[Event]] = {}
    if pred_path:
        print(f"Loading predictions: {pred_path}  (key={key})")
        pred_events = filter_by_confidence(
            load_events(pred_path, key, is_gt=False), min_confidence,
        )
        pred_lookup = build_frame_lookup(pred_events)
        print(f"  {len(pred_events)} predictions (conf >= {min_confidence})")

    gt_events: List[Event] = []
    gt_lookup: Dict[int, List[Event]] = {}
    if gt_path:
        print(f"Loading ground truth: {gt_path}  (key={key})")
        gt_events = load_events(gt_path, key, is_gt=True)
        gt_lookup = build_frame_lookup(gt_events)
        print(f"  {len(gt_events)} GT events")

    # --- Load tactical data ---
    print(f"Loading tactical data: {h5_path}  (key={key})")
    arr, frame_slices = load_tactical(h5_path, key)
    tact_min = min(frame_slices) if frame_slices else 0
    tact_max = max(frame_slices) if frame_slices else 0
    print(f"  {len(frame_slices)} frames with tactical data "
          f"(range {tact_min}–{tact_max})")

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0
    display_frames = int(display_seconds * fps)
    print(f"  Video: {width}x{height} @ {fps:.1f} fps")
    print(f"  Display duration: {display_seconds}s = {display_frames} frames")

    # Seek to the first tactical frame so we don't waste time on lead-in.
    cap.set(cv2.CAP_PROP_POS_FRAMES, tact_min)

    out = Path(output_path)
    if out.is_dir() or out.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise ValueError(
            f"--output must be a video file path (e.g. foo.mp4), got: {output_path}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for: {output_path}")

    show_pred = show in ("pred", "both") and pred_path is not None
    show_gt   = show in ("gt", "both")   and gt_path   is not None

    pred_anchor_x = 10
    gt_anchor_x   = width - PANEL_WIDTH - 10

    # Minimap: bottom-right, compact.
    minimap_w = min(160, width // 4)
    minimap_h = int(minimap_w / 1.54)  # real pitch aspect ratio 105 × 68
    mm_x2 = width - 10
    mm_y2 = height - 10
    mm_x1 = mm_x2 - minimap_w
    mm_y1 = mm_y2 - minimap_h
    minimap_anchor = (mm_x1, mm_y1, mm_x2, mm_y2)

    active_pred: List[Event] = []
    active_gt:   List[Event] = []
    pred_slots:  Dict[Tuple[int, int, int, int], int] = {}
    gt_slots:    Dict[Tuple[int, int, int, int], int] = {}
    max_slots = max(1, (height - PANEL_Y_START - 10) // (PANEL_HEIGHT + PANEL_GAP))

    frame_idx = tact_min
    while frame_idx <= tact_max:
        ret, frame = cap.read()
        if not ret:
            break

        if show_pred and frame_idx in pred_lookup:
            active_pred.extend(pred_lookup[frame_idx])
        if show_gt and frame_idx in gt_lookup:
            active_gt.extend(gt_lookup[frame_idx])

        active_pred = [e for e in active_pred if frame_idx < e[0] + display_frames]
        active_gt   = [e for e in active_gt   if frame_idx < e[0] + display_frames]

        assign_slots(active_pred, pred_slots, max_slots)
        assign_slots(active_gt,   gt_slots,   max_slots)

        # Group active events by player for ring colouring on the minimap.
        pred_by_player: Dict[Tuple[int, int], Event] = {
            (e[1], e[2]): e for e in active_pred
        }
        gt_by_player: Dict[Tuple[int, int], Event] = {
            (e[1], e[2]): e for e in active_gt
        }
        matched_players:  set = set()
        misclass_players: set = set()
        for key_ in set(pred_by_player) & set(gt_by_player):
            if pred_by_player[key_][3] == gt_by_player[key_][3]:
                matched_players.add(key_)
            else:
                misclass_players.add(key_)

        sl = frame_slices.get(frame_idx)
        if sl is not None:
            players = arr[sl[0]:sl[1]]
            draw_minimap(
                frame, players,
                pred_by_player=pred_by_player,
                gt_by_player=gt_by_player,
                matched_players=matched_players,
                misclass_players=misclass_players,
                anchor=minimap_anchor,
            )

        if show_pred and active_pred:
            frame = draw_panels(frame, active_pred, pred_slots, frame_idx,
                                display_frames, anchor_x=pred_anchor_x, is_gt=False)
        if show_gt and active_gt:
            frame = draw_panels(frame, active_gt, gt_slots, frame_idx,
                                display_frames, anchor_x=gt_anchor_x, is_gt=True)

        draw_legend(frame)
        writer.write(frame)
        frame_idx += 1

        if (frame_idx - tact_min) % 500 == 0:
            total = tact_max - tact_min + 1
            pct = (frame_idx - tact_min) / total * 100
            print(f"  {frame_idx - tact_min}/{total} frames ({pct:.1f}%)")

    cap.release()
    writer.release()
    print(f"Done. Saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize FOOTPASS events on video using tactical pitch "
                    "data (HDF5). Acting players are highlighted on a minimap."
    )
    parser.add_argument("--video",  required=True)
    parser.add_argument("--h5",     required=True,
                        help="Path to the tactical HDF5 file.")
    parser.add_argument("--key",    required=True,
                        help="Dataset / game key, e.g. 'game_10_H1'")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gt",     default=None,
                        help="Path to GT play-by-play JSON (6-field format).")
    parser.add_argument("--json",   default=None,
                        help="Path to predictions JSON (5-field format).")
    parser.add_argument("--display_seconds", type=float, default=3.0)
    parser.add_argument("--fps",    type=float, default=None)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--show", choices=["pred", "gt", "both"], default="both")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        video_path=args.video,
        h5_path=args.h5,
        key=args.key,
        output_path=args.output,
        gt_path=args.gt,
        pred_path=args.json,
        display_seconds=args.display_seconds,
        fps_override=args.fps,
        min_confidence=args.min_confidence,
        show=args.show,
    )
