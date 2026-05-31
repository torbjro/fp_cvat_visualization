"""
Standalone script for visualizing FOOTPASS events on a video.
Predictions are shown on the LEFT, Ground Truth (GT) on the RIGHT.

Usage:
    uv run python visualization/visualize_events.py \
        --video  data/videos/game_103.mp4 \
        --json   data/playbyplay_PRED/fold_103/playbyplay_TAAD_val.json \
        --gt     data/playbyplay_GT/playbyplay_train.json \
        --key    game_103_H1 \
        --output output/game_103_vis.mp4 \
        --min_confidence 0.15

    # Without GT (only predictions):
    uv run python visualization/visualize_events.py \
        --video  data/videos/game_103.mp4 \
        --json   data/playbyplay_Pred/fold_103/playbyplay_TAAD_val.json \
        --key    game_103_H1 \
        --output output/game_103_vis.mp4

    # With CVAT player tracks (mark the acting player on the pitch):
    uv run python visualization/visualize_events.py \
        --video  data/videos/game_103.mp4 \
        --json   data/playbyplay_PRED/fold_103/playbyplay_TAAD_val.json \
        --gt     data/playbyplay_GT/playbyplay_train.json \
        --key    game_103_H1 \
        --cvat   data/xml/game_103.xml \
        --cvat_team_map 0:home,1:away \
        --output output/game_103_vis.mp4

Event JSON formats:
    Predictions – 5 fields: [frame, team, jersey, class, confidence]
    GT          – 6 fields: [frame, team, jersey, class, visible, split]

Action classes:
    0=Background, 1=Drive, 2=Pass, 3=Cross, 4=Shot,
    5=Header, 6=Throw-in, 7=Tackle, 8=Block
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_LABELS = {
    0: "Background",
    1: "Drive",
    2: "Pass",
    3: "Cross",
    4: "Shot",
    5: "Header",
    6: "Throw-in",
    7: "Tackle",
    8: "Block",
}

# Color per action class (BGR)
ACTION_COLORS = {
    0: (160, 160, 160),  # Background – grey
    1: (50,  200,  50),  # Drive      – green
    2: (255, 150,   0),  # Pass       – orange
    3: (0,   200, 255),  # Cross      – cyan
    4: (0,    50, 255),  # Shot       – red
    5: (255, 255,   0),  # Header     – yellow
    6: (200,   0, 200),  # Throw-in   – purple
    7: (0,   100, 255),  # Tackle     – blue
    8: (255,   0, 100),  # Block      – pink
}

TEAM_LABELS = {0: "Left", 1: "Right"}

# Marker colors (BGR). Panels and on-pitch boxes share the same palette so
# colour means source (Pred / GT / match / wrong class) everywhere — never
# action class.
MARKER_PRED_COLOR     = (255, 180,  60)   # blue   — prediction only, dashed
MARKER_GT_COLOR       = ( 80, 255, 120)   # green  — ground truth only, dashed
MARKER_MATCH_COLOR    = (  0, 220, 255)   # gold   — same player, same class, solid
MARKER_MISCLASS_COLOR = ( 60,  60, 255)   # red    — same player, wrong class, solid
PANEL_TEXT_COLOR      = (230, 230, 230)   # neutral text for panels

# ---------------------------------------------------------------------------
# Panel layout
# ---------------------------------------------------------------------------

PANEL_WIDTH   = 320
PANEL_HEIGHT  = 70
PANEL_Y_START = 50   # leave room for the header label
PANEL_GAP     = 8
CORNER_RADIUS = 8
ALPHA         = 0.78  # panel transparency

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

# (frame, team, jersey, cls, conf_or_visible)
Event = Tuple[int, int, int, int, float]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_events(json_path: str, key: str, is_gt: bool = False) -> List[Event]:
    """
    Load events from a FOOTPASS JSON file.

    Predictions format (5 fields): [frame, team, jersey, class, confidence]
    GT format          (6 fields): [frame, team, jersey, class, visible, split]

    For GT files the confidence field is set to 1.0 (all GT events are certain).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if key not in data["events"]:
        available = data.get("keys", list(data["events"].keys()))
        raise KeyError(f"Key '{key}' not found. Available keys: {available}")

    raw = data["events"][key]
    events: List[Event] = []
    for entry in raw:
        frame  = int(entry[0])
        team   = int(entry[1])
        jersey = int(entry[2])
        cls    = int(entry[3])
        # GT has visible flag in field 4; predictions have confidence
        conf   = 1.0 if is_gt else float(entry[4])
        events.append((frame, team, jersey, cls, conf))
    return events


def filter_by_confidence(events: List[Event], min_confidence: float) -> List[Event]:
    return [e for e in events if e[4] >= min_confidence]


def build_frame_lookup(events: List[Event]) -> Dict[int, List[Event]]:
    lookup: Dict[int, List[Event]] = defaultdict(list)
    for event in events:
        lookup[event[0]].append(event)
    return lookup


# ---------------------------------------------------------------------------
# CVAT player tracks
# ---------------------------------------------------------------------------

# {frame_idx: {(cvat_team_str, jersey_int): (x1, y1, x2, y2)}}
CvatTracks = Dict[int, Dict[Tuple[str, int], Tuple[int, int, int, int]]]


def parse_team_map(spec: str) -> Dict[int, str]:
    """
    Parse a mapping like '0:home,1:away' into {0: 'home', 1: 'away'}.

    Predictions/GT carry team as 0/1; CVAT XML carries team as 'home'/'away'.
    """
    mapping: Dict[int, str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k, _, v = chunk.partition(":")
        v = v.strip().lower()
        if v not in {"home", "away"}:
            raise ValueError(f"team map value must be 'home' or 'away', got '{v}'")
        mapping[int(k.strip())] = v
    return mapping


def load_cvat_tracks(xml_path: str) -> CvatTracks:
    """
    Parse a CVAT-for-images XML and return per-frame player bounding boxes,
    keyed by (team, jersey).

    Frame index is parsed from the <image> element's name (e.g. 'frame_000123').
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracks: CvatTracks = {}
    for img in root.findall("image"):
        name = img.get("name", "")
        m = re.search(r"(\d+)", name)
        if not m:
            continue
        frame_idx = int(m.group(1))

        frame_players: Dict[Tuple[str, int], Tuple[int, int, int, int]] = {}
        for box in img.findall("box"):
            # CVAT labels outfielders as 'player' but keepers as 'goalkeeper'
            # and referees as 'referee' — include keepers since play-by-play
            # events can reference them, and drop only non-human labels.
            if box.get("label") not in {"player", "goalkeeper"}:
                continue
            attrs = {a.get("name"): (a.text or "") for a in box.findall("attribute")}
            team = attrs.get("team", "").strip().lower()
            jersey_raw = attrs.get("jersey", "").strip()
            if team not in {"home", "away"} or not jersey_raw:
                continue
            try:
                jersey = int(jersey_raw)
            except ValueError:
                continue
            try:
                x1 = int(round(float(box.get("xtl", "0"))))
                y1 = int(round(float(box.get("ytl", "0"))))
                x2 = int(round(float(box.get("xbr", "0"))))
                y2 = int(round(float(box.get("ybr", "0"))))
            except ValueError:
                continue
            frame_players[(team, jersey)] = (x1, y1, x2, y2)

        if frame_players:
            tracks[frame_idx] = frame_players

    return tracks


def lookup_player_box(
    tracks: CvatTracks,
    team_map: Dict[int, str],
    frame_idx: int,
    team_int: int,
    jersey: int,
    frame_offset: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Look up a player's bounding box in the CVAT tracks.

    `frame_offset` is added to the *video* frame index before indexing CVAT,
    i.e. cvat_frame = video_frame + frame_offset. Use a negative offset when
    the CVAT clip starts later in the source video than the visualization
    video, and a positive offset in the opposite case.
    """
    frame_players = tracks.get(frame_idx + frame_offset)
    if not frame_players:
        return None
    cvat_team = team_map.get(team_int)
    if cvat_team is None:
        return None
    return frame_players.get((cvat_team, jersey))


def diagnose_cvat_alignment(
    tracks: CvatTracks,
    team_map: Dict[int, str],
    pred_events: List[Event],
    gt_events: List[Event],
    frame_offset: int,
    label: str = "",
) -> Tuple[int, int, int]:
    """
    For every event, check whether the (team, jersey) lookup hits a CVAT box
    at the (offset-adjusted) frame. Returns (hits, miss_player, miss_frame).
    Misses indicate the wrong CVAT file, wrong team_map, wrong frame_offset,
    or that the player simply isn't on screen at that exact frame.
    """
    def count(events: List[Event]) -> Tuple[int, int, int]:
        h = mp = mf = 0
        for f, team, jersey, _cls, _ in events:
            cvat_frame = f + frame_offset
            frame_players = tracks.get(cvat_frame)
            if not frame_players:
                mf += 1
                continue
            cvat_team = team_map.get(team)
            if cvat_team is not None and (cvat_team, jersey) in frame_players:
                h += 1
            else:
                mp += 1
        return h, mp, mf

    pred_stats = count(pred_events)
    gt_stats = count(gt_events)
    total = (
        pred_stats[0] + pred_stats[1] + pred_stats[2]
        + gt_stats[0] + gt_stats[1] + gt_stats[2]
    )
    print(f"  CVAT alignment check{(' ' + label) if label else ''}:")
    print(f"    predictions  hits={pred_stats[0]}  miss_player={pred_stats[1]}  miss_frame={pred_stats[2]}")
    print(f"    ground truth hits={gt_stats[0]}  miss_player={gt_stats[1]}  miss_frame={gt_stats[2]}")
    if total and (pred_stats[0] + gt_stats[0]) == 0:
        print("    WARNING: zero hits — wrong --cvat file, wrong --cvat_team_map, "
              "or wrong --cvat_frame_offset.")
    return (pred_stats[0] + gt_stats[0],
            pred_stats[1] + gt_stats[1],
            pred_stats[2] + gt_stats[2])


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_rounded_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                      radius: int, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_header(img: np.ndarray, text: str, x: int, color: Tuple[int, int, int]) -> np.ndarray:
    """Draw a small label above the panel column (e.g. 'PREDICTED' / 'GROUND TRUTH')."""
    cv2.putText(img, text, (x, PANEL_Y_START - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return img


def draw_dashed_rect(
    img: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash: int = 10,
    gap: int = 6,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    step = dash + gap
    for x in range(x1, x2, step):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, step):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_player_marker(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    kind: str,  # 'pred' | 'gt' | 'match'
    jersey: int,
) -> None:
    """
    Mark the acting player on the pitch.
        pred     -> dashed blue  (pred only, no matching GT)
        gt       -> dashed green (GT only, no matching pred)
        match    -> solid gold   (same player + same class)
        misclass -> solid red    (same player, wrong class)
    """
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if kind == "gt":
        color = MARKER_GT_COLOR
        draw_dashed_rect(img, (x1, y1), (x2, y2), color, thickness=3)
        label_below = True
    elif kind == "pred":
        color = MARKER_PRED_COLOR
        draw_dashed_rect(img, (x1, y1), (x2, y2), color, thickness=3)
        label_below = False
    elif kind == "match":
        color = MARKER_MATCH_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 1)
        label_below = False
    else:  # 'misclass'
        color = MARKER_MISCLASS_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 1)
        label_below = False

    # Downward arrow / triangle above the head
    cx = (x1 + x2) // 2
    arrow_top = max(0, y1 - 22)
    arrow_tip = max(0, y1 - 4)
    pts = np.array(
        [[cx - 8, arrow_top], [cx + 8, arrow_top], [cx, arrow_tip]],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=1)

    # Label rendering
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    if label_below:
        ly = min(h - 4, y2 + th + 6)
    else:
        ly = max(th + 4, arrow_top - 4)
    bg_x1 = x1
    bg_x2 = x1 + tw + 8
    bg_y1 = ly - th - 4
    bg_y2 = ly + 4
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
    cv2.putText(
        img, label, (bg_x1 + 4, ly),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
    )


def draw_panels(
    frame_img: np.ndarray,
    active_events: List[Event],
    event_slots: Dict[Tuple[int, int, int, int], int],
    current_frame: int,
    display_frames: int,
    anchor_x: int,
    is_gt: bool,
) -> np.ndarray:
    """
    Draw event panels in fixed grid slots. `event_slots` maps an event key
    (frame, team, jersey, cls) to its row index so events keep their slot
    for their whole on-screen life — no shuffling when earlier events expire.
    """
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
        if y2 > img_h - 10:
            continue
        x1 = anchor_x
        x2 = anchor_x + PANEL_WIDTH

        img = draw_rounded_rect(img, x1, y1, x2, y2, CORNER_RADIUS, (20, 20, 20), ALPHA)

        if is_gt:
            img = draw_rounded_rect(img, x2 - 8, y1, x2, y2, CORNER_RADIUS, accent_color, 0.9)
        else:
            img = draw_rounded_rect(img, x1, y1, x1 + 8, y2, CORNER_RADIUS, accent_color, 0.9)

        action_name = ACTION_LABELS.get(cls, f"Class {cls}")
        cv2.putText(img, action_name,
                    (x1 + 18, y1 + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.80,
                    PANEL_TEXT_COLOR, 2, cv2.LINE_AA)

        team_str = f"Team: {TEAM_LABELS.get(team, str(team))}   #{jersey}"
        cv2.putText(img, team_str,
                    (x1 + 18, y1 + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (200, 200, 200), 1, cv2.LINE_AA)

        if not is_gt:
            conf_str = f"{conf * 100:.0f}%"
            cv2.putText(img, conf_str,
                        (x2 - 58, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (160, 160, 160), 1, cv2.LINE_AA)

        frames_remaining = max(0, (evt_frame + display_frames) - current_frame)
        progress  = frames_remaining / display_frames
        bar_max_w = PANEL_WIDTH - 16
        bar_w     = int(bar_max_w * progress)
        bar_y     = y2 - 6
        cv2.rectangle(img, (x1 + 8, bar_y), (x1 + 8 + bar_max_w, bar_y + 4), (60, 60, 60), -1)
        cv2.rectangle(img, (x1 + 8, bar_y), (x1 + 8 + bar_w,     bar_y + 4), accent_color, -1)

    return img


def assign_slots(
    active_events: List[Event],
    slot_map: Dict[Tuple[int, int, int, int], int],
    max_slots: int,
) -> None:
    """
    Keep assigned slots stable across frames; give new events the lowest
    free index (list-semantics, top-to-bottom fill). Expired events are
    removed by the caller before this runs.
    """
    active_keys = {(e[0], e[1], e[2], e[3]) for e in active_events}
    for key in list(slot_map.keys()):
        if key not in active_keys:
            del slot_map[key]
    used = set(slot_map.values())
    for evt in active_events:
        key = (evt[0], evt[1], evt[2], evt[3])
        if key in slot_map:
            continue
        for i in range(max_slots):
            if i not in used:
                slot_map[key] = i
                used.add(i)
                break


def draw_legend(img: np.ndarray) -> None:
    """Bottom-left overlay explaining what each marker colour/style means."""
    entries = [
        ("Pred only (dashed)",  MARKER_PRED_COLOR,     False),
        ("GT only (dashed)",    MARKER_GT_COLOR,       False),
        ("Match (solid)",       MARKER_MATCH_COLOR,    True),
        ("Wrong class (solid)", MARKER_MISCLASS_COLOR, True),
    ]
    h, w = img.shape[:2]
    pad = 10
    row_h = 22
    box_w = 230
    box_h = pad * 2 + row_h * len(entries) + 20
    x1 = 20
    y2 = h - 20
    y1 = y2 - box_h
    x2 = x1 + box_w
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, dst=img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), 1)
    cv2.putText(img, "LEGEND", (x1 + pad, y1 + pad + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    for i, (label, color, solid) in enumerate(entries):
        ry = y1 + pad + 26 + i * row_h
        sx1, sy1 = x1 + pad, ry
        sx2, sy2 = sx1 + 28, ry + 14
        if solid:
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), color, 2)
        else:
            draw_dashed_rect(img, (sx1, sy1), (sx2, sy2), color, thickness=2, dash=5, gap=3)
        cv2.putText(img, label, (sx2 + 8, sy2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    json_path: str,
    key: str,
    output_path: str,
    display_seconds: float,
    fps_override: Optional[float],
    min_confidence: float,
    gt_path: Optional[str],
    cvat_path: Optional[str] = None,
    cvat_team_map: Optional[Dict[int, str]] = None,
    cvat_frame_offset: int = 0,
    show: str = "both",
) -> None:
    # --- Load predictions ---
    print(f"Loading predictions: {json_path}  (key={key})")
    pred_events = load_events(json_path, key, is_gt=False)
    pred_events = filter_by_confidence(pred_events, min_confidence)
    pred_lookup = build_frame_lookup(pred_events)
    print(f"  {len(pred_events)} predictions (conf >= {min_confidence})")

    # --- Load GT (optional) ---
    gt_lookup: Dict[int, List[Event]] = {}
    gt_events: List[Event] = []
    if gt_path:
        print(f"Loading ground truth: {gt_path}  (key={key})")
        gt_events = load_events(gt_path, key, is_gt=True)
        gt_lookup = build_frame_lookup(gt_events)
        print(f"  {len(gt_events)} GT events")

    # --- Load CVAT player tracks (optional) ---
    cvat_tracks: CvatTracks = {}
    team_map: Dict[int, str] = cvat_team_map or {}
    if cvat_path:
        print(f"Loading CVAT tracks: {cvat_path}")
        cvat_tracks = load_cvat_tracks(cvat_path)
        if cvat_tracks:
            covered = sorted(cvat_tracks.keys())
            print(
                f"  {len(cvat_tracks)} annotated frames "
                f"(range {covered[0]}–{covered[-1]})"
            )
        else:
            print("  No player annotations found in CVAT file.")
        if not team_map:
            team_map = {0: "home", 1: "away"}
        print(f"  Team mapping (pred -> CVAT): {team_map}")
        print(f"  CVAT frame offset (cvat_frame = video_frame + offset): {cvat_frame_offset}")
        diagnose_cvat_alignment(
            cvat_tracks, team_map,
            pred_events=pred_events,
            gt_events=gt_events,
            frame_offset=cvat_frame_offset,
        )

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    display_frames = int(display_seconds * fps)

    print(f"  Video: {width}x{height} @ {fps:.1f} fps  ({total} frames)")
    print(f"  Display duration: {display_seconds}s = {display_frames} frames")
    print(f"  Output: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Anchor positions
    pred_anchor_x = 20                        # predictions: left side
    gt_anchor_x   = width - PANEL_WIDTH - 20  # GT: right side

    show_pred = show in ("pred", "both")
    show_gt   = show in ("gt", "both") and gt_path is not None

    active_pred: List[Event] = []
    active_gt:   List[Event] = []
    # event_key -> locked (w, h) from the first CVAT hit at activation time.
    # Size is frozen for the rest of the event's lifetime so boxes don't
    # pulse as the player scales with the camera.
    locked_size: Dict[Tuple[int, int, int, int], Tuple[int, int]] = {}
    pred_slots: Dict[Tuple[int, int, int, int], int] = {}
    gt_slots:   Dict[Tuple[int, int, int, int], int] = {}
    max_slots = max(1, (height - PANEL_Y_START - 10) // (PANEL_HEIGHT + PANEL_GAP))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Activate new events for this frame
        if show_pred and frame_idx in pred_lookup:
            active_pred.extend(pred_lookup[frame_idx])
        if show_gt and frame_idx in gt_lookup:
            active_gt.extend(gt_lookup[frame_idx])

        # Expire old events
        active_pred = [e for e in active_pred if frame_idx < e[0] + display_frames]
        active_gt   = [e for e in active_gt   if frame_idx < e[0] + display_frames]

        assign_slots(active_pred, pred_slots, max_slots)
        assign_slots(active_gt,   gt_slots,   max_slots)

        # Drop locked sizes for events that have expired.
        active_keys = {(e[0], e[1], e[2], e[3]) for e in active_pred + active_gt}
        for k in list(locked_size.keys()):
            if k not in active_keys:
                del locked_size[k]

        def marker_box(evt: Event) -> Optional[Tuple[int, int, int, int]]:
            """Look up the current CVAT box, then reshape to the locked size."""
            team_, jersey_ = evt[1], evt[2]
            cur = lookup_player_box(
                cvat_tracks, team_map, frame_idx, team_, jersey_,
                frame_offset=cvat_frame_offset,
            )
            if cur is None:
                return None
            key = (evt[0], evt[1], evt[2], evt[3])
            if key not in locked_size:
                locked_size[key] = (cur[2] - cur[0], cur[3] - cur[1])
            lw, lh = locked_size[key]
            cx = (cur[0] + cur[2]) // 2
            cy = (cur[1] + cur[3]) // 2
            return (cx - lw // 2, cy - lh // 2, cx + lw // 2, cy + lh // 2)

        # Mark acting players on the pitch (under the panels so the
        # text overlays stay readable).
        if cvat_tracks:
            # Group active events per (team, jersey) so we can detect when
            # the same player is named by both a prediction and a GT event.
            pred_by_player: Dict[Tuple[int, int], Event] = {}
            for evt in active_pred:
                pred_by_player[(evt[1], evt[2])] = evt
            gt_by_player: Dict[Tuple[int, int], Event] = {}
            for evt in active_gt:
                gt_by_player[(evt[1], evt[2])] = evt

            matched_players = set(pred_by_player) & set(gt_by_player)

            # Matches first (gold for same-class, red for class mismatch)
            for player_key in matched_players:
                team, jersey = player_key
                pred_evt = pred_by_player[player_key]
                gt_evt = gt_by_player[player_key]
                box = marker_box(pred_evt)
                if box is None:
                    box = marker_box(gt_evt)
                if box is None:
                    continue
                pred_action = ACTION_LABELS.get(pred_evt[3], f"Class {pred_evt[3]}")
                gt_action = ACTION_LABELS.get(gt_evt[3], f"Class {gt_evt[3]}")
                if pred_evt[3] == gt_evt[3]:
                    label = f"MATCH #{jersey} {pred_action}"
                    kind = "match"
                else:
                    label = f"WRONG CLASS #{jersey} P:{pred_action} / G:{gt_action}"
                    kind = "misclass"
                draw_player_marker(frame, box, label=label, kind=kind, jersey=jersey)

            # Predictions only (dashed blue)
            for player_key, evt in pred_by_player.items():
                if player_key in matched_players:
                    continue
                _, jersey = player_key
                box = marker_box(evt)
                if box is None:
                    continue
                action_name = ACTION_LABELS.get(evt[3], f"Class {evt[3]}")
                draw_player_marker(
                    frame, box,
                    label=f"PRED #{jersey} {action_name}",
                    kind="pred", jersey=jersey,
                )

            # GT only (dashed green)
            for player_key, evt in gt_by_player.items():
                if player_key in matched_players:
                    continue
                _, jersey = player_key
                box = marker_box(evt)
                if box is None:
                    continue
                action_name = ACTION_LABELS.get(evt[3], f"Class {evt[3]}")
                draw_player_marker(
                    frame, box,
                    label=f"GT #{jersey} {action_name}",
                    kind="gt", jersey=jersey,
                )

        # Panels
        if show_pred and active_pred:
            frame = draw_panels(frame, active_pred, pred_slots, frame_idx,
                                display_frames, anchor_x=pred_anchor_x, is_gt=False)
        if show_gt and active_gt:
            frame = draw_panels(frame, active_gt, gt_slots, frame_idx,
                                display_frames, anchor_x=gt_anchor_x, is_gt=True)

        draw_legend(frame)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 500 == 0:
            pct = (frame_idx / total * 100) if total > 0 else 0
            print(f"  {frame_idx}/{total} frames ({pct:.1f}%)")

    cap.release()
    writer.release()
    print(f"Done. Saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize FOOTPASS events on video. "
                    "Predictions on the LEFT, Ground Truth on the RIGHT."
    )
    parser.add_argument("--video",  required=True,
                        help="Path to input video file")
    parser.add_argument("--json",   required=True,
                        help="Path to predictions JSON (5-field FOOTPASS format)")
    parser.add_argument("--gt",     default=None,
                        help="Path to GT JSON (6-field FOOTPASS format). Optional.")
    parser.add_argument("--key",    required=True,
                        help="Game key, e.g. 'game_103_H1'")
    parser.add_argument("--output", required=True,
                        help="Path to output video file (.mp4)")
    parser.add_argument("--display_seconds", type=float, default=3.0,
                        help="Seconds each event overlay stays visible (default: 3.0)")
    parser.add_argument("--fps",    type=float, default=None,
                        help="Override FPS. FOOTPASS standard is 25 (default: read from video)")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="Min confidence for predictions (default: 0.0 = show all)")
    parser.add_argument("--cvat", default=None,
                        help="Path to CVAT-for-images XML with player tracks. "
                             "When supplied, the acting player is marked on the pitch.")
    parser.add_argument("--cvat_team_map", default="0:home,1:away",
                        help="Map prediction team ids to CVAT team strings, "
                             "e.g. '0:home,1:away' (default) or '0:away,1:home'.")
    parser.add_argument("--show", choices=["pred", "gt", "both"], default="both",
                        help="Which overlays to render: predictions only, "
                             "GT only, or both (default). Use this to produce "
                             "separate videos for toggle-style comparison.")
    parser.add_argument("--cvat_frame_offset", type=int, default=0,
                        help="Offset added to the video frame index before "
                             "looking up CVAT (cvat_frame = video_frame + offset). "
                             "Use this if the CVAT clip and the video do not "
                             "start at the same point in the source recording.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    team_map = parse_team_map(args.cvat_team_map) if args.cvat else None
    process_video(
        video_path=args.video,
        json_path=args.json,
        key=args.key,
        output_path=args.output,
        display_seconds=args.display_seconds,
        fps_override=args.fps,
        min_confidence=args.min_confidence,
        gt_path=args.gt,
        cvat_path=args.cvat,
        cvat_team_map=team_map,
        cvat_frame_offset=args.cvat_frame_offset,
        show=args.show,
    )
