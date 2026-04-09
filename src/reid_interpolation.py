from ultralytics import YOLO  # type: ignore
from ultralytics.engine.results import Results
import argparse
import cv2
import numpy as np
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO + ReID tracker with linear interpolation for missing boxes."
    )
    parser.add_argument("input", help="Path to the input media file")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show comparison output window",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save comparison output to disk",
    )
    parser.add_argument(
        "--model_path",
        default="models/yolo26n.pt",
        help="Path to YOLO model weights",
    )
    parser.add_argument(
        "--tracker",
        default="botsort_reid.yaml",
        help="Tracker configuration file (default: botsort_reid.yaml)",
    )
    parser.add_argument(
        "--gap_frame",
        type=int,
        default=10,
        help="Number of future frames to look ahead for interpolation",
    )
    return parser.parse_args()


BOX_COLOR = (255, 77, 54)
INTERP_BOX_COLOR = (155, 155, 255)
TITLE_COLOR = (255, 255, 255)
COUNTER_COLOR = (155, 155, 255)


def write_title(frame, title="", color=TITLE_COLOR):
    frame_h, _ = frame.shape[:2]
    cv2.putText(frame, title, (int(0.02 * frame_h), int(0.05 * frame_h)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def write_counter(frame, label, value, y_offset=0.05, color=COUNTER_COLOR):
    frame_h, frame_w = frame.shape[:2]
    text = f"{label}: {value}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 2
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = frame_w - text_w - 20
    y = int(y_offset * frame_h)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


def draw_boxes(frame, boxes, color=BOX_COLOR):
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
    for (x1, y1, x2, y2), track_id in zip(xyxy, ids):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{track_id}" if track_id is not None else "X"
        cv2.putText(frame, label, ((x1 + x2) // 2 - 5 * len(label), (y1 + y2) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_interpolated_boxes(frame, interpolated_boxes, color=INTERP_BOX_COLOR):
    for (box, track_id) in interpolated_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{track_id}*"
        cv2.putText(frame, label, ((x1 + x2) // 2 - 5 * len(label), (y1 + y2) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# --- Interpolation functions (from interpolation.py) ---

def get_ids_at(results_list: list[Results], index: int) -> set[int]:
    boxes = results_list[index].boxes
    if boxes.id is None:
        return set()
    return set(boxes.id.cpu().numpy().astype(int).tolist())


def get_box_for_id(results_list: list[Results], index: int, track_id: int) -> np.ndarray | None:
    boxes = results_list[index].boxes
    if boxes.id is None:
        return None
    ids = boxes.id.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    for i, tid in enumerate(ids):
        if tid == track_id:
            return xyxy[i]
    return None


def interpolate_missing_boxes(results_list: list[Results], target_index: int) -> list[tuple[np.ndarray, int]]:
    """For IDs missing at target_index but present before AND after,
    compute linearly interpolated bounding boxes."""
    target_ids = get_ids_at(results_list, target_index)

    before_ids = get_ids_at(results_list, 0)
    missing_ids = before_ids - target_ids
    interpolated = []

    for track_id in missing_ids:
        last_before = None
        if track_id in get_ids_at(results_list, 0):
            last_before = 0
        if last_before is None:
            continue

        first_after = None
        for i in range(target_index + 1, len(results_list)):
            if track_id in get_ids_at(results_list, i):
                first_after = i
                break
        if first_after is None:
            continue

        box_before = get_box_for_id(results_list, last_before, track_id)
        box_after = get_box_for_id(results_list, first_after, track_id)
        if box_before is None or box_after is None:
            continue

        total_gap = first_after - last_before
        t = (target_index - last_before) / total_gap
        interp_box = box_before + t * (box_after - box_before)
        interpolated.append((interp_box.astype(int), track_id))

    return interpolated


# --- Main ---

args = parse_args()

target_index = 1
length_of_buffer = args.gap_frame + 2

# Left side: basic YOLO tracker (default botsort, no ReID)
model_base = YOLO(args.model_path)
model_base.overrides['classes'] = 0

# Right side: ReID tracker + interpolation
model_reid = YOLO(args.model_path)
model_reid.overrides['classes'] = 0

# Buffers (both sides need same delay for sync)
base_buffer: list[Results] = []
reid_buffer: list[Results] = []

# Stats: base tracker (left)
base_lost_total = 0
base_prev_count = 0
base_all_ids: set[int] = set()

# Stats: ReID + interpolation (right)
reid_lost_total = 0
reid_interp_lost_total = 0
reid_prev_count = 0
reid_all_ids: set[int] = set()

writer = None

try:
    results_base = model_base.track(source=args.input, stream=True, verbose=False,
                                     persist=True, tracker="botsort.yaml")
    results_reid = model_reid.track(source=args.input, stream=True, verbose=False,
                                     persist=True, tracker=args.tracker)

    for r_base, r_reid in zip(results_base, results_reid):
        # Buffer both sides equally
        base_buffer.append(r_base)
        reid_buffer.append(r_reid)
        if len(base_buffer) > length_of_buffer:
            base_buffer.pop(0)
        if len(reid_buffer) > length_of_buffer:
            reid_buffer.pop(0)

        if len(reid_buffer) < 2:
            continue

        interpolating = len(reid_buffer) >= length_of_buffer

        # --- Left side: base tracker (same delay as right) ---
        res_base = base_buffer[target_index] if interpolating else r_base
        base_count = len(res_base.boxes.xyxy) if res_base.boxes is not None else 0
        base_ids: set[int] = set()
        if res_base.boxes is not None and res_base.boxes.id is not None:
            base_ids = set(res_base.boxes.id.cpu().numpy().astype(int).tolist())
        base_all_ids.update(base_ids)
        base_drop = max(0, base_prev_count - base_count)
        base_lost_total += base_drop
        base_prev_count = base_count

        # --- Right side: ReID + interpolation (same delay) ---
        res_reid = reid_buffer[target_index] if interpolating else r_reid
        reid_count = len(res_reid.boxes.xyxy) if res_reid.boxes is not None else 0
        current_reid_ids: set[int] = set()
        if res_reid.boxes is not None and res_reid.boxes.id is not None:
            current_reid_ids = set(res_reid.boxes.id.cpu().numpy().astype(int).tolist())
        reid_all_ids.update(current_reid_ids)

        reid_drop = max(0, reid_prev_count - reid_count)
        reid_lost_total += reid_drop

        interp_boxes: list[tuple[np.ndarray, int]] = []
        if interpolating:
            interp_boxes = interpolate_missing_boxes(reid_buffer, target_index)

        interp_drop = max(0, reid_drop - len(interp_boxes))
        reid_interp_lost_total += interp_drop
        reid_prev_count = reid_count

        if args.show or args.save:
            # Left: base YOLO tracker
            left_frame = res_base.orig_img.copy()
            if res_base.boxes is not None and len(res_base.boxes.xyxy) > 0:
                draw_boxes(left_frame, res_base.boxes, BOX_COLOR)
            write_title(left_frame, title="YOLO Base Tracker")
            write_counter(left_frame, "Lost boxes", base_lost_total, y_offset=0.05)
            write_counter(left_frame, "Unique IDs", len(base_all_ids), y_offset=0.10)
            write_counter(left_frame, "Boxes", base_count, y_offset=0.15)

            # Right: ReID + interpolation
            right_frame = res_reid.orig_img.copy()
            if res_reid.boxes is not None and len(res_reid.boxes.xyxy) > 0:
                draw_boxes(right_frame, res_reid.boxes, BOX_COLOR)
            if interp_boxes:
                draw_interpolated_boxes(right_frame, interp_boxes)
            write_title(right_frame, title="ReID + Interpolation")
            write_counter(right_frame, "Lost boxes", reid_interp_lost_total, y_offset=0.05)
            write_counter(right_frame, "Unique IDs", len(reid_all_ids), y_offset=0.10)
            write_counter(right_frame, "Boxes", reid_count + len(interp_boxes), y_offset=0.15)

            concat_frame = cv2.hconcat([left_frame, right_frame])

            if args.save and writer is None:
                h, w = concat_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter("reid_interpolation_output.mp4", fourcc, 30, (w, h))

            if args.save and writer is not None:
                writer.write(concat_frame)

            if args.show:
                cv2.imshow("Base vs ReID + Interpolation", concat_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

except KeyboardInterrupt:
    print("Process interrupted by user.")
finally:
    print(f"\n--- Results ---")
    print(f"Base tracker        - Lost boxes: {base_lost_total} | Unique IDs: {len(base_all_ids)}")
    print(f"ReID + interpolation - Lost boxes: {reid_interp_lost_total} | Unique IDs: {len(reid_all_ids)}")
    if writer is not None:
        writer.release()
    if args.show:
        time.sleep(1)
        cv2.destroyAllWindows()
