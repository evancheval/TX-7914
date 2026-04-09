from ultralytics import YOLO  # type: ignore
import argparse
import cv2
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare basic YOLO detection vs YOLO + tracker side by side."
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
        default="botsort.yaml",
        help="Tracker configuration file or path to custom YAML (default: botsort.yaml)",
    )
    return parser.parse_args()


BOX_COLOR = (255, 77, 54)
TRACKER_BOX_COLOR = (155, 255, 155)
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


args = parse_args()

file_path = args.input
model_source = args.model_path

# Two separate model instances to avoid state interference
model_detect = YOLO(model_source)
model_track = YOLO(model_source)

model_detect.overrides['classes'] = 0
model_track.overrides['classes'] = 0

# Streams for both models
results_detect = model_detect.predict(source=file_path, stream=True, verbose=False)
results_track = model_track.track(source=file_path, stream=True, verbose=False,
                                   persist=True, tracker=args.tracker)

# Counters
detect_lost_total = 0
track_lost_total = 0
detect_prev_count = 0
track_prev_count = 0

# For tracking unique IDs that disappeared (tracker side)
track_seen_ids: set[int] = set()
track_disappeared_ids: set[int] = set()

# Video writer
writer = None

try:
    for r_detect, r_track in zip(results_detect, results_track):
        # --- Detection side (left): count box drops ---
        detect_count = len(r_detect.boxes.xyxy) if r_detect.boxes is not None else 0
        detect_drop = max(0, detect_prev_count - detect_count)
        detect_lost_total += detect_drop
        detect_prev_count = detect_count

        # --- Tracker side (right): count box drops ---
        track_count = len(r_track.boxes.xyxy) if r_track.boxes is not None else 0
        track_drop = max(0, track_prev_count - track_count)
        track_lost_total += track_drop
        track_prev_count = track_count

        # Track unique ID disappearances
        current_ids: set[int] = set()
        if r_track.boxes is not None and r_track.boxes.id is not None:
            current_ids = set(r_track.boxes.id.cpu().numpy().astype(int).tolist())
        # IDs that were seen before but not in current frame
        newly_disappeared = track_seen_ids - current_ids
        track_disappeared_ids.update(newly_disappeared)
        track_seen_ids = current_ids | track_seen_ids  # accumulate all seen IDs
        # IDs currently disappeared = all ever seen minus currently visible
        currently_missing = track_seen_ids - current_ids

        if args.show or args.save:
            # Left frame: basic detection
            left_frame = r_detect.orig_img.copy()
            if r_detect.boxes is not None and len(r_detect.boxes.xyxy) > 0:
                draw_boxes(left_frame, r_detect.boxes, BOX_COLOR)
            write_title(left_frame, title="YOLO Detection (no tracker)")
            write_counter(left_frame, "Lost", detect_lost_total, y_offset=0.05)
            write_counter(left_frame, "Boxes", detect_count, y_offset=0.10)

            # Right frame: tracker
            right_frame = r_track.orig_img.copy()
            if r_track.boxes is not None and len(r_track.boxes.xyxy) > 0:
                draw_boxes(right_frame, r_track.boxes, TRACKER_BOX_COLOR)
            tracker_name = args.tracker.replace(".yaml", "").upper()
            write_title(right_frame, title=f"YOLO + {tracker_name}")
            write_counter(right_frame, "Lost", track_lost_total, y_offset=0.05)
            write_counter(right_frame, "Boxes", track_count, y_offset=0.10)
            write_counter(right_frame, "Missing IDs", len(currently_missing), y_offset=0.15)

            concat_frame = cv2.hconcat([left_frame, right_frame])

            if args.save and writer is None:
                h, w = concat_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter("tracker_comparison_output.mp4", fourcc, 30, (w, h))

            if args.save and writer is not None:
                writer.write(concat_frame)

            if args.show:
                cv2.imshow("Detection vs Tracker", concat_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

except KeyboardInterrupt:
    print("Process interrupted by user.")
finally:
    print(f"\n--- Results ---")
    print(f"Detection (no tracker) - Total lost boxes: {detect_lost_total}")
    print(f"Tracker ({args.tracker})    - Total lost boxes: {track_lost_total}")
    print(f"Tracker - Unique IDs that disappeared at least once: {len(track_disappeared_ids)}")
    if writer is not None:
        writer.release()
    if args.show:
        time.sleep(1)
        cv2.destroyAllWindows()
