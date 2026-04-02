from ultralytics import YOLO #type:ignore
import argparse
import cv2
import numpy as np
from ultralytics.engine.results import Boxes, Results
import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO tracking on an input media file."
    )
    parser.add_argument("input", help="Path to the input media file")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show tracking output window",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save tracking output to disk",
    )
    parser.add_argument(
        "--model_path",
        default="models/yolo26n.pt",
        help="Path to YOLO model weights",
    )
    parser.add_argument(
        "--gap_frame",
        type=int,
        default=10,
        help="Maximal number of frames to keep a track of (used for box interpolation in case of missing boxes) "
    )
    parser.add_argument(
        "--max-box-shift",
        type=int,
        default=10,
        help = "Maximal number of pixel shift to consider a box the same between two frames (used for box interpolation in case of missing boxes). The default value is 10, based on the original work of this script using 360x640 frames."
    )
    return parser.parse_args()


BOX_COLOR = (255, 77, 54)
ADDITIONAL_BOX_COLOR = (155, 155, 255)
TITLE_COLOR = (255, 255, 255)

def write_title(frame, title="", color=TITLE_COLOR):
    frame_h, frame_w = frame.shape[:2]
    cv2.putText(frame, title, (int(0.02 * frame_h), int(0.05 * frame_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def write_lost_counter(frame, count, color=ADDITIONAL_BOX_COLOR):
    frame_h, frame_w = frame.shape[:2]
    label = f"Lost: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 2
    (text_w, _), _ = cv2.getTextSize(label, font, scale, thickness)
    x = frame_w - text_w - 20
    y = int(0.05 * frame_h)
    cv2.putText(frame, label, (x, y), font, scale, color, thickness)

def draw_boxes(frame, boxes, color=BOX_COLOR):
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    ids  = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
    confs = boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), track_id, conf in zip(xyxy, ids, confs):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{track_id}" if track_id is not None else "X"
        cv2.putText(frame, label, ((x1 + x2)//2 - 5*len(label), (y1 + y2)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

###################################################
# Linear interpolation on nearest box method, not tested yet
  
def find_near_boxes(boxes: Boxes, target_box: list, max_shift: int):
    target_x1, target_y1, target_x2, target_y2 = target_box
    near_boxes = []
    for box in boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = box
        if (abs(x1 - target_x1) <= max_shift and abs(y1 - target_y1) <= max_shift and
            abs(x2 - target_x2) <= max_shift and abs(y2 - target_y2) <= max_shift):
            near_boxes.append(box)
    return near_boxes

def find_nearest_box(boxes: Boxes, target_box: list, max_shift: int):
    near_boxes = find_near_boxes(boxes, target_box, max_shift)
    if not near_boxes:
        return None
    target_x1, target_y1, target_x2, target_y2 = target_box
    nearest_box = min(near_boxes, key=lambda box: ((box[0] - target_x1) ** 2 + (box[1] - target_y1) ** 2 + 
                                                     (box[2] - target_x2) ** 2 + (box[3] - target_y2) ** 2) ** 0.5)
    return nearest_box

###################################################

###################################################
# Linear interpolation on missing id-box method

def lost_ids(results_list: list[Results], target_index: int) -> list[int]:
    target_ids = results_list[target_index].boxes.id.cpu().numpy().astype(int)
    lost_ids = []
    for i in range(target_index-1, -1, -1):
        actual_ids = results_list[i].boxes.id.cpu().numpy().astype(int)
        for actual_id in actual_ids:
            if actual_id not in target_ids and actual_id not in lost_ids:
                lost_ids.append(actual_id)
    return lost_ids

def lost_bounding_box(results_list: list[Results], target_index: int) -> bool:
    n_boxes_on_target_frame = len(results_list[target_index].boxes.xyxy.cpu().numpy())
    for i in range(target_index-1, -1, -1):
        n_boxes_on_actual_frame = len(results_list[i].boxes.xyxy.cpu().numpy())
        if n_boxes_on_actual_frame < n_boxes_on_target_frame:
            return True
    return False

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
    compute linearly interpolated bounding boxes.
    Returns list of (xyxy, track_id) tuples."""
    target_ids = get_ids_at(results_list, target_index)

    # All IDs seen in frames before target
    before_ids = get_ids_at(results_list, 0)
    missing_ids = before_ids - target_ids
    interpolated = []

    for track_id in missing_ids:
        # Last frame before target where this ID was present
        last_before = None
        if track_id in get_ids_at(results_list, 0):
            last_before = 0
        if last_before is None:
            continue

        # First frame after target where this ID reappears
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

        # Linear interpolation
        total_gap = first_after - last_before
        t = (target_index - last_before) / total_gap
        interp_box = box_before + t * (box_after - box_before)
        interpolated.append((interp_box.astype(int), track_id))

    return interpolated

def draw_interpolated_boxes(frame, interpolated_boxes, color=ADDITIONAL_BOX_COLOR):
    for (box, track_id) in interpolated_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{track_id}*"
        cv2.putText(frame, label, ((x1 + x2)//2 - 5*len(label), (y1 + y2)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

###################################################


args = parse_args()

target_index = 1 # Second image of the buffer, as the first frame is the previous

file_path = args.input
model_source = args.model_path

model = YOLO(model_source)

try:
    model.overrides['classes'] = 0

    results_trough_time = []
    og_lost_total = 0
    interp_lost_total = 0
    prev_box_count = 0
    length_of_buffer = args.gap_frame + 2 # first frame is the previous, second frame is the target, and the rest are future frames for interpolation

    results = model.track(source=file_path, save=args.save, stream=True, verbose=False, show_labels=False)
    for r in results:
        # If the following condition is not true, it means that we don't have enough frames to start interpolation,
        # but that we still got the first frame to interpolate on, then we wait until we have enough frames to start
        # interpolation on it (collecting "future frames")
        if len(results_trough_time)!=0:
            interpolating: bool = len(results_trough_time) >= length_of_buffer
            if interpolating:
                res = results_trough_time[target_index]
            else:
                # Not enough frames for interpolation, use the current result
                res = r

            # Count lost bounding boxes (drop vs previous frame)
            current_box_count = len(res.boxes.xyxy) if res.boxes is not None else 0
            og_drop = max(0, prev_box_count - current_box_count)
            og_lost_total += og_drop

            if args.show:
                boxes = res.boxes
                og_frame_w_boxes = res.orig_img.copy()
                new_frame = res.orig_img.copy()


                if boxes is not None and boxes.xyxy is not None:
                    draw_boxes(og_frame_w_boxes, boxes)
                    write_title(og_frame_w_boxes, title="Original box from the model")
                    draw_boxes(new_frame, boxes)
                    write_title(new_frame, title="Output")
                    if interpolating:
                        interp_boxes = interpolate_missing_boxes(results_trough_time, target_index)
                        if interp_boxes:
                            draw_interpolated_boxes(new_frame, interp_boxes)
                    else:
                        interp_boxes = []
                    

                    interp_drop = max(0, og_drop - len(interp_boxes))
                    interp_lost_total += interp_drop

                    write_lost_counter(og_frame_w_boxes, og_lost_total)
                    write_lost_counter(new_frame, interp_lost_total)

                concat_frame = cv2.hconcat([og_frame_w_boxes, new_frame])
                cv2.imshow("Tracking", concat_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            prev_box_count = current_box_count

        results_trough_time.append(r)
        if len(results_trough_time) > length_of_buffer:
            results_trough_time.pop(0)

except KeyboardInterrupt:
    print("Process interrupted by user.")
finally:
    if args.show:
        time.sleep(1)
        cv2.destroyAllWindows()