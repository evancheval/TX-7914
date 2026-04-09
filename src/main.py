import os
from ultralytics import YOLO #type:ignore
from typing import Generator
import argparse
import cv2
import numpy as np
from ultralytics.engine.results import Boxes, Results
import time
import csv

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
        help="Save tracking output to disk, csv format",
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

class ReductedResults():
    def __init__(self, frame_index: int, orig_img: np.ndarray, xyxys: np.ndarray, confs: np.ndarray | None, ids: np.ndarray | None):
        self.frame_index = frame_index
        self.orig_img = orig_img
        self.xyxys = xyxys
        self.confs = confs if confs is not None else [None] * len(xyxys)
        self.ids : list[int|None] = list(ids.astype(int)) if ids is not None else [None] * len(xyxys)
    
    def from_boxes(frame_index: int, orig_img: np.ndarray, boxes: Boxes) -> ReductedResults:
        return ReductedResults(
            frame_index,
            orig_img,
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy() if boxes.conf is not None else None,
            boxes.id.cpu().numpy() if boxes.id is not None else None
        )
    
    def get_number_of_boxes(self) -> int:
        return len(self.xyxys) if self.xyxys is not None else 0

BOX_COLOR = (255, 77, 54)
ADDITIONAL_BOX_COLOR = (155, 155, 255)
TITLE_COLOR = (255, 255, 255)
global raw_results_to_save_buffer
global edited_results_to_save_buffer


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

def draw_boxes(frame, res: ReductedResults, color=BOX_COLOR):
    xyxy = res.xyxys.astype(int)
    ids  = res.ids
    confs = res.confs

    for (x1, y1, x2, y2), track_id, conf in zip(xyxy, ids, confs):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{track_id}" if track_id is not None else "X"
        cv2.putText(frame, label, ((x1 + x2)//2 - 5*len(label), (y1 + y2)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

###################################################
# Linear interpolation on nearest box method, not tested yet
# TO REMAKE
  
# def find_near_boxes(boxes: Boxes, target_box: list, max_shift: int):
#     target_x1, target_y1, target_x2, target_y2 = target_box
#     near_boxes = []
#     for box in boxes.xyxy.cpu().numpy().astype(int):
#         x1, y1, x2, y2 = box
#         if (abs(x1 - target_x1) <= max_shift and abs(y1 - target_y1) <= max_shift and
#             abs(x2 - target_x2) <= max_shift and abs(y2 - target_y2) <= max_shift):
#             near_boxes.append(box)
#     return near_boxes

# def find_nearest_box(boxes: Boxes, target_box: list, max_shift: int):
#     near_boxes = find_near_boxes(boxes, target_box, max_shift)
#     if not near_boxes:
#         return None
#     target_x1, target_y1, target_x2, target_y2 = target_box
#     nearest_box = min(near_boxes, key=lambda box: ((box[0] - target_x1) ** 2 + (box[1] - target_y1) ** 2 + 
#                                                      (box[2] - target_x2) ** 2 + (box[3] - target_y2) ** 2) ** 0.5)
#     return nearest_box

###################################################

###################################################
# Linear interpolation on missing id-box method

def lost_ids(results_list: list[ReductedResults], target_index: int) -> list[int]:
    target_ids = results_list[target_index].ids
    lost_ids: set[int] = set()
    for i in range(target_index-1, -1, -1):
        actual_ids = results_list[i].ids
        for actual_id in actual_ids:
            if actual_id not in target_ids:
                lost_ids.add(actual_id)
    return list(lost_ids)

def lost_bounding_box(results_list: list[ReductedResults], target_index: int) -> bool:
    n_boxes_on_target_frame = results_list[target_index].get_number_of_boxes()
    for i in range(target_index-1, -1, -1):
        if results_list[i].get_number_of_boxes() < n_boxes_on_target_frame:
            return True
    return False

def get_box_for_id(results_list: list[ReductedResults], index: int, track_id: int) -> np.ndarray | None:
    ids = results_list[index].ids
    xyxy = results_list[index].xyxys
    for i, tid in enumerate(ids):
        if tid == track_id:
            return xyxy[i]
    return None

def interpolate_missing_boxes(results_list: list[ReductedResults], target_index: int) -> ReductedResults:
    """For IDs missing at target_index but present before AND after,
    compute linearly interpolated bounding boxes.
    Returns an object of class ReductedResults."""
    target_results = results_list[target_index]
    target_ids = set(target_results.ids)

    # All IDs seen in frames before target
    before_ids = set(results_list[0].ids)
    missing_ids = before_ids - target_ids
    interpolated = {}

    for track_id in missing_ids:
        # Last frame before target where this ID was present
        last_before = None
        if track_id in before_ids:
            last_before = 0
        if last_before is None:
            continue

        # First frame after target where this ID reappears
        first_after = None
        for i in range(target_index + 1, len(results_list)):
            if track_id in results_list[i].ids:
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
        interpolated[track_id] = interp_box
    
    interpolated_results = ReductedResults(
        frame_index=target_results.frame_index,
        orig_img=target_results.orig_img,
        xyxys=np.array(list(interpolated.values())),
        confs=np.array([None] * len(interpolated)),
        ids=np.array(list(interpolated.keys()))
    )

    return interpolated_results

def draw_interpolated_boxes(frame, interpolated_results : ReductedResults, color=ADDITIONAL_BOX_COLOR):
    for box, track_id in zip(interpolated_results.xyxys.astype(int), interpolated_results.ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{track_id}*"
        cv2.putText(frame, label, ((x1 + x2)//2 - 5*len(label), (y1 + y2)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

###################################################

###################################################
# Save results to csv file

OUTPUT_HEADERS = ["frame_index", "track_id", "x1", "y1", "x2", "y2", "confidence", "interpolated"]

def result_to_csv_row(res: ReductedResults, interpolated: bool = False) -> list[dict]:
    rows: list[dict] = []
    for (box, conf, track_id) in zip(res.xyxys, res.confs, res.ids):
        x1, y1, x2, y2 = box
        row = {
            "frame_index": res.frame_index,
            "track_id": track_id if track_id is not None else "None",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": conf,
            "interpolated": interpolated
        }
        rows.append(row)
    return rows

def add_results_to_csv(results_list: list[ReductedResults], output_path: str, interpolated: bool = False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=OUTPUT_HEADERS)
        if csvfile.tell() == 0:  # Write header only if file is new
            writer.writeheader()
        for results in results_list:
            rows = result_to_csv_row(results, interpolated=interpolated)
            for row in rows:
                writer.writerow(row)

def flush_results_buffers():
    add_results_to_csv(raw_results_to_save_buffer, OUTPUT_FOLDER_PATH_RAW)
    add_results_to_csv(raw_results_to_save_buffer, OUTPUT_FOLDER_PATH_EDITED)
    add_results_to_csv(edited_results_to_save_buffer, OUTPUT_FOLDER_PATH_EDITED, interpolated=True)
    raw_results_to_save_buffer.clear()
    edited_results_to_save_buffer.clear()

###################################################

args = parse_args()

target_index = 1 # Second image of the buffer, as the first frame is the previous

file_path = args.input
file_name = file_path.split("/")[-1].split(".")[0] if "/" in file_path else file_path.split("\\")[-1].split(".")[0]
OUTPUT_FOLDER_PATH = f"data/output/{file_name}/"
OUTPUT_FOLDER_PATH_RAW = f"{OUTPUT_FOLDER_PATH}raw_tracking_results.csv"
OUTPUT_FOLDER_PATH_EDITED = f"{OUTPUT_FOLDER_PATH}tracking_results.csv"
os.remove(OUTPUT_FOLDER_PATH_RAW) if os.path.exists(OUTPUT_FOLDER_PATH_RAW) else None
os.remove(OUTPUT_FOLDER_PATH_EDITED) if os.path.exists(OUTPUT_FOLDER_PATH_EDITED) else None
model_source = args.model_path
model = YOLO(model_source)

try:
    model.overrides['classes'] = 0

    results_trough_time : list[ReductedResults] = []
    og_lost_total = 0
    interp_lost_total = 0
    prev_box_count = 0
    length_of_buffer = args.gap_frame + 2 # first frame is the previous, second frame is the target, and the rest are future frames for interpolation
    frame_index = 0
    raw_results_to_save_buffer: list[ReductedResults] = [] # List of ReductedResults to save to csv at the end of processing
    edited_results_to_save_buffer: list[ReductedResults] = [] # List of ReductedResults with interpolated boxes added, to save to csv at the end of processing

    results : Generator[Results] = model.track(source=file_path, save=False, stream=True, verbose=False, show_labels=False)

    # Processing first frame as we cannot add more info than YOLO has, as it is the first frame.
    first_res_raw = next(results)
    first_res = ReductedResults.from_boxes(frame_index, first_res_raw.orig_img, first_res_raw.boxes)
    if args.show or args.save:
        first_frame = first_res.orig_img.copy()
        if first_res.get_number_of_boxes() > 0:
            draw_boxes(first_frame, first_res)
        write_title(first_frame, title="Original box from the model")
        write_title(first_frame, title="Output")
        concat_frame = cv2.hconcat([first_frame, first_frame])
        if args.show:
            cv2.imshow("Tracking", concat_frame)
        if args.save:
            raw_results_to_save_buffer.append(first_res)
    frame_index += 1
    prev_box_count = first_res.get_number_of_boxes()
    results_trough_time.append(first_res)
    
    for r in results:
        # If the following condition is not true, it means that we don't have enough frames to start interpolation,
        # but that we still got the first frame to interpolate on, then we wait until we have enough frames to start
        # interpolation on it (i.e. collecting "future frames")
        if len(results_trough_time) >= length_of_buffer:
            res = results_trough_time[target_index]

            # Count lost bounding boxes (drop vs previous frame)
            current_box_count = res.get_number_of_boxes()
            og_drop = max(0, prev_box_count - current_box_count)
            og_lost_total += og_drop

            if args.show or args.save:
                og_frame_w_boxes = res.orig_img.copy()
                new_frame = res.orig_img.copy()

                if res.get_number_of_boxes() > 0:
                    draw_boxes(og_frame_w_boxes, res)
                    draw_boxes(new_frame, res)
                    res_w_interp = interpolate_missing_boxes(results_trough_time, target_index)
                    if res_w_interp.get_number_of_boxes() > 0:
                        draw_interpolated_boxes(new_frame, res_w_interp)
                    n_interp_boxes = res_w_interp.get_number_of_boxes()                    

                    interp_drop = max(0, og_drop - n_interp_boxes)
                    interp_lost_total += interp_drop

                    write_lost_counter(og_frame_w_boxes, og_lost_total)
                    write_lost_counter(new_frame, interp_lost_total)

                write_title(og_frame_w_boxes, title="Original box from the model")
                write_title(new_frame, title="Output")

                concat_frame = cv2.hconcat([og_frame_w_boxes, new_frame])
                if args.show:
                    cv2.imshow("Tracking", concat_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if args.save:
                    raw_results_to_save_buffer.append(res)
                    edited_results_to_save_buffer.append(res_w_interp)
                    if len(raw_results_to_save_buffer) >= 100 or len(edited_results_to_save_buffer) >= 100: # Save in batches of 100 frames to avoid memory issues
                        flush_results_buffers()
            
            frame_index += 1
            prev_box_count = current_box_count

        results_trough_time.append(ReductedResults.from_boxes(frame_index, r.orig_img, r.boxes))
        if len(results_trough_time) > length_of_buffer:
            results_trough_time.pop(0)

except KeyboardInterrupt:
    
    print("Process interrupted by user.")
finally:
    results.close()  # Ensure resources are released
    if args.save:
        flush_results_buffers()
    if args.show:
        time.sleep(1)
        cv2.destroyAllWindows()