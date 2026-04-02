from ultralytics import YOLO
import argparse
import cv2
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
ADDITIONAL_BOX_COLOR = (255, 0, 0)
TITLE_COLOR = (255, 255, 255)

def write_title(frame, title="", color=TITLE_COLOR):
    frame_h, frame_w = frame.shape[:2]
    cv2.putText(frame, title, (int(0.02 * frame_h), int(0.08 * frame_h)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

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
# First method, not tested yet
  
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

def lost_id(results_list: list[Results], target_index: int) -> bool:
    target_ids = results_list[target_index].boxes.id.cpu().numpy().astype(int)
    for i in range(target_index-1, -1, -1):
        actual_ids = results_list[i].boxes.id.cpu().numpy().astype(int)
        if not any(id in target_ids for id in actual_ids):
            return True
    return False

###################################################


args = parse_args()

median_index = args.gap_frame // 2

file_path = args.input
model_source = args.model_path

model = YOLO(model_source)

try:
    model.overrides['classes'] = 0

    results_trough_time = []

    i = 0
    
    results = model.track(source=file_path, save=args.save, stream=True, verbose=False, show_labels=False)
    for r in results:
        # If the following condition is not true, it means that we don't have enough frames to start interpolation,
        # but that we still got the first frame to interpolate on, then we wait until we have enough frames to start
        # interpolation on it (collecting "future frames")
        if not(len(results_trough_time) > median_index & len(results_trough_time) < args.gap_frame):
            interpolating: bool = len(results_trough_time) >= args.gap_frame
            if interpolating:
                res = results_trough_time[median_index]
            else:
                # Not enough frames for interpolation, use the current result
                res = r
            if args.show:                
                boxes = res.boxes
                og_frame_w_boxes = res.orig_img.copy()
                new_frame = res.orig_img.copy()


                if boxes is not None and boxes.xyxy is not None:
                    draw_boxes(og_frame_w_boxes, boxes)
                    write_title(og_frame_w_boxes, title="Original box from the model")
                    draw_boxes(new_frame, boxes)
                    # if interpolating and lost_id(results_trough_time, median_index):
                    #     write_title(new_frame, title="LOST ID")
                    # else:
                    #     write_title(new_frame, title="Output")

                concat_frame = cv2.hconcat([og_frame_w_boxes, new_frame])
                cv2.imshow("Tracking", concat_frame)
                # time.sleep(0.5)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        print(f"Frame {i} processed. len(results_trough_time)={len(results_trough_time)}")
        i += 1
        results_trough_time.append(r)
        if len(results_trough_time) > args.gap_frame:
            results_trough_time.pop(0)

    if args.show:
        time.sleep(1)
        cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("Process interrupted by user.")
    if args.show:
        cv2.destroyAllWindows()