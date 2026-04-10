from ultralytics import YOLO #type:ignore
from typing import Generator
import argparse
import cv2
from ultralytics.engine.results import Results
import time
import csv

from my_types import ReductedResults
from utils.showing import write_title, write_lost_counter, draw_boxes, draw_interpolated_boxes
from utils.lin_inter_using_ids import interpolate_missing_boxes
from utils.saving import init_output_files, flush_results_buffers

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
        "--from-results-file",
        help="Path to a csv file containing precomputed results (got from the --save flag)",
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
    # parser.add_argument(
    #     "--max-box-shift",
    #     type=int,
    #     default=10,
    #     help = "Maximal number of pixel shift to consider a box the same between two frames (used for box interpolation in case of missing boxes). The default value is 10, based on the original work of this script using 360x640 frames."
    # )
    return parser.parse_args()

global raw_results_to_save_buffer
global edited_results_to_save_buffer

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


args = parse_args()

if args.save and args.from_results_file:
    raise ValueError("Cannot use --save option when --from-boxes-file is provided, as the boxes are precomputed and saved in the provided file, so there is no need to save them again.")

target_index = 1 # Second image of the buffer, as the first frame is the previous

file_path = args.input
file_name = file_path.split("/")[-1].split(".")[0] if "/" in file_path else file_path.split("\\")[-1].split(".")[0]
if args.save:
    init_output_files(file_name)

try:
    results_trough_time : list[ReductedResults] = []
    og_lost_total = 0
    interp_lost_total = 0
    prev_box_count = 0
    length_of_buffer = args.gap_frame + 2 # first frame is the previous, second frame is the target, and the rest are future frames for interpolation
    frame_index = 0
    raw_results_to_save_buffer: list[ReductedResults] = [] # List of ReductedResults to save to csv at the end of processing
    edited_results_to_save_buffer: list[ReductedResults] = [] # List of ReductedResults with interpolated boxes added, to save to csv at the end of processing

    if args.from_results_file:
        results_file = open(args.from_results_file, "r")
        csv_reader = csv.DictReader(results_file)
        results: Generator[ReductedResults] = ReductedResults.reducted_results_from_dict_reader(csv_reader, capture=cv2.VideoCapture(args.input))

    else:            
        model_source = args.model_path
        model = YOLO(model_source)
        model.overrides['classes'] = 0
        results: Generator[ReductedResults] = ReductedResults.from_results_generator(model.track(source=file_path, save=False, stream=True, verbose=False))

    # Processing first frame as we cannot add more info than YOLO has, as it is the first frame.
    first_res = next(results)
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
        if len(results_trough_time) >= length_of_buffer or (args.from_results_file):
            res = results_trough_time[target_index] if not args.from_results_file else r

            # Count lost bounding boxes (drop vs previous frame)
            if args.from_results_file:
                current_box_count = res.interpolated.count(False)
            else:
                current_box_count = res.get_number_of_boxes()
            og_drop = max(0, prev_box_count - current_box_count)
            og_lost_total += og_drop

            if args.show or args.save:
                og_frame_w_boxes = res.orig_img.copy()
                new_frame = res.orig_img.copy()

                if res.get_number_of_boxes() > 0:
                    # If the results are from a file, it means that they already have the interpolated boxes, so we don't need to interpolate them again
                    if args.from_results_file:
                        res_raw = res.get_only_interpolated_results(invert=True)
                        draw_boxes(og_frame_w_boxes, res_raw)
                        draw_boxes(new_frame, res_raw)
                        res_w_interp = res.get_only_interpolated_results()
                    else:
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
                        flush_results_buffers(raw_results_to_save_buffer, edited_results_to_save_buffer, file_name)
            
            frame_index += 1
            prev_box_count = current_box_count

        results_trough_time.append(r)
        if len(results_trough_time) > length_of_buffer:
            results_trough_time.pop(0)

except KeyboardInterrupt:
    print("Process interrupted by user.")
finally:
    results.close()  # Ensure resources are released
    results_file.close() if args.from_results_file else None
    if args.save:
        flush_results_buffers(raw_results_to_save_buffer, edited_results_to_save_buffer, file_name)
    if args.show:
        time.sleep(1)
        cv2.destroyAllWindows()