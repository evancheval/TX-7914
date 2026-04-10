import os
import csv
import cv2

from my_types import ReductedResults
from constants import OUTPUT_HEADERS

###################################################
# Save results to csv file

def OUTPUT_FOLDER_PATH(file_name: str) -> str:
    return f"data/output/{file_name}/"

def OUTPUT_CSV_RAW_PATH(file_name: str) -> str:
    return f"{OUTPUT_FOLDER_PATH(file_name)}raw_tracking_results.csv"

def OUTPUT_CSV_EDITED_RAW(file_name: str) -> str:
    return f"{OUTPUT_FOLDER_PATH(file_name)}tracking_results.csv"

def OUTPUT_VIDEO_PATH(file_name: str) -> str:
    return f"{OUTPUT_FOLDER_PATH(file_name)}tracking_output.mp4"

def init_output_files(file_name: str):
    os.remove(OUTPUT_CSV_RAW_PATH(file_name)) if os.path.exists(OUTPUT_CSV_RAW_PATH(file_name)) else None
    os.remove(OUTPUT_CSV_EDITED_RAW(file_name)) if os.path.exists(OUTPUT_CSV_EDITED_RAW(file_name)) else None

def result_to_csv_row(res: ReductedResults) -> list[dict]:
    rows: list[dict] = []
    for (box, conf, track_id, interpolated) in zip(res.xyxys, res.confs, res.ids, res.interpolated):
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

def add_results_to_csv(results_list: list[ReductedResults], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=OUTPUT_HEADERS)
        if csvfile.tell() == 0:  # Write header only if file is new
            writer.writeheader()
        for results in results_list:
            rows = result_to_csv_row(results)
            for row in rows:
                writer.writerow(row)

def flush_results_buffers(raw_results_to_save_buffer: list[ReductedResults], edited_results_to_save_buffer: list[ReductedResults], file_name: str):
    add_results_to_csv(raw_results_to_save_buffer, OUTPUT_CSV_RAW_PATH(file_name))
    # Merging raw and edited results, sorting by frame index, and saving to the edited output file
    # in order to regenerate easily the video from output csv file if needed
    merged_results = sorted(raw_results_to_save_buffer + edited_results_to_save_buffer, key=lambda r: r.frame_index)
    add_results_to_csv(merged_results, OUTPUT_CSV_EDITED_RAW(file_name))
    raw_results_to_save_buffer.clear()
    edited_results_to_save_buffer.clear()

def init_output_video(input_video_name: str, cap = cv2.VideoCapture) -> cv2.VideoWriter:
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2 # We concatenate original and new frame side by side, so we need to double the width of the output video
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(OUTPUT_VIDEO_PATH(input_video_name), fourcc, fps, (w, h))

###################################################