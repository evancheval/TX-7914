import cv2

from my_types import ReductedResults
from constants import BOX_COLOR, TITLE_COLOR, ADDITIONAL_BOX_COLOR

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
    for (x1, y1, x2, y2), track_id in zip(res.xyxys.astype(int), res.ids):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{track_id}" if track_id is not None else "X"
        cv2.putText(frame, label, ((x1 + x2)//2 - 5*len(label), (y1 + y2)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_interpolated_boxes(frame, interpolated_results : ReductedResults, color=ADDITIONAL_BOX_COLOR):
    draw_boxes(frame, interpolated_results, color=color)