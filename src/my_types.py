import cv2
from ultralytics.engine.results import Boxes, Results
import numpy as np
from csv import DictReader
from typing import Generator

class ReductedResults():
    def __init__(self, frame_index: int, orig_img: np.ndarray, xyxys: np.ndarray, confs: np.ndarray | None, ids: np.ndarray | None, interpolated: list[bool]):
        self.frame_index = frame_index
        self.orig_img = orig_img
        self.xyxys = xyxys
        self.confs = confs if confs is not None else [None] * len(xyxys)
        self.ids : list[int|None] = list(ids.astype(int)) if ids is not None else [None] * len(xyxys)
        self.interpolated = interpolated
    
    def from_boxes(frame_index: int, orig_img: np.ndarray, boxes: Boxes, interpolated: list[bool]) -> ReductedResults:
        return ReductedResults(
            frame_index,
            orig_img,
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy() if boxes.conf is not None else None,
            boxes.id.cpu().numpy() if boxes.id is not None else None,
            interpolated
        )
    
    def get_number_of_boxes(self) -> int:
        return len(self.xyxys) if self.xyxys is not None else 0
    
    def from_results_generator(results_gen: Generator[Results]) -> Generator[ReductedResults]:
        for i, res in enumerate(results_gen):
            yield ReductedResults.from_boxes(i, res.orig_img, res.boxes, interpolated=[False] * len(res.boxes))

    def reducted_results_from_dict_reader(dict_reader: DictReader, capture: cv2.VideoCapture) -> Generator[ReductedResults]:
        results_on_the_same_frame = []
        current_frame_index = None
        while dict_reader:
            try:
                row = next(dict_reader)
                current_frame_index = int(row["frame_index"])
                while row and int(row["frame_index"]) == current_frame_index:
                    results_on_the_same_frame.append(row)
                    row = next(dict_reader)
            except StopIteration:
                row = None
            finally:
                yield ReductedResults(
                    frame_index=current_frame_index,
                    orig_img=np.asarray(capture.read()[1][:,:]),
                    xyxys=np.array([[int(float(row["x1"])), int(float(row["y1"])), int(float(row["x2"])), int(float(row["y2"]))] for row in results_on_the_same_frame if row["x1"] and row["y1"] and row["x2"] and row["y2"]]),
                    confs=np.array([float(row["confidence"]) if row["confidence"] else None for row in results_on_the_same_frame]) if any(row["confidence"] for row in results_on_the_same_frame) else None,
                    ids=np.array([int(row["track_id"]) for row in results_on_the_same_frame if row["track_id"]]) if any(row["track_id"] for row in results_on_the_same_frame) else None,
                    interpolated=[True if row.get("interpolated", "false").lower() == "true" else False for row in results_on_the_same_frame]
                )
                results_on_the_same_frame = [row] if row else []
            if row is None:
                break
    
    def get_only_interpolated_results(self, invert: bool = False) -> ReductedResults:
        interpolated_indices = [i for i, interp in enumerate(self.interpolated) if interp != invert]
        return ReductedResults(
            frame_index=self.frame_index,
            orig_img=self.orig_img,
            xyxys=self.xyxys[interpolated_indices],
            confs=self.confs[interpolated_indices] if self.confs is not None else None,
            ids=np.array(self.ids)[interpolated_indices].astype(int) if self.ids is not None else None,
            interpolated=[True] * len(interpolated_indices)
        )