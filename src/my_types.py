from ultralytics.engine.results import Boxes
import numpy as np

class ReductedResults():
    def __init__(self, frame_index: int, orig_img: np.ndarray, xyxys: np.ndarray, confs: np.ndarray | None, ids: np.ndarray | None, interpolated: bool = False):
        self.frame_index = frame_index
        self.orig_img = orig_img
        self.xyxys = xyxys
        self.confs = confs if confs is not None else [None] * len(xyxys)
        self.ids : list[int|None] = list(ids.astype(int)) if ids is not None else [None] * len(xyxys)
        self.interpolated = interpolated
    
    def from_boxes(frame_index: int, orig_img: np.ndarray, boxes: Boxes, interpolated: bool = False) -> ReductedResults:
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
