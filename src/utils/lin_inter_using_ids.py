import numpy as np

from my_types import ReductedResults

###################################################
# Linear interpolation on missing id-box methods

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
        ids=np.array(list(interpolated.keys())),
        interpolated = True
    )

    return interpolated_results

###################################################