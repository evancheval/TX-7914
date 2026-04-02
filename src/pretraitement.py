import cv2 #type :ignore
import numpy as np
import sys
import os


def build_background_model(cap, n_frames=200, quantile_margin=0.1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Cannot read the first frame of the video.")

    h, w = first.shape[:2]
    buf = np.empty((n_frames, h, w, 3), dtype=np.uint8)
    buf[0] = first

    collected = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_window = min(total_frames, int(cap.get(cv2.CAP_PROP_FPS) * 10))
    step = max(1, sample_window // n_frames)

    for i in range(1, n_frames):
        target = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            break
        buf[collected] = frame
        collected += 1

    buf = buf[:collected].astype(np.float32)

    bg_low = np.percentile(buf, 2, axis=0).astype(np.float32)
    bg_high = np.percentile(buf, 98, axis=0).astype(np.float32)

    margin = quantile_margin * 255
    bg_low = np.clip(bg_low - margin, 0, 255)
    bg_high = np.clip(bg_high + margin, 0, 255)

    return bg_low, bg_high


def process_video(input_path, output_path, n_model_frames=200, quantile_margin=0.1):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.1f} fps, {total} frames")
    print("Building background model ...")
    bg_low, bg_high = build_background_model(cap, n_model_frames, quantile_margin)
    print("Background model ready.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ff = frame.astype(np.float32)

        in_range = (ff >= bg_low) & (ff <= bg_high)
        is_background = np.all(in_range, axis=2)

        frame[is_background] = 255

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = frame_idx / total * 100
            print(f"  Processed {frame_idx}/{total} frames ({pct:.1f}%)")

    cap.release()
    out.release()
    print(f"Done. Output saved to {output_path}")


def process_video_mog2(input_path, output_path, history=500, var_threshold=16, detect_shadows=True):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.1f} fps, {total} frames")
    print(f"MOG2 params: history={history}, varThreshold={var_threshold}, detectShadows={detect_shadows}")

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = fgbg.apply(frame)
        # mask values: 255 = foreground, 127 = shadow (if detectShadows), 0 = background
        fg_mask = (mask == 255)

        result = np.full_like(frame, 255)
        result[fg_mask] = frame[fg_mask]

        out.write(result)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = frame_idx / total * 100
            print(f"  Processed {frame_idx}/{total} frames ({pct:.1f}%)")

    cap.release()
    out.release()
    print(f"Done. Output saved to {output_path}")


if __name__ == "__main__":
    input_video = "../data/10min.mp4"

    # --- Percentile-based method ---
    # output_video = "../data/10min_01_motion.mp4"
    # N_MODEL_FRAMES = 200
    # QUANTILE_MARGIN = 0.1
    # process_video(input_video, output_video, N_MODEL_FRAMES, QUANTILE_MARGIN)

    # --- MOG2 method ---
    output_video = "../data/10min_500_6_mog2_motion.mp4"
    process_video_mog2(input_video, output_video, history=500, var_threshold=8, detect_shadows=True)
