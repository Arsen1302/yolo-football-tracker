from fractions import Fraction
from itertools import chain

import av
import cv2


def iter_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def get_video_frame(video_path, frame_number):
    if frame_number < 0:
        raise ValueError("frame_number must be non-negative")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise OSError(f"Could not open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
    finally:
        cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_number} from video: {video_path}")
    return frame

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise OSError(f"Could not open video: {video_path}")
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()

    if not frame_rate or frame_rate <= 0:
        raise ValueError(f"Video has an invalid frame rate: {video_path}")
    return float(frame_rate)

def save_video(output_video_frames, output_video_path, frame_rate=24):
    if frame_rate <= 0:
        raise ValueError("frame_rate must be greater than zero")
    frames = iter(output_video_frames)
    first_frame = next(frames, None)
    if first_frame is None:
        raise ValueError("Cannot save a video with no frames")
    frame_size = (first_frame.shape[1], first_frame.shape[0])
    width = frame_size[0] + frame_size[0] % 2
    height = frame_size[1] + frame_size[1] % 2
    output_rate = Fraction(str(frame_rate)).limit_denominator(1001)

    with av.open(output_video_path, mode="w") as container:
        stream = container.add_stream("libx264", rate=output_rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "23"}

        for frame in chain((first_frame,), frames):
            if (frame.shape[1], frame.shape[0]) != frame_size:
                raise ValueError("All output frames must have the same dimensions")
            if width != frame_size[0] or height != frame_size[1]:
                frame = cv2.copyMakeBorder(
                    frame,
                    0,
                    height - frame_size[1],
                    0,
                    width - frame_size[0],
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
