import os

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.image_size = int(os.environ.get("FOOTBALL_ANALYSIS_IMAGE_SIZE", "640"))
        if self.image_size <= 0:
            raise ValueError("FOOTBALL_ANALYSIS_IMAGE_SIZE must be greater than zero")
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30
        )

    def add_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_name == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_name][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_best_ball_detection(
        self,
        detection_supervision,
        ball_class_id,
        previous_ball_bbox=None,
        frame_shape=None,
    ):
        ball_indices = np.where(detection_supervision.class_id == ball_class_id)[0]
        if len(ball_indices) == 0:
            return None

        boxes = detection_supervision.xyxy[ball_indices].astype(float)
        widths = np.maximum(0, boxes[:, 2] - boxes[:, 0])
        heights = np.maximum(0, boxes[:, 3] - boxes[:, 1])
        areas = widths * heights
        aspect_ratios = widths / np.maximum(heights, 1)

        if frame_shape is not None:
            frame_area = frame_shape[0] * frame_shape[1]
        else:
            max_x = max(float(np.max(boxes[:, 2])), 1.0)
            max_y = max(float(np.max(boxes[:, 3])), 1.0)
            frame_area = max_x * max_y

        plausible = (
            (areas >= max(4.0, frame_area * 0.000005))
            & (areas <= frame_area * 0.005)
            & (aspect_ratios >= 0.35)
            & (aspect_ratios <= 2.8)
        )
        candidate_positions = np.flatnonzero(plausible)
        if len(candidate_positions) == 0:
            candidate_positions = np.arange(len(boxes))

        confidences = detection_supervision.confidence
        if confidences is None:
            confidences = np.ones(len(detection_supervision), dtype=float)
        else:
            confidences = np.asarray(confidences, dtype=float)[ball_indices]
            confidences = np.nan_to_num(confidences, nan=0.0)

        if previous_ball_bbox is not None:
            previous_center = np.array([
                (previous_ball_bbox[0] + previous_ball_bbox[2]) / 2,
                (previous_ball_bbox[1] + previous_ball_bbox[3]) / 2,
            ])
            centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2,
                                       (boxes[:, 1] + boxes[:, 3]) / 2))
            frame_diagonal = max(float(np.hypot(*frame_shape[:2])), 1.0) if frame_shape is not None else 1.0
            motion_penalty = np.minimum(
                np.linalg.norm(centers - previous_center, axis=1) / (frame_diagonal * 0.15),
                1.0,
            )
            previous_area = max(
                (previous_ball_bbox[2] - previous_ball_bbox[0])
                * (previous_ball_bbox[3] - previous_ball_bbox[1]),
                1.0,
            )
            size_penalty = np.minimum(
                np.abs(np.log(np.maximum(areas, 1.0) / previous_area)) / np.log(4.0),
                1.0,
            )
            scores = 0.6 * motion_penalty + 0.25 * size_penalty + 0.15 * (1.0 - confidences)
        else:
            expected_area = max(frame_area * 0.00015, 4.0)
            size_penalty = np.minimum(
                np.abs(np.log(np.maximum(areas, 1.0) / expected_area)) / np.log(4.0),
                1.0,
            )
            scores = 0.65 * size_penalty + 0.35 * (1.0 - confidences)

        best_position = candidate_positions[np.argmin(scores[candidate_positions])]
        return boxes[best_position].tolist()


    def get_object_tracks(self, frames):
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        previous_ball_bbox = None
        batch_size = 1
        batch_frames = []

        def process_batch():
            nonlocal previous_ball_bbox
            detections = self.model.predict(
                batch_frames,
                conf=0.1,
                imgsz=self.image_size,
            )
            if len(detections) != len(batch_frames):
                raise RuntimeError(
                    f"YOLO returned {len(detections)} detections for "
                    f"{len(batch_frames)} input frames."
                )
            for frame, detection in zip(batch_frames, detections):
                frame_num = len(tracks["players"])
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}
                detection_supervision = sv.Detections.from_ultralytics(detection)

                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_supervision.class_id[object_ind] = cls_names_inv["player"]

                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})

                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    if cls_id == cls_names_inv['player']:
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}

                    if cls_id == cls_names_inv['referee']:
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                ball_cls_id = cls_names_inv.get('ball')
                if ball_cls_id is not None:
                    best_ball_bbox = self.get_best_ball_detection(
                        detection_supervision,
                        ball_cls_id,
                        previous_ball_bbox=previous_ball_bbox,
                        frame_shape=frame.shape,
                    )
                    if best_ball_bbox is not None:
                        tracks["ball"][frame_num][1] = {"bbox": best_ball_bbox}
                        previous_ball_bbox = best_ball_bbox

        for frame in frames:
            batch_frames.append(frame)
            if len(batch_frames) == batch_size:
                process_batch()
                batch_frames.clear()

        if batch_frames:
            process_batch()

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, team_1, team_2):
        _, width, _ = frame.shape

        rect_width, rect_height = 520, 120
        margin = 20

        x1 = width - rect_width - margin
        y1 = margin
        x2 = width - margin
        y2 = margin + rect_height

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        text_x = x1 + 20
        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", (text_x, y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", (text_x, y1 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        team_ball_control = np.asarray(team_ball_control)
        team_1_control = np.cumsum(team_ball_control == 1)
        team_2_control = np.cumsum(team_ball_control == 2)

        for frame_num, frame in enumerate(video_frames):
            if frame_num >= len(tracks["players"]):
                break
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            total_control = team_1_control[frame_num] + team_2_control[frame_num]
            if total_control > 0:
                team_1 = team_1_control[frame_num] / total_control
                team_2 = team_2_control[frame_num] / total_control
            else:
                team_1, team_2 = 0.5, 0.5
            frame = self.draw_team_ball_control(frame, team_1, team_2)
            yield frame