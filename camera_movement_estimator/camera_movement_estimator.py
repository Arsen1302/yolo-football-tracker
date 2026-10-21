import cv2
import numpy as np


class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, width = first_frame_grayscale.shape

        mask_features = np.zeros_like(first_frame_grayscale)
        left_margin = int(width * 0.05)
        right_start = int(width * 0.85)
        right_end = int(width * 0.98)

        mask_features[:, 0:left_margin] = 1
        mask_features[:, right_start:right_end] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                if frame_num >= len(camera_movement_per_frame):
                    break
                for track_id, track_info in track.items():
                    if 'position' in track_info:
                        position = track_info['position']
                        camera_movement = camera_movement_per_frame[frame_num]
                        position_adjusted = (
                            position[0] - camera_movement[0],
                            position[1] - camera_movement[1]
                        )
                        tracks[object_name][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames):
        frames = iter(frames)
        first_frame = next(frames, None)
        if first_frame is None:
            return []

        camera_movement = [[0, 0]]
        old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame in frames:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if old_features is None or len(old_features) == 0:
                old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
                if old_features is None or len(old_features) == 0:
                    old_gray = frame_gray.copy()
                    camera_movement.append([0, 0])
                    continue

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            if new_features is not None and status is not None:
                good_new = new_features[status == 1]
                good_old = old_features[status == 1]

                if len(good_new) > 0:
                    # Հաշվում ենք բոլոր կետերի X և Y տեղաշարժերը
                    diffs = good_new - good_old  # Shape: (N, 2)
                    distances = np.linalg.norm(diffs, axis=1)

                    # Վերցնում ենք բոլոր կետերի տեղաշարժերի մեդիանը (որպեսզի 1 սխալ կետը ազդեցություն չունենա)
                    camera_movement_x = float(np.median(diffs[:, 0]))
                    camera_movement_y = float(np.median(diffs[:, 1]))
                    max_distance = float(np.max(distances))

            if max_distance > self.minimum_distance:
                camera_movement.append([camera_movement_x, camera_movement_y])
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            else:
                camera_movement.append([0, 0])

            old_gray = frame_gray.copy()

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        for frame_num, frame in enumerate(frames):
            if frame_num >= len(camera_movement_per_frame):
                break
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (350, 80), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            yield frame