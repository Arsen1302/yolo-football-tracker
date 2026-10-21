import cv2
import numpy as np

from utils import get_foot_position


class SpeedAndDistance_Estimator:
    def __init__(self, frame_rate=30.0, smoothing_window=5, max_speed_kmh=45.0):
        if frame_rate <= 0:
            raise ValueError("frame_rate must be greater than zero")
        if smoothing_window < 3 or smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be an odd number of at least 3")

        self.frame_rate = float(frame_rate)
        self.smoothing_window = smoothing_window
        self.max_speed_mps = max_speed_kmh / 3.6
        self.max_gap_frames = 3

    def _smooth_positions(self, positions):
        smoothed = positions.copy()
        radius = self.smoothing_window // 2
        weights = np.arange(1, radius + 2, dtype=float)
        weights = np.concatenate((weights, weights[-2::-1]))

        for index in range(radius, len(positions) - radius):
            window = positions[index - radius:index + radius + 1]
            smoothed[index] = np.average(window, axis=0, weights=weights)
        return smoothed

    def _reject_position_outliers(self, positions):
        cleaned = positions.copy()
        if len(positions) < 5:
            return cleaned

        radius = self.smoothing_window // 2
        for index in range(radius, len(positions) - radius):
            window = positions[index - radius:index + radius + 1]
            local_median = np.median(window, axis=0)
            residuals = np.linalg.norm(window - local_median, axis=1)
            median_residual = np.median(residuals)
            mad = np.median(np.abs(residuals - median_residual))
            threshold = max(0.75, median_residual + 4.0 * 1.4826 * mad)
            if np.linalg.norm(positions[index] - local_median) > threshold:
                cleaned[index] = local_median
        return cleaned

    def _estimate_track(self, object_tracks, track_id):
        observations = []
        for frame_number, frame_tracks in enumerate(object_tracks):
            track = frame_tracks.get(track_id)
            if track is None:
                continue

            position = track.get("position_transformed")
            if position is None:
                continue
            position = np.asarray(position, dtype=float).reshape(-1)
            if position.size != 2 or not np.all(np.isfinite(position)):
                continue
            observations.append((frame_number, track, position))

        if not observations:
            return

        frame_numbers = np.asarray([item[0] for item in observations], dtype=float)
        positions = np.asarray([item[2] for item in observations], dtype=float)
        positions = self._reject_position_outliers(positions)
        smoothed_positions = self._smooth_positions(positions)
        speeds = np.zeros(len(observations), dtype=float)
        distances = np.zeros(len(observations), dtype=float)
        radius = self.smoothing_window // 2

        cumulative_distance = 0.0
        for index in range(len(observations)):
            if index > 0:
                frame_gap = frame_numbers[index] - frame_numbers[index - 1]
                elapsed = frame_gap / self.frame_rate
                if 0 < frame_gap <= self.max_gap_frames:
                    step_distance = float(np.linalg.norm(
                        smoothed_positions[index] - smoothed_positions[index - 1]
                    ))
                    step_speed = step_distance / elapsed
                    if step_speed <= self.max_speed_mps:
                        cumulative_distance += step_distance

            distances[index] = cumulative_distance
            left = max(0, index - radius)
            right = min(len(observations), index + radius + 1)
            local_frames = frame_numbers[left:right]
            local_positions = smoothed_positions[left:right]

            if len(local_frames) >= 2:
                times = (local_frames - frame_numbers[index]) / self.frame_rate
                centered_times = times - np.mean(times)
                denominator = float(np.dot(centered_times, centered_times))
                if denominator > 0:
                    velocity = np.sum(
                        centered_times[:, None] * local_positions,
                        axis=0,
                    ) / denominator
                    speed = float(np.linalg.norm(velocity))
                    if speed <= self.max_speed_mps:
                        speeds[index] = speed * 3.6
                    elif index > 0:
                        speeds[index] = speeds[index - 1]

        for index, (_, track, _) in enumerate(observations):
            track["speed"] = float(speeds[index])
            track["distance"] = float(distances[index])

    def add_speed_and_distance_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            if object_name in {"ball", "referees"}:
                continue

            track_ids = {
                track_id
                for frame_tracks in object_tracks
                for track_id in frame_tracks
            }
            for track_id in track_ids:
                self._estimate_track(object_tracks, track_id)
        return tracks

    def draw_speed_and_distance(self, frames, tracks):
        for frame_number, frame in enumerate(frames):
            if frame_number >= len(tracks["players"]):
                break
            for object_name, object_tracks in tracks.items():
                if object_name in {"ball", "referees"} or frame_number >= len(object_tracks):
                    continue

                for track_info in object_tracks[frame_number].values():
                    speed = track_info.get("speed")
                    distance = track_info.get("distance")
                    if speed is None or distance is None:
                        continue

                    position = get_foot_position(track_info["bbox"])
                    position = (int(position[0]), int(position[1] + 40))
                    cv2.putText(
                        frame,
                        f"{speed:.1f} km/h",
                        position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{distance:.1f} m",
                        (position[0], position[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )
            yield frame
