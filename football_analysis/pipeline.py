from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path
from typing_extensions import TypeAlias,Any
import numpy as np

from camera_movement_estimator import CameraMovementEstimator
from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from trackers import Tracker
from utils import get_video_fps, get_video_frame, iter_video_frames, save_video
from view_transformer import ViewTransformer

from .config import resolve_model_path, resolve_output_path

TrackData: TypeAlias = dict[str, list[dict[int, dict[str, Any]]]]


def _find_team_reference_frame(
    tracks: TrackData,
    frame_count: int,
) -> int:
    reference_frame = next(
        (
            frame_number
            for frame_number in range(frame_count)
            if len(tracks["players"][frame_number]) >= 2
        ),
        None,
    )
    if reference_frame is None:
        raise ValueError("Could not detect at least two players to identify the teams.")
    return reference_frame


def _track_video(
    video_path: Path,
    first_frame: np.ndarray,
    video_fps: float,
    tracker: Tracker,
) -> tuple[
    TrackData,
    list[list[float]],
    int,
    CameraMovementEstimator,
    SpeedAndDistance_Estimator,
]:
    tracks = tracker.get_object_tracks(iter_video_frames(str(video_path)))
    tracker.add_position_to_tracks(tracks)

    camera_estimator = CameraMovementEstimator(first_frame)
    camera_movement = camera_estimator.get_camera_movement(
        iter_video_frames(str(video_path))
    )
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

    ViewTransformer().add_transformed_position_to_tracks(tracks)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_estimator = SpeedAndDistance_Estimator(frame_rate=video_fps)
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    frame_count = min(len(tracks["players"]), len(camera_movement))
    return tracks, camera_movement, frame_count, camera_estimator, speed_estimator


def _assign_teams_and_possession(
    video_path: Path,
    tracks: TrackData,
    frame_count: int,
) -> np.ndarray:
    reference_frame = _find_team_reference_frame(tracks, frame_count)
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(
        get_video_frame(str(video_path), reference_frame),
        tracks["players"][reference_frame],
    )

    for frame_number, frame in enumerate(iter_video_frames(str(video_path))):
        if frame_number >= frame_count:
            break
        for player_id, player_track in tracks["players"][frame_number].items():
            team = team_assigner.get_player_team(
                frame,
                player_track["bbox"],
                player_id,
            )
            player_track["team"] = team
            player_track["team_color"] = team_assigner.team_colors[team]

    player_assigner = PlayerBallAssigner()
    team_ball_control: list[int] = []
    for frame_number in range(frame_count):
        player_tracks = tracks["players"][frame_number]
        ball_tracks = tracks["ball"][frame_number]
        assigned_player = -1
        if 1 in ball_tracks:
            assigned_player = player_assigner.assign_ball_to_player(
                player_tracks,
                ball_tracks[1]["bbox"],
            )

        if assigned_player != -1:
            player_tracks[assigned_player]["has_ball"] = True
            team_ball_control.append(player_tracks[assigned_player]["team"])
        elif team_ball_control:
            team_ball_control.append(team_ball_control[-1])
        else:
            team_ball_control.append(0)

    return np.asarray(team_ball_control)


def _render_video(
    video_path: Path,
    output_path: Path,
    video_fps: float,
    tracks: TrackData,
    camera_movement: list[list[float]],
    frame_count: int,
    team_ball_control: np.ndarray,
    tracker: Tracker,
    camera_estimator: CameraMovementEstimator,
    speed_estimator: SpeedAndDistance_Estimator,
) -> None:
    annotated_frames: Iterable[np.ndarray] = tracker.draw_annotations(
        iter_video_frames(str(video_path)),
        tracks,
        team_ball_control,
    )
    annotated_frames = camera_estimator.draw_camera_movement(
        annotated_frames,
        camera_movement[:frame_count],
    )
    annotated_frames = speed_estimator.draw_speed_and_distance(
        annotated_frames,
        tracks,
    )
    save_video(annotated_frames, str(output_path), frame_rate=video_fps)


def process_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> str:
    video_path = Path(video_path).expanduser().resolve()
    model_path = resolve_model_path(model_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    resolved_output_path = resolve_output_path(output_path)
    video_fps = get_video_fps(str(video_path))
    first_frame = get_video_frame(str(video_path), 0)
    tracker = Tracker(str(model_path))

    (
        tracks,
        camera_movement,
        frame_count,
        camera_estimator,
        speed_estimator,
    ) = _track_video(
        video_path,
        first_frame,
        video_fps,
        tracker,
    )
    team_ball_control = _assign_teams_and_possession(
        video_path,
        tracks,
        frame_count,
    )
    _render_video(
        video_path,
        resolved_output_path,
        video_fps,
        tracks,
        camera_movement,
        frame_count,
        team_ball_control,
        tracker,
        camera_estimator,
        speed_estimator,
    )
    return str(resolved_output_path)
