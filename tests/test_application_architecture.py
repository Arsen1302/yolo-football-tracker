import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from football_analysis.config import resolve_model_path, resolve_output_path
from football_analysis.pipeline import _find_team_reference_frame


class PipelineConfigurationTests(unittest.TestCase):
    def test_explicit_model_path_takes_precedence_over_environment(self):
        explicit_path = Path("weights/explicit.pt")
        with patch.dict(os.environ, {"FOOTBALL_ANALYSIS_MODEL_PATH": "env.pt"}):
            self.assertEqual(
                resolve_model_path(explicit_path),
                explicit_path.resolve(),
            )

    def test_environment_model_path_is_used_when_no_path_is_passed(self):
        configured_path = Path("weights/from-env.pt")
        with patch.dict(
            os.environ,
            {"FOOTBALL_ANALYSIS_MODEL_PATH": str(configured_path)},
        ):
            self.assertEqual(
                resolve_model_path(),
                configured_path.resolve(),
            )

    def test_output_path_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "nested" / "result.mp4"
            self.assertEqual(resolve_output_path(output_path), output_path.resolve())
            self.assertTrue(output_path.parent.is_dir())


class PipelineSelectionTests(unittest.TestCase):
    def test_uses_first_frame_with_two_players(self):
        tracks = {
            "players": [
                {},
                {1: {"bbox": []}},
                {1: {"bbox": []}, 2: {"bbox": []}},
            ]
        }
        self.assertEqual(_find_team_reference_frame(tracks, frame_count=3), 2)

    def test_raises_when_no_frame_has_two_players(self):
        tracks = {"players": [{}, {1: {"bbox": []}}]}
        with self.assertRaisesRegex(
            ValueError,
            "Could not detect at least two players",
        ):
            _find_team_reference_frame(tracks, frame_count=2)


if __name__ == "__main__":
    unittest.main()
