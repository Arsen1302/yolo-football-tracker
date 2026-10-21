import argparse

from football_analysis import process_video
from football_analysis.config import DEFAULT_VIDEO_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a football video.")
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO_PATH,
        help="Input video path (defaults to the sample video path).",
    )
    parser.add_argument("--output", help="Optional output MP4 path.")
    parser.add_argument("--model", help="Optional YOLO weights path.")
    args = parser.parse_args()

    result_path = process_video(args.video, args.output, args.model)
    print(f"Saved in {result_path}")


if __name__ == "__main__":
    main()
