import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from resistor_color_detector_v4 import analyze_resistor_image, result_for_cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def timestamp_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"


def capture_with_picam2(output_path: Path, width: int, height: int, warmup: float) -> None:
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError(
            "找不到 picamera2。請在 Raspberry Pi 上安裝："
            "sudo apt install -y python3-picamera2"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (width, height)})
    picam2.configure(config)
    picam2.start()
    try:
        time.sleep(warmup)
        picam2.capture_file(str(output_path))
    finally:
        picam2.stop()
        picam2.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Take a JPG with Raspberry Pi Camera Module v2 and run resistor color detection."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Captured JPG path. Default: captures/resistor_<timestamp>.jpg",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Annotated result image path. Default: results/annotated_<captured name>.jpg",
    )
    parser.add_argument("--width", type=int, default=1920, help="Capture width. Default: 1920")
    parser.add_argument("--height", type=int, default=1080, help="Capture height. Default: 1080")
    parser.add_argument("--warmup", type=float, default=1.0, help="Camera warmup seconds. Default: 1.0")
    parser.add_argument("--json", action="store_true", help="Print the detection result as JSON.")
    args = parser.parse_args()

    image_path = args.image or (PROJECT_ROOT / "captures" / timestamp_name("resistor", ".jpg"))
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    output_path = args.output or (PROJECT_ROOT / "results" / f"annotated_{image_path.name}")
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    capture_with_picam2(image_path, args.width, args.height, args.warmup)
    result = analyze_resistor_image(str(image_path), str(output_path), strict=False)
    compact = result_for_cli(result)
    compact["captured_image"] = str(image_path)

    if args.json:
        print(json.dumps(compact, ensure_ascii=False, indent=2))
    else:
        print(f"captured_image: {image_path}")
        print(f"ok: {compact['ok']}")
        print(f"message: {compact['message']}")
        print(f"band_colors: {compact['band_colors']}")
        print(f"resistance: {compact['resistance_text']}")
        print(f"tolerance: {compact['tolerance']}")
        print(f"saved_image: {compact['saved_image']}")

    return 0 if compact["ok"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
