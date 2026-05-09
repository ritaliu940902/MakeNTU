import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import serial

from picam_capture_detect import (
    PROJECT_ROOT,
    analyze_resistor_image,
    capture_with_picam2,
    timestamp_name,
)


# Edit these ranges to match your five physical boxes.
# Each item is (min_ohm inclusive, max_ohm exclusive, box_number).
DEFAULT_BUCKETS = [
    (0.0, 100.0, 1),
    (100.0, 1_000.0, 2),
    (1_000.0, 10_000.0, 3),
    (10_000.0, 100_000.0, 4),
    (100_000.0, math.inf, 5),
]


class ProtocolError(RuntimeError):
    pass


LogFn = Callable[[str], None]


def parse_bucket(spec: str) -> Tuple[float, float, int]:
    try:
        min_text, max_text, box_text = spec.split(":")
        min_ohm = float(min_text)
        max_ohm = math.inf if max_text.lower() == "inf" else float(max_text)
        box = int(box_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "bucket must use min:max:box, for example 1000:10000:3"
        ) from exc

    if min_ohm < 0 or max_ohm <= min_ohm or not 1 <= box <= 5:
        raise argparse.ArgumentTypeError("bucket values must satisfy 0 <= min < max and 1 <= box <= 5")
    return min_ohm, max_ohm, box


def resolve_project_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_line_until(
    ser: serial.Serial,
    predicate: Callable[[str], bool],
    timeout_s: float,
    label: str,
    log: LogFn = print,
) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue

        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        log(f"< {line}")
        if predicate(line):
            return line

    raise ProtocolError(f"timeout waiting for {label}")


def send_line(ser: serial.Serial, line: str, log: LogFn = print) -> None:
    log(f"> {line}")
    ser.write(f"{line}\n".encode("utf-8"))
    ser.flush()


def wait_for_event(ser: serial.Serial, expected: Iterable[str], timeout_s: float, log: LogFn = print) -> str:
    prefixes = tuple(expected)
    return read_line_until(
        ser,
        lambda line: line.startswith(prefixes),
        timeout_s,
        " / ".join(prefixes),
        log,
    )


def capture_and_detect(args: argparse.Namespace) -> dict:
    image_path = resolve_project_path(args.image)
    if image_path is None:
        image_path = PROJECT_ROOT / "captures" / timestamp_name("resistor", ".jpg")
        capture_with_picam2(image_path, args.width, args.height, args.warmup)

    output_path = resolve_project_path(args.output)
    if output_path is None:
        output_path = PROJECT_ROOT / "results" / f"annotated_{image_path.name}"

    result = analyze_resistor_image(str(image_path), str(output_path), strict=False)
    result["captured_image"] = str(image_path)
    result["saved_image"] = result.get("result_debug_image")
    return result


def parse_measured_ohm(line: str) -> float:
    if not line.startswith("MEAS,"):
        raise ProtocolError(f"expected MEAS,<ohm>, got {line}")
    try:
        return float(line.split(",", 1)[1])
    except ValueError as exc:
        raise ProtocolError(f"bad measurement line: {line}") from exc


def choose_bucket(ohm: float, buckets: List[Tuple[float, float, int]]) -> Optional[int]:
    for min_ohm, max_ohm, box in buckets:
        if min_ohm <= ohm < max_ohm:
            return box
    return None


def relative_error(measured_ohm: float, vision_ohm: float) -> float:
    if vision_ohm <= 0:
        return math.inf
    return abs(measured_ohm - vision_ohm) / vision_ohm


def run_cycle_on_serial(args: argparse.Namespace, ser: serial.Serial, log: LogFn = print) -> dict:
    buckets = args.bucket or DEFAULT_BUCKETS

    send_line(ser, "START", log)
    start_line = wait_for_event(ser, ("CAMERA_READY", "ERR,"), args.motion_timeout, log)
    if start_line.startswith("ERR,"):
        raise ProtocolError(f"Arduino refused START: {start_line}")

    try:
        vision_result = capture_and_detect(args)
    except Exception as exc:
        vision_result = {
            "ok": False,
            "message": f"vision exception: {exc}",
            "resistance_ohm": None,
        }
    log(json.dumps(vision_result, ensure_ascii=False))

    vision_ok = bool(vision_result.get("ok"))
    vision_ohm = vision_result.get("resistance_ohm")
    if vision_ohm is not None:
        vision_ohm = float(vision_ohm)

    send_line(ser, "MEASURE", log)
    measure_line = wait_for_event(ser, ("MEAS,", "ERR,"), args.measure_timeout, log)

    if measure_line.startswith("ERR,"):
        send_line(ser, "REJECT", log)
        reject_done_line = wait_for_event(ser, ("DONE", "ERR,"), args.motion_timeout, log)
        if reject_done_line.startswith("ERR,"):
            raise ProtocolError(f"Arduino failed to reject after measurement error: {reject_done_line}")
        log(f"Rejected because Arduino returned {measure_line}")
        return {
            "ok": False,
            "action": "REJECT",
            "reject_reason": measure_line,
            "vision_result": vision_result,
            "vision_ohm": vision_ohm,
            "measured_ohm": None,
            "box": None,
        }

    measured_ohm = parse_measured_ohm(measure_line)

    box = None
    reject_reason = None
    error_ratio = None
    if not vision_ok or vision_ohm is None:
        reject_reason = "vision failed"
    else:
        error_ratio = relative_error(measured_ohm, vision_ohm)
        log(f"vision_ohm={vision_ohm:.2f}, measured_ohm={measured_ohm:.2f}, error={error_ratio:.3f}")
        if error_ratio > args.max_error:
            reject_reason = f"mismatch over {args.max_error:.0%}"
        else:
            box = choose_bucket(measured_ohm, buckets)
            if box is None:
                reject_reason = "measurement outside bucket ranges"

    if reject_reason:
        send_line(ser, "REJECT", log)
        log(f"decision=REJECT ({reject_reason})")
    else:
        send_line(ser, f"SORT,{box}", log)
        log(f"decision=SORT,{box}")

    done_line = wait_for_event(ser, ("DONE", "ERR,"), args.motion_timeout, log)
    if done_line.startswith("ERR,"):
        raise ProtocolError(f"Arduino failed to finish sorting: {done_line}")

    return {
        "ok": box is not None,
        "action": "SORT" if box is not None else "REJECT",
        "reject_reason": reject_reason,
        "vision_result": vision_result,
        "vision_ohm": vision_ohm,
        "measured_ohm": measured_ohm,
        "error_ratio": error_ratio,
        "box": box,
    }


def run_cycle(args: argparse.Namespace) -> int:
    with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
        time.sleep(args.boot_delay)
        ser.reset_input_buffer()
        result = run_cycle_on_serial(args, ser)
        return 0 if result["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one resistor capture/measure/sort cycle.")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Arduino serial port.")
    parser.add_argument("--baud", type=int, default=9600, help="Arduino serial baud rate.")
    parser.add_argument("--boot-delay", type=float, default=2.0, help="Seconds to wait after opening serial.")
    parser.add_argument("--motion-timeout", type=float, default=20.0, help="Seconds to wait for motion events.")
    parser.add_argument("--measure-timeout", type=float, default=5.0, help="Seconds to wait for MEAS event.")
    parser.add_argument("--max-error", type=float, default=0.20, help="Max allowed vision-vs-measure relative error.")

    parser.add_argument("--image", type=Path, default=None, help="Use an existing image instead of taking a new photo.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Annotated result image path.")
    parser.add_argument("--width", type=int, default=1920, help="Capture width.")
    parser.add_argument("--height", type=int, default=1080, help="Capture height.")
    parser.add_argument("--warmup", type=float, default=1.0, help="Camera warmup seconds.")

    parser.add_argument(
        "--bucket",
        type=parse_bucket,
        action="append",
        help="Override bucket ranges. Format min:max:box. Repeat for multiple buckets.",
    )

    args = parser.parse_args()
    return run_cycle(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProtocolError, serial.SerialException, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
