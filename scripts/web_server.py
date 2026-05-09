from argparse import Namespace
from datetime import datetime
from pathlib import Path
import os
import threading
import time

from flask import Flask, jsonify, request, send_from_directory
import serial
from serial.tools import list_ports

from run_sort_cycle import DEFAULT_BUCKETS, ProtocolError, run_cycle_on_serial


app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"
ASSET_DIR = WEB_DIR / "assets"

# Arduino currently uses Serial.begin(9600) in firmware/Arduino_measure/src/main.cpp.
SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/ttyUSB0")
SERIAL_BAUD = int(os.environ.get("SERIAL_BAUD", "9600"))

AUTO_INTERVAL_SEC = float(os.environ.get("AUTO_INTERVAL_SEC", "3.0"))
MAX_ERROR = float(os.environ.get("MAX_ERROR", "0.20"))
CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "1920"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "1080"))
CAMERA_WARMUP = float(os.environ.get("CAMERA_WARMUP", "1.0"))

state_lock = threading.Lock()
worker_thread = None
stop_requested = False
auto_enabled = False
latest_value = "---"
latest_seq = 0
status_seq = 0

status = {
    "mode": "idle",
    "running": False,
    "auto_enabled": False,
    "serial_port": SERIAL_PORT,
    "serial_baud": SERIAL_BAUD,
    "message": "等待啟動",
    "error": None,
    "latest": None,
    "events": [],
}


def now_text():
    return datetime.now().strftime("%H:%M:%S")


def available_ports_text():
    return ", ".join(port.device for port in list_ports.comports()) or "none"


def update_status(**kwargs):
    global status_seq
    with state_lock:
        status.update(kwargs)
        status["auto_enabled"] = auto_enabled
        status_seq += 1
        status["seq"] = status_seq


def add_event(message):
    global status_seq
    line = f"[{now_text()}] {message}"
    with state_lock:
        status["events"].append(line)
        status["events"] = status["events"][-120:]
        status_seq += 1
        status["seq"] = status_seq


def mark_latest(result):
    global latest_value, latest_seq
    measured = result.get("measured_ohm")
    with state_lock:
        status["latest"] = result
        if measured is not None:
            latest_value = f"MEAS,{measured:.2f}"
            latest_seq += 1


def make_cycle_args():
    return Namespace(
        port=SERIAL_PORT,
        baud=SERIAL_BAUD,
        boot_delay=2.0,
        motion_timeout=20.0,
        measure_timeout=5.0,
        max_error=MAX_ERROR,
        image=None,
        output=None,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        warmup=CAMERA_WARMUP,
        bucket=DEFAULT_BUCKETS,
    )


def run_one_cycle_with_open_serial(ser):
    update_status(mode="running", running=True, message="執行分類循環", error=None)
    add_event("cycle started")
    result = run_cycle_on_serial(make_cycle_args(), ser, add_event)
    mark_latest(result)

    action = result.get("action")
    box = result.get("box")
    measured = result.get("measured_ohm")
    if action == "SORT":
        update_status(mode="idle", running=False, message=f"完成：分類到盒 {box}", error=None)
        add_event(f"cycle done: SORT,{box}, measured={measured:.2f} ohm")
    else:
        reason = result.get("reject_reason") or "unknown"
        update_status(mode="idle", running=False, message=f"完成：reject ({reason})", error=None)
        add_event(f"cycle done: REJECT ({reason})")


def cycle_worker(*, loop):
    global auto_enabled, stop_requested

    try:
        update_status(mode="connecting", running=True, message=f"開啟 {SERIAL_PORT}", error=None)
        with serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.2) as ser:
            time.sleep(2.0)
            ser.reset_input_buffer()

            while True:
                with state_lock:
                    should_stop = stop_requested or (loop and not auto_enabled)
                if should_stop:
                    break

                run_one_cycle_with_open_serial(ser)

                if not loop:
                    break

                update_status(mode="waiting", running=False, message=f"{AUTO_INTERVAL_SEC:.1f} 秒後執行下一輪")
                end_wait = time.monotonic() + AUTO_INTERVAL_SEC
                while time.monotonic() < end_wait:
                    with state_lock:
                        should_stop = stop_requested or not auto_enabled
                    if should_stop:
                        break
                    time.sleep(0.1)
    except (serial.SerialException, ProtocolError, RuntimeError) as exc:
        update_status(
            mode="error",
            running=False,
            message="發生錯誤",
            error=f"{exc}. Available ports: {available_ports_text()}",
        )
        add_event(f"error: {exc}")
    finally:
        with state_lock:
            status["running"] = False
            if not auto_enabled:
                status["mode"] = "idle"
            status["auto_enabled"] = auto_enabled


def start_worker(*, loop):
    global worker_thread, stop_requested, auto_enabled
    with state_lock:
        if worker_thread and worker_thread.is_alive():
            return False
        stop_requested = False
        auto_enabled = loop
        worker_thread = threading.Thread(target=cycle_worker, kwargs={"loop": loop}, daemon=True)
        worker_thread.start()
    return True


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index2.html")


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(status)


@app.route("/api/cycle", methods=["POST"])
def api_cycle():
    if not start_worker(loop=False):
        return jsonify({"ok": False, "error": "cycle already running"}), 409
    return jsonify({"ok": True})


@app.route("/api/auto/start", methods=["POST"])
def api_auto_start():
    global auto_enabled
    with state_lock:
        auto_enabled = True
    started = start_worker(loop=True)
    update_status(mode="waiting" if not started else "starting", message="自動模式已啟動", error=None)
    return jsonify({"ok": True, "started": started})


@app.route("/api/auto/stop", methods=["POST"])
def api_auto_stop():
    global auto_enabled, stop_requested
    with state_lock:
        auto_enabled = False
        stop_requested = True
    update_status(message="自動模式停止中")
    return jsonify({"ok": True})


@app.route("/get_resistance")
def get_resistance():
    with state_lock:
        return jsonify({"value": latest_value, "seq": latest_seq, "error": status.get("error")})


@app.route("/<path:filename>")
def web_file(filename):
    web_path = WEB_DIR / filename
    if web_path.exists():
        return send_from_directory(WEB_DIR, filename)
    return send_from_directory(ASSET_DIR, filename)


if __name__ == "__main__":
    if os.environ.get("AUTO_START", "0") == "1":
        start_worker(loop=True)
    port = int(os.environ.get("PORT", os.environ.get("WEB_PORT", "5000")))
    app.run(debug=False, host=os.environ.get("HOST", "0.0.0.0"), port=port, use_reloader=False)
