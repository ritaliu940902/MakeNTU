from pathlib import Path
import os

from flask import Flask, jsonify, send_from_directory
import serial
from serial.tools import list_ports


app = Flask(__name__)
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
ASSET_DIR = WEB_DIR / "assets"

# Arduino currently uses Serial.begin(9600) in firmware/Arduino_measure/src/main.cpp.
SERIAL_PORT = os.environ.get("SERIAL_PORT", "COM5")
SERIAL_BAUD = int(os.environ.get("SERIAL_BAUD", "9600"))

ser = None
latest_value = "---"
latest_seq = 0
serial_error = None


def open_serial():
    global ser, serial_error

    if ser and ser.is_open:
        return ser

    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        serial_error = None
        return ser
    except serial.SerialException as exc:
        available_ports = ", ".join(port.device for port in list_ports.comports()) or "none"
        serial_error = (
            f"Could not open {SERIAL_PORT}. "
            f"Available ports: {available_ports}. "
            f"Original error: {exc}"
        )
        ser = None
        return None


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index2.html")


@app.route("/get_resistance")
def get_resistance():
    global latest_value, latest_seq

    serial_conn = open_serial()
    if serial_conn is None:
        return jsonify({"value": latest_value, "seq": latest_seq, "error": serial_error})

    if serial_conn.in_waiting > 0:
        while serial_conn.in_waiting > 0:
            data = serial_conn.readline().decode("utf-8", errors="ignore").strip()
            if data:
                latest_value = data
                latest_seq += 1

    return jsonify({"value": latest_value, "seq": latest_seq, "error": None})


@app.route("/<path:filename>")
def web_file(filename):
    web_path = WEB_DIR / filename
    if web_path.exists():
        return send_from_directory(WEB_DIR, filename)
    return send_from_directory(ASSET_DIR, filename)


if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)
