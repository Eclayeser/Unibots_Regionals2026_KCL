"""
debugView.py
────────────
Streams annotated camera frames as MJPEG over HTTP.
No display or GUI required — view in any browser on the same network.

Endpoints
---------
    /              Live annotated feed
    /stream        Raw MJPEG stream
    /masks         Diagnostic 2x2 tile: original | ball mask | obstacle mask | blank
    /masks/stream  Raw MJPEG for masks tile

Usage
-----
    python debugView.py
    python debugView.py --port 8080
    python debugView.py --camera 1 --quality 60

Open on your laptop (find Pi IP with: hostname -I):
    http://raspberrypi.local:5000
    http://192.168.x.x:5000
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional

import cv2
import numpy as np

from ballDetector import MovementVector, PingPongDetector


# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────

class _FrameBuffer:
    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._frame: Optional[bytes] = None
        self._event = threading.Event()

    def put(self, jpeg: bytes) -> None:
        with self._lock:
            self._frame = jpeg
        self._event.set()
        self._event.clear()

    def get_latest(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def wait_for_next(self, timeout: float = 2.0) -> Optional[bytes]:
        self._event.wait(timeout)
        return self.get_latest()


_main_buf  = _FrameBuffer()
_masks_buf = _FrameBuffer()


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

_C_BALL    = (0,   255,   0)
_C_VECTOR  = (0,   200, 255)
_C_CLOSEST = (0,   100, 255)
_C_LABEL   = (255, 255, 255)
_C_ROBOT   = (255,  80,  80)
_FONT      = cv2.FONT_HERSHEY_SIMPLEX


def draw_debug_frame(
    frame: np.ndarray,
    vectors: List[MovementVector],
    robot_px: tuple,
    arrow_scale: float = 120.0,
) -> np.ndarray:
    out = frame.copy()
    rx, ry = robot_px

    cv2.drawMarker(out, (rx, ry), _C_ROBOT,
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.putText(out, "robot", (rx + 8, ry - 8),
                _FONT, 0.45, _C_ROBOT, 1, cv2.LINE_AA)

    for idx, v in enumerate(vectors):
        bx, by     = v.target_px
        is_closest = (idx == 0)
        colour     = _C_CLOSEST if is_closest else _C_BALL

        cv2.circle(out, (bx, by), 12, colour, 2)
        cv2.circle(out, (bx, by),  3, colour, -1)

        label = ("[!] " if is_closest else "") + f"{v.distance_cm:.1f}cm"
        cv2.putText(out, label, (bx + 14, by - 6),
                    _FONT, 0.45, colour, 1, cv2.LINE_AA)

        angle_rad = math.radians(v.angle)
        dx =  math.sin(angle_rad) * v.magnitude * arrow_scale
        dy = -math.cos(angle_rad) * v.magnitude * arrow_scale

        tip = (int(rx + dx), int(ry + dy))
        cv2.arrowedLine(out, (rx, ry), tip, _C_VECTOR,
                        2, tipLength=0.2, line_type=cv2.LINE_AA)
        cv2.putText(out, f"{v.angle:+.1f}  mag={v.magnitude:.2f}",
                    (tip[0] + 5, tip[1]), _FONT, 0.4, _C_VECTOR, 1, cv2.LINE_AA)

    n = len(vectors)
    summary = f"Balls: {n}" if n else "No balls detected"
    if n:
        c = vectors[0]
        summary += f"  |  Closest: {c.distance_cm:.1f}cm  ang={c.angle:+.1f}"
    cv2.putText(out, summary, (8, 22), _FONT, 0.55, _C_LABEL, 1, cv2.LINE_AA)
    return out


def build_masks_tile(
    frame: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    2x2 diagnostic tile:
        [ Original frame       |  Ball mask (green)      ]
        [ Obstacle mask (red)  |  Overlay (both on frame) ]

    Tune ball_hsv in ballConfig.py until the ball mask (green) covers
    the balls cleanly.  If obstacles appear on the white floor, lower
    min_saturation in ballConfig.py (but that should not be needed).
    """
    h, w = frame.shape[:2]

    def colour_mask(mask: np.ndarray, bgr: tuple) -> np.ndarray:
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[mask > 0] = bgr
        return out

    ball_vis = colour_mask(masks["ball"],     (0, 220,   0))
    obs_vis  = colour_mask(masks["obstacle"], (0,   0, 220))

    # Overlay: draw both masks coloured on the original frame
    overlay = frame.copy()
    overlay[masks["ball"]     > 0] = (0, 200, 0)
    overlay[masks["obstacle"] > 0] = (0, 0,   200)

    def labelled(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.putText(out, text, (8, 22), _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    top    = np.hstack([labelled(frame,    "Original"),
                        labelled(ball_vis, "Ball mask — tune ball_hsv")])
    bottom = np.hstack([labelled(obs_vis,  "Obstacle mask — tune min_saturation"),
                        labelled(overlay,  "Overlay")])
    return np.vstack([top, bottom])


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────

_NAV = ('<a href="/" style="color:#7af;margin:0 .8rem">Live feed</a>'
        '<a href="/masks" style="color:#7af;margin:0 .8rem">Mask debug</a>')

def _html_page(title: str, stream_path: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"><title>{title}</title>
  <style>
    body{{margin:0;background:#111;display:flex;flex-direction:column;
         align-items:center;justify-content:center;min-height:100vh;font-family:monospace}}
    h1{{color:#eee;font-size:1rem;margin:.5rem 0}}
    nav{{margin-bottom:.4rem}}</style>
</head>
<body>
  <h1>{title}</h1>
  <nav>{_NAV}</nav>
  <img src="{stream_path}" style="max-width:100%;border:2px solid #333">
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ─────────────────────────────────────────────────────────────────────────────

class _StreamHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if   self.path == "/":             self._page("PingPong — live feed",  "/stream")
        elif self.path == "/stream":       self._mjpeg(_main_buf)
        elif self.path == "/masks":        self._page("PingPong — mask debug", "/masks/stream")
        elif self.path == "/masks/stream": self._mjpeg(_masks_buf)
        else:                              self.send_error(404)

    def _page(self, title: str, stream: str) -> None:
        body = _html_page(title, stream).encode()
        self.send_response(200)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mjpeg(self, buf: _FrameBuffer) -> None:
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=--jpgboundary")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection",    "close")
        self.end_headers()
        try:
            while True:
                jpeg = buf.wait_for_next(timeout=2.0)
                if jpeg is None:
                    continue
                header = (
                    b"--jpgboundary\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                )
                self.wfile.write(header + jpeg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Capture + detection loop
# ─────────────────────────────────────────────────────────────────────────────

def _capture_loop(det: PingPongDetector, camera_index: int,
                  jpeg_quality: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  det._frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, det._frame_h)
    print(f"[debugView] Camera {camera_index}  "
          f"{int(cap.get(3))}x{int(cap.get(4))}")

    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            vectors   = det.analyze_frame(frame)
            annotated = draw_debug_frame(frame, vectors, det._robot_px)
            ok2, jpg  = cv2.imencode(".jpg", annotated, params)
            if ok2:
                _main_buf.put(jpg.tobytes())

            masks     = det.get_debug_masks(frame)
            tile      = build_masks_tile(frame, masks)
            ok3, jpg2 = cv2.imencode(".jpg", tile, params)
            if ok3:
                _masks_buf.put(jpg2.tobytes())

            if vectors:
                v = vectors[0]
                print(f"\r  balls={len(vectors)}"
                      f"  closest={v.distance_cm:.1f}cm"
                      f"  angle={v.angle:+.1f}"
                      f"  mag={v.magnitude:.2f}   ",
                      end="", flush=True)
            else:
                print("\r  No balls detected            ", end="", flush=True)

    finally:
        cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream annotated frames as MJPEG over HTTP."
    )
    p.add_argument("--config-module", default="ballConfig")
    p.add_argument("--camera",  type=int, default=0)
    p.add_argument("--port",    type=int, default=5000)
    p.add_argument("--host",    default="0.0.0.0")
    p.add_argument("--quality", type=int, default=80)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    det  = PingPongDetector(args.config_module)

    threading.Thread(
        target=_capture_loop,
        args=(det, args.camera, args.quality),
        daemon=True,
    ).start()

    server = HTTPServer((args.host, args.port), _StreamHandler)
    print(f"[debugView] Live feed  ->  http://<pi-ip>:{args.port}/")
    print(f"[debugView] Mask debug ->  http://<pi-ip>:{args.port}/masks")
    print( "[debugView] Find Pi IP:    hostname -I")
    print( "[debugView] Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("\n[debugView] Stopped.")


if __name__ == "__main__":
    main()