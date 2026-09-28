"""Render Vision Lab's recorded demo with real YOLO11n + ByteTrack person tracking.

Usage:
    python scripts/render_yolo_demo.py SOURCE_VIDEO web/crosswalk-yolo-demo.mp4 \
        --start 38 --seconds 8 --poster web/crosswalk-yolo-poster.jpg --sample web/crosswalk-sample.jpg

Requires: ultralytics (YOLO11n weights download on first run), opencv-python, imageio-ffmpeg.
Output is H.264 (yuv420p, faststart) so every browser can play it.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
from ultralytics import YOLO

PERSON = 0
YELLOW = (10, 214, 255)  # BGR for Vision Lab's person color, #ffd60a
INK = (10, 10, 10)
WIDTH, HEIGHT = 1280, 720


def draw_track(frame, box, track_id: int, confidence: float) -> None:
    x1, y1, x2, y2 = map(int, box)
    label = f"PERSON {track_id:02d}  {confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 2)
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    top = max(0, y1 - text_height - 10)
    cv2.rectangle(frame, (x1 - 1, top), (x1 + text_width + 10, y1), YELLOW, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK, 1, cv2.LINE_AA)


def render(source: Path, output: Path, start: float, seconds: float, poster: Path | None, sample: Path | None) -> None:
    model = YOLO("yolo11n.pt")
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = round(seconds * fps)

    encoder = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", f"{fps:.6f}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )

    written = 0
    tracked_ids: set[int] = set()
    while written < frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        if sample and written == frames // 2:
            cv2.imwrite(str(sample), frame, [cv2.IMWRITE_JPEG_QUALITY, 86])  # clean frame for live detection

        result = model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON], conf=0.35, imgsz=640, verbose=False
        )[0]
        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id, confidence in zip(
                result.boxes.xyxy.cpu().tolist(), result.boxes.id.int().cpu().tolist(), result.boxes.conf.cpu().tolist()
            ):
                draw_track(frame, box, track_id, confidence)
                tracked_ids.add(track_id)

        if poster and written == 0:
            cv2.imwrite(str(poster), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        encoder.stdin.write(frame.tobytes())
        written += 1

    capture.release()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("ffmpeg failed to encode the clip")
    print(f"wrote {written} frames at {fps:.3f} fps to {output}; {len(tracked_ids)} unique person IDs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0, help="seconds into the source to begin")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--poster", type=Path, help="write the first tracked frame as a JPEG")
    parser.add_argument("--sample", type=Path, help="write a clean, untracked mid-clip frame as a JPEG")
    args = parser.parse_args()
    render(args.source, args.output, args.start, args.seconds, args.poster, args.sample)
