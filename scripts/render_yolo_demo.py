"""Render the portfolio traffic demo with real YOLO11n + ByteTrack detections."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


VEHICLE_CLASSES = {2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}
CYAN = (214, 201, 89)  # BGR


def render(source: Path, output: Path, frames: int) -> None:
    model = YOLO("yolo11n.pt")
    capture = cv2.VideoCapture(str(source))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(source_fps / 30))
    width, height = 1280, 720
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), source_fps / step, (width, height)
    )

    source_index = 0
    written = 0
    while written < frames:
        ok, frame = capture.read()
        if not ok:
            break
        source_index += 1
        if source_index % step:
            continue

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLE_CLASSES),
            conf=0.35,
            imgsz=640,
            verbose=False,
        )[0]

        if result.boxes is not None:
            ids = result.boxes.id
            for index, (box, cls, confidence) in enumerate(
                zip(result.boxes.xyxy.cpu(), result.boxes.cls.cpu(), result.boxes.conf.cpu())
            ):
                x1, y1, x2, y2 = map(int, box.tolist())
                class_id = int(cls)
                track_id = int(ids[index]) if ids is not None else index + 1
                label = f"{VEHICLE_CLASSES[class_id]}  ID {track_id:02d}  {float(confidence):.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), CYAN, 2)
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1
                )
                label_top = max(0, y1 - text_height - 10)
                cv2.rectangle(frame, (x1, label_top), (x1 + text_width + 10, y1), CYAN, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (8, 15, 16),
                    1,
                    cv2.LINE_AA,
                )

        cv2.putText(frame, "YOLO11n  /  BYTETRACK", (22, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FRAME {written:04d}", (1135, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 200, 196), 1, cv2.LINE_AA)
        writer.write(frame)
        written += 1

    capture.release()
    writer.release()
    print(f"wrote {written} tracked frames to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=180)
    args = parser.parse_args()
    render(args.source, args.output, args.frames)
