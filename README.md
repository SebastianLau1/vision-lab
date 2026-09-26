# Vision Lab

YOLO detection in your browser, with webcam and image upload support.

[Live app](https://sebastianlau.is-a.dev/projects/vision-lab/index.html) · [Portfolio](https://sebastianlau.is-a.dev)

## Run

```sh
npm run dev
# Open http://localhost:8080
npm test
```

## What it does

Live webcam and uploaded-image detection run YOLOv8n with ONNX Runtime Web 1.22.0. The model uses 640 × 640 letterboxed RGB tensors, per-class confidence scores, and class-aware non-maximum suppression. The first run downloads ~13 MB of weights plus the runtime. Images and camera frames never leave the browser. Webcam access requires HTTPS or localhost and explicit permission; stopping, navigating away, or hiding the page releases the camera.

The recorded example uses YOLO11n + ByteTrack. It is **pre-rendered output**, separate from the live YOLOv8n detector. Its 29.97 FPS label describes video playback, not measured inference performance. The live interface reports actual per-call inference latency. The live detector does not assign persistent tracking IDs.

Model: [webml/yolov8n](https://huggingface.co/webml/yolov8n), pinned revision `85bc8d7`. Runtime: [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html). YOLO weights are subject to their upstream license, including Ultralytics licensing terms; this repository does not relicense them. Video: [João Pavese / Pexels](https://www.pexels.com/video/traffic-on-highway-in-city-12240861/). The original Python rendering script is included under `scripts/`.

Limitations: CPU/WASM speed varies by device; detections can be wrong; 80 COCO classes only; this is a demonstration, not a safety-critical system. External model/CDN availability is needed for the first inference run.

## Deployment

The app is independently deployable as a Cloudflare Worker/Pages static asset app using `wrangler.jsonc`. Its public demo is also deployed with the portfolio under `/projects/vision-lab/`. Static app files are committed into the portfolio; to refresh them, run `python3 scripts/sync_projects.py` from the portfolio checkout with sibling project checkouts present.

No credentials are stored in source.
