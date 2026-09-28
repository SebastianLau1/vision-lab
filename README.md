# Vision Lab

YOLO object detection in your browser, on your webcam or any image, styled as a camera viewfinder.

[Live app](https://sebastianlau1.github.io/vision-lab/) · [Portfolio](https://sebastianlau.is-a.dev)

## Run

```sh
npm run dev
# Open http://localhost:8080
npm test
```

## What it does

Pick a source (recorded demo, webcam, image upload, or a sample frame), or drop or paste an image anywhere on the page. Boxes are color-coded by class group (people, vehicles, animals, other); the confidence slider and class chips re-filter without re-running the model, and detections export as JSON. Webcam boxes are drawn on the exact frame that was inferred, so they stay aligned with moving objects.

Live detection runs YOLOv8n with ONNX Runtime Web 1.22.0. The model uses 640 × 640 letterboxed RGB tensors, per-class confidence scores, and class-aware non-maximum suppression. The first run downloads ~13 MB of weights plus the runtime. Images and camera frames never leave the browser. Webcam access requires HTTPS or localhost and explicit permission; stopping, navigating away, or hiding the page releases the camera.

The recorded example uses YOLO11n + ByteTrack. It is **pre-rendered output**, separate from the live YOLOv8n detector. Its 29.97 FPS label describes video playback, not measured inference performance. The live interface reports actual per-call inference latency. The live detector does not assign persistent tracking IDs.

Model: [webml/yolov8n](https://huggingface.co/webml/yolov8n), pinned revision `85bc8d7`. Runtime: [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html). YOLO weights are subject to their upstream license, including Ultralytics licensing terms; this repository does not relicense them. Video: [João Pavese / Pexels](https://www.pexels.com/video/traffic-on-highway-in-city-12240861/). The original Python rendering script is included under `scripts/`.

Limitations: CPU/WASM speed varies by device; detections can be wrong; 80 COCO classes only; this is a demonstration, not a safety-critical system. External model/CDN availability is needed for the first inference run.

## Deployment

Every push to `main` runs the tests and publishes `web/` to GitHub Pages (`.github/workflows/deploy.yml`). The app can also be deployed as a Cloudflare Workers static-asset app with `wrangler.jsonc`.

No credentials are stored in source.
