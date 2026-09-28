import { classGroup, decode } from "./detector.js";

const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
// ONNX export of Ultralytics YOLO11n (same weights as the recorded clip), pinned to a revision.
const MODEL_URL = "https://huggingface.co/webnn/yolo11n/resolve/9c5acfd/onnx/yolo11n.onnx";
const INPUT = 640;
const DECODE_FLOOR = 0.1; // decode generously once; the slider filters without re-running the model
const FEED_LIMIT = 40;
const RECORDED_FPS = 23.976;
const COLORS = { person: "#ffd60a", vehicle: "#ff6a3d", animal: "#c38bff", other: "#8cf56b" };

const $ = (id) => document.getElementById(id);
const canvas = $("display");
const ctx = canvas.getContext("2d");
const prep = document.createElement("canvas"); // 640x640 letterboxed model input
const prepCtx = prep.getContext("2d", { willReadFrequently: true });
const frame = document.createElement("canvas"); // webcam frame that was actually inferred
const frameCtx = frame.getContext("2d");
prep.width = prep.height = INPUT;

let session = null;
let loading = null;
let stream = null;
let version = 0; // bumped on every source change so stale async work can bail out
let mode = "recorded";
let starting = false;
let image = null;
let boxes = [];
let latency = 0;
let fps = 0;
let filter = "all";

const pad = (n) => String(n).padStart(2, "0");

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function setText(values) {
  for (const [id, text] of Object.entries(values)) $(id).textContent = text;
}

function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

function setRec(state, label) {
  $("rec").className = state ? `rec is-${state}` : "rec";
  $("hud-mode").textContent = label;
}

function feedMessage(text) {
  $("detections").replaceChildren(el("li", text, "feed-empty"));
  $("summary").replaceChildren();
}

function download(text, name, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = el("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ---------- Model ---------- */

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = resolve;
    script.onerror = () => reject(Error("Could not load the inference runtime. Check your connection."));
    document.head.append(script);
  });
}

async function loadModel() {
  if (session) return session;
  loading ??= (async () => {
    $("loader").hidden = false;
    status("Loading ONNX Runtime and YOLO11n weights (~11 MB)…");
    if (!window.ort) await loadScript(`${ORT_BASE}ort.min.js`);
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return session;
  })()
    .catch((error) => {
      loading = null;
      throw error;
    })
    .finally(() => {
      $("loader").hidden = true;
    });
  return loading;
}

async function infer(source, ticket) {
  const w = canvas.width;
  const h = canvas.height;
  const scale = Math.min(INPUT / w, INPUT / h);
  const dx = (INPUT - w * scale) / 2;
  const dy = (INPUT - h * scale) / 2;
  prepCtx.fillStyle = "rgb(114,114,114)";
  prepCtx.fillRect(0, 0, INPUT, INPUT);
  prepCtx.drawImage(source, dx, dy, w * scale, h * scale);

  const pixels = prepCtx.getImageData(0, 0, INPUT, INPUT).data;
  const area = INPUT * INPUT;
  const data = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    data[i] = pixels[4 * i] / 255;
    data[area + i] = pixels[4 * i + 1] / 255;
    data[2 * area + i] = pixels[4 * i + 2] / 255;
  }

  const input = new ort.Tensor("float32", data, [1, 3, INPUT, INPUT]);
  const start = performance.now();
  let output;
  try {
    output = await session.run({ [session.inputNames[0]]: input });
    if (ticket !== version) return;
    latency = performance.now() - start;
    const tensor = output[session.outputNames[0]];
    boxes = decode(tensor.data, tensor.dims, DECODE_FLOOR, w, h);
    draw(source);
  } finally {
    input.dispose();
    if (output) Object.values(output).forEach((tensor) => tensor.dispose());
  }
}

/* ---------- Rendering ---------- */

function visible() {
  const threshold = Number($("confidence").value) / 100;
  return boxes.filter((box) => box.score >= threshold && (filter === "all" || classGroup(box.classId) === filter));
}

function draw(source) {
  if (!source) return;
  // The webcam is shown mirrored, like a mirror; boxes are flipped to match. Exports stay unmirrored.
  const mirror = mode === "camera";
  if (mirror) {
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  if (mirror) ctx.restore();

  const found = visible();
  const line = Math.max(2, canvas.width / 480);
  const size = Math.max(12, Math.round(canvas.width / 64));
  ctx.lineWidth = line;
  ctx.font = `600 ${size}px "JetBrains Mono", ui-monospace, monospace`;
  ctx.textBaseline = "middle";

  for (const box of found) {
    const color = COLORS[classGroup(box.classId)];
    const x = mirror ? canvas.width - box.x - box.w : box.x;
    ctx.fillStyle = `${color}1c`;
    ctx.fillRect(x, box.y, box.w, box.h);
    ctx.strokeStyle = color;
    ctx.strokeRect(x, box.y, box.w, box.h);

    const text = `${box.label} ${Math.round(box.score * 100)}%`;
    const width = ctx.measureText(text).width + size;
    const height = size + 10;
    const left = Math.min(x - line / 2, canvas.width - width);
    const top = box.y >= height ? box.y - height : box.y;
    ctx.fillStyle = color;
    ctx.fillRect(left, top, width, height);
    ctx.fillStyle = "#0a0a0a";
    ctx.fillText(text, left + size / 2, top + height / 2 + 1);
  }

  renderFeed(found);
  renderReadouts(found.length);
  $("export").disabled = false;
}

function renderFeed(found) {
  const list = $("detections");
  list.classList.toggle("is-live", mode === "camera");
  if (!found.length) {
    feedMessage("No detections match the current confidence and class filter.");
    return;
  }

  list.replaceChildren(...found.slice(0, FEED_LIMIT).map((box) => {
    const group = classGroup(box.classId);
    const row = el("li");
    row.style.setProperty("--c", COLORS[group]);
    const bar = el("span", undefined, "bar");
    const fill = el("i");
    fill.style.width = `${box.score * 100}%`;
    bar.append(fill);
    row.append(el("i", undefined, `swatch ${group}`), el("span", box.label), bar, el("b", `${(box.score * 100).toFixed(1)}%`));
    return row;
  }));
  if (found.length > FEED_LIMIT) list.append(el("li", `+ ${found.length - FEED_LIMIT} more in the JSON export`, "feed-more"));

  const counts = new Map();
  found.forEach((box) => counts.set(box.label, (counts.get(box.label) || 0) + 1));
  $("summary").replaceChildren(...[...counts]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([label, count]) => {
      const chip = el("span");
      chip.append(el("b", String(count)), ` ${label}`);
      return chip;
    }));
}

function renderReadouts(count) {
  setText({
    count: pad(count),
    speed: String(Math.round(latency)),
    "speed-note": mode === "camera" && fps ? `Per model call · about ${fps.toFixed(1)} fps overall` : "Per model call on this device",
    "hud-left": `DET ${pad(count)} · ${Math.round(latency)} MS`,
  });
}

/* ---------- Sources ---------- */

function updateToggle() {
  const toggle = $("toggle");
  toggle.hidden = mode === "image";
  if (mode === "recorded") {
    toggle.disabled = false;
    toggle.textContent = $("recording").paused ? "Play clip" : "Pause clip";
  } else if (mode === "camera") {
    toggle.disabled = starting;
    toggle.textContent = stream || starting ? "Stop webcam" : "Restart webcam";
  }
}

function stop() {
  const wasStreaming = Boolean(stream);
  version++;
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  $("webcam").srcObject = null;
  if (mode === "camera") {
    setRec("", "STOPPED");
    updateToggle();
    if (wasStreaming) status("Webcam stopped and released. The last frame stays on screen.");
  }
}

function setMode(next, sourceId) {
  stop();
  mode = next;
  boxes = [];
  image = null;
  fps = 0;
  const recorded = next === "recorded";
  $("stage").classList.toggle("is-camera", next === "camera");
  $("recording").hidden = !recorded;
  canvas.hidden = recorded;
  $("export").disabled = true;
  document.querySelectorAll(".source").forEach((button) => {
    const active = button.id === sourceId;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });

  if (recorded) {
    $("recording").play().catch(() => {});
    setRec("", "PLAYBACK");
    setText({
      mode: "RECORDED",
      "hud-res": "1280 × 720",
      "hud-right": "YOLO11n + BYTETRACK",
      "count-label": "Tracked class",
      count: "Person",
      "count-note": "IDs persist across frames",
      "speed-label": "Playback rate",
      speed: RECORDED_FPS.toFixed(2),
      "speed-unit": "fps",
      "speed-note": "Video frame rate, not inference speed",
      "model-name": "YOLO11n",
      "model-note": "+ ByteTrack, pre-rendered",
      execution: "Playback",
    });
    feedMessage("Boxes and IDs in the recording are baked into the video. Start the webcam or analyze an image to inspect live confidence scores here.");
    status("Recorded example. Pick a live source to load YOLO11n (~11 MB).");
  } else {
    $("recording").pause();
    canvas.width = 1280; // resizing also clears the previous frame
    canvas.height = 720;
    setRec(next === "camera" ? "" : "still", next === "camera" ? "STARTING" : "STILL");
    setText({
      mode: next === "camera" ? "LIVE" : "STILL",
      "hud-right": next === "camera" ? "YOLO11n · MIRRORED" : "YOLO11n · WASM",
      "hud-left": "DET 00",
      "count-label": "Visible detections",
      count: "00",
      "count-note": "After confidence and class filter",
      "speed-label": "Inference latency",
      speed: "--",
      "speed-unit": "ms",
      "speed-note": "Per model call on this device",
      "model-name": "YOLO11n",
      "model-note": "ONNX · 80 COCO classes",
      execution: "CPU · WASM",
    });
    feedMessage(next === "camera" ? "Waiting for the camera…" : "Reading image…");
  }
  updateToggle();
}

function cameraError(error) {
  if (error.name === "NotAllowedError") return "Camera permission was denied. Try an image instead.";
  if (error.name === "NotFoundError") return "No camera found. Try an image instead.";
  return error.message || "Could not start the webcam.";
}

async function startCamera() {
  setMode("camera", "camera");
  const ticket = version;
  starting = true;
  $("camera").disabled = true;
  updateToggle();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw Error("Webcam needs HTTPS and a browser with camera support.");
    await loadModel();
    if (ticket !== version) return;
    status("Waiting for camera permission…");
    const media = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    if (ticket !== version) {
      media.getTracks().forEach((track) => track.stop());
      return;
    }
    stream = media;
    const video = $("webcam");
    video.srcObject = stream;
    await video.play();
    canvas.width = frame.width = video.videoWidth;
    canvas.height = frame.height = video.videoHeight;
    $("hud-res").textContent = `${canvas.width} × ${canvas.height}`;
    setRec("live", "LIVE");
    status("Webcam live. Every frame is analyzed on this device.");
    starting = false;
    $("camera").disabled = false;
    updateToggle();

    let last = performance.now();
    while (stream && ticket === version) {
      frameCtx.drawImage(video, 0, 0, frame.width, frame.height);
      await infer(frame, ticket);
      const now = performance.now();
      const instant = 1000 / (now - last);
      fps = fps ? fps * 0.85 + instant * 0.15 : instant;
      last = now;
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  } catch (error) {
    if (ticket === version) {
      stop();
      status(cameraError(error), true);
      feedMessage("The webcam isn't running. Upload an image or use the sample image instead.");
    }
  } finally {
    starting = false;
    $("camera").disabled = false;
    updateToggle();
  }
}

async function analyzeFile(file, sourceId) {
  if (!file) return;
  setMode("image", sourceId);
  const ticket = version;
  let url;
  try {
    if (file.type && !file.type.startsWith("image/")) throw Error("That file isn't an image.");
    if (file.size > 15_000_000) throw Error("Choose an image smaller than 15 MB.");
    url = URL.createObjectURL(file);
    const img = new Image();
    img.src = url;
    await img.decode();
    if (ticket !== version) return;
    if (img.naturalWidth * img.naturalHeight > 40_000_000) throw Error("Choose an image smaller than 40 megapixels.");

    const scale = Math.min(1, 1920 / Math.max(img.naturalWidth, img.naturalHeight));
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    image = img;
    $("hud-res").textContent = `${canvas.width} × ${canvas.height}`;

    await loadModel();
    if (ticket !== version) return;
    await infer(image, ticket);
    if (ticket !== version) return;
    status(`Analyzed in ${Math.round(latency)} ms. Drag the confidence slider to explore.`);
  } catch (error) {
    if (ticket === version) {
      status(error.message || "Could not read that image.", true);
      feedMessage("Nothing analyzed yet.");
    }
  } finally {
    if (url) URL.revokeObjectURL(url);
  }
}

/* ---------- Wiring ---------- */

$("recorded").onclick = () => setMode("recorded", "recorded");
$("camera").onclick = startCamera;
$("upload-button").onclick = () => $("upload").click();
$("upload").onchange = (event) => {
  analyzeFile(event.target.files[0], "upload-button");
  event.target.value = "";
};
$("example").onclick = async () => {
  $("example").disabled = true;
  try {
    const response = await fetch("crosswalk-sample.jpg"); // clean frame, no baked-in boxes
    if (!response.ok) throw Error("Could not load the sample image.");
    await analyzeFile(await response.blob(), "example");
  } catch (error) {
    status(error.message, true);
  } finally {
    $("example").disabled = false;
  }
};

$("toggle").onclick = () => {
  if (mode === "recorded") {
    const video = $("recording");
    if (video.paused) video.play().catch(() => {});
    else video.pause();
  } else if (mode === "camera") {
    if (stream) stop();
    else startCamera();
  }
};

const recording = $("recording");
recording.onplay = recording.onpause = () => {
  if (mode !== "recorded") return;
  setRec("", recording.paused ? "PAUSED" : "PLAYBACK");
  updateToggle();
};
recording.ontimeupdate = () => {
  if (mode === "recorded") $("hud-left").textContent = `FRAME ${String(Math.floor(recording.currentTime * RECORDED_FPS)).padStart(4, "0")}`;
};

function redraw() {
  if (mode === "image" && image) draw(image);
}

$("confidence").oninput = () => {
  $("confidence-value").textContent = `${$("confidence").value}%`;
  redraw();
};

document.querySelectorAll(".chip").forEach((chip) => {
  chip.onclick = () => {
    filter = chip.dataset.filter;
    document.querySelectorAll(".chip").forEach((other) => {
      other.classList.toggle("is-active", other === chip);
      other.setAttribute("aria-pressed", String(other === chip));
    });
    redraw();
  };
});

$("export").onclick = () => {
  const round = (value) => Math.round(value * 10) / 10;
  download(JSON.stringify({
    model: "YOLO11n",
    mode,
    width: canvas.width,
    height: canvas.height,
    threshold: Number($("confidence").value) / 100,
    classFilter: filter,
    inferenceMs: round(latency),
    detections: visible().map(({ x, y, w, h, score, label, classId }) => ({
      label, classId, score: Math.round(score * 1e4) / 1e4, x: round(x), y: round(y), w: round(w), h: round(h),
    })),
  }, null, 2), "yolo-detections.json", "application/json");
};

// Drop or paste an image anywhere on the page.
const stage = $("stage");
let dragDepth = 0;
const hasFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", (event) => {
  if (!hasFiles(event)) return;
  dragDepth++;
  stage.classList.add("is-dragging");
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) stage.classList.remove("is-dragging");
});
window.addEventListener("dragover", (event) => {
  if (hasFiles(event)) event.preventDefault();
});
window.addEventListener("drop", (event) => {
  if (!hasFiles(event)) return;
  event.preventDefault();
  dragDepth = 0;
  stage.classList.remove("is-dragging");
  analyzeFile(event.dataTransfer.files[0], "upload-button");
});
document.addEventListener("paste", (event) => {
  const file = [...(event.clipboardData?.files || [])].find((item) => item.type.startsWith("image/"));
  if (file) analyzeFile(file, "upload-button");
});

window.addEventListener("pagehide", stop);
document.addEventListener("visibilitychange", () => {
  if (document.hidden && mode === "camera") stop();
});

updateToggle();
