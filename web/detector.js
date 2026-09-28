// YOLOv8 output decoding: [1, 84, N] tensor -> boxes in source-image pixels.
// Channels 0-3 are cx, cy, w, h in 640x640 letterboxed space; 4-83 are COCO class scores.

export const classes = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
  "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
  "teddy bear", "hair drier", "toothbrush",
];

const INPUT = 640;
const NMS_IOU = 0.45;
const MAX_CANDIDATES = 500;
const MAX_DETECTIONS = 100;

/** Coarse grouping used for the class filter and box colors. */
export function classGroup(classId) {
  if (classId === 0) return "person";
  if (classId >= 1 && classId <= 8) return "vehicle";
  if (classId >= 14 && classId <= 23) return "animal";
  return "other";
}

export function iou(a, b) {
  const w = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const h = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const intersection = w * h;
  return intersection / (a.w * a.h + b.w * b.h - intersection || 1);
}

export function decode(data, dims, threshold, width, height) {
  if (dims.length !== 3 || dims[1] !== 84) throw Error("Unexpected model output shape.");

  const n = dims[2];
  const scale = Math.min(INPUT / width, INPUT / height);
  const dx = (INPUT - width * scale) / 2;
  const dy = (INPUT - height * scale) / 2;
  const boxes = [];

  for (let i = 0; i < n; i++) {
    let score = 0;
    let cls = 0;
    for (let c = 0; c < 80; c++) {
      const s = data[(c + 4) * n + i];
      if (s > score) {
        score = s;
        cls = c;
      }
    }
    if (score < threshold) continue;

    const cx = data[i];
    const cy = data[n + i];
    const bw = data[2 * n + i];
    const bh = data[3 * n + i];
    const x = Math.max(0, (cx - bw / 2 - dx) / scale);
    const y = Math.max(0, (cy - bh / 2 - dy) / scale);
    const right = Math.min(width, (cx + bw / 2 - dx) / scale);
    const bottom = Math.min(height, (cy + bh / 2 - dy) / scale);
    if (right > x && bottom > y) {
      boxes.push({ x, y, w: right - x, h: bottom - y, score, label: classes[cls], classId: cls });
    }
  }

  // Class-aware non-maximum suppression.
  boxes.sort((a, b) => b.score - a.score);
  const selected = [];
  for (const box of boxes.slice(0, MAX_CANDIDATES)) {
    if (!selected.some((kept) => kept.classId === box.classId && iou(kept, box) > NMS_IOU)) {
      selected.push(box);
    }
    if (selected.length >= MAX_DETECTIONS) break;
  }
  return selected;
}
