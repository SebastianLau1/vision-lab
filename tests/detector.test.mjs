import test from "node:test";
import assert from "node:assert/strict";
import { classes, classGroup, decode, iou } from "../web/detector.js";

test("class list matches COCO output channels", () => {
  assert.equal(classes.length, 80);
  assert.equal(classes[0], "person");
  assert.equal(classes[79], "toothbrush");
});

test("NMS removes same-class overlap while preserving different classes", () => {
  const n = 3;
  const data = new Float32Array(84 * n);
  for (let i = 0; i < n; i++) {
    data[i] = 320;
    data[n + i] = 320;
    data[2 * n + i] = 100;
    data[3 * n + i] = 100;
  }
  data[4 * n] = 0.9; // anchor 0: person
  data[4 * n + 1] = 0.8; // anchor 1: person, same box, suppressed
  data[5 * n + 2] = 0.7; // anchor 2: bicycle, same box, kept
  const out = decode(data, [1, 84, n], 0.35, 1280, 720);
  assert.equal(out.length, 2);
  assert.equal(out[0].label, "person");
  assert.equal(out[1].label, "bicycle");
  assert.equal(out[0].w, 200);
  assert.equal(out[0].y, 260);
});

test("threshold and shape validation", () => {
  assert.deepEqual(decode(new Float32Array(84), [1, 84, 1], 0.5, 640, 640), []);
  assert.throws(() => decode([], [], 0.5, 640, 640), /shape/);
  assert.equal(iou({ x: 0, y: 0, w: 10, h: 10 }, { x: 20, y: 20, w: 5, h: 5 }), 0);
});

test("class groups drive the filter chips and box colors", () => {
  assert.equal(classGroup(0), "person");
  for (const name of ["bicycle", "car", "motorcycle", "bus", "truck", "boat"]) {
    assert.equal(classGroup(classes.indexOf(name)), "vehicle", name);
  }
  for (const name of ["bird", "dog", "giraffe"]) {
    assert.equal(classGroup(classes.indexOf(name)), "animal", name);
  }
  assert.equal(classGroup(classes.indexOf("traffic light")), "other");
  assert.equal(classGroup(classes.indexOf("laptop")), "other");
});
