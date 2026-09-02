import { test } from "node:test";
import assert from "node:assert/strict";
import { resize, resample, srgbToLinear, linearToSrgb, Filter } from "./index.js";

test("a 1px checkerboard reduces to half the light, ~188 not 128", async () => {
  const n = 256, ch = 3;
  const src = new Uint8Array(n * n * ch);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const v = (x + y) % 2 ? 255 : 0; for (let c = 0; c < ch; c++) src[(y * n + x) * ch + c] = v; }
  const out = await resize(src, n, n, ch, 16, 16, Filter.Box);
  let t = 0; for (let i = 0; i < out.length; i++) t += out[i];
  const mean = t / out.length;
  assert.ok(Math.abs(mean - 187.5) < 1.5, `mean ${mean}`);
});

test("srgb round trip is exact at 8 bit", async () => {
  const all = new Uint8Array(256); for (let i = 0; i < 256; i++) all[i] = i;
  const back = await linearToSrgb(await srgbToLinear(all));
  for (let i = 0; i < 256; i++) assert.equal(back[i], i);
});

test("unit scale is identity for Box", async () => {
  const src = new Float32Array(64 * 64).map((_, i) => (i % 251) / 251);
  const out = await resample(src, 64, 64, 1, 64, 64, Filter.Box);
  for (let i = 0; i < src.length; i++) assert.ok(Math.abs(src[i] - out[i]) < 1e-5);
});
