// halflight for JavaScript. The wasm module is instantiated once, lazily, so
// importing this file costs nothing until the first call.
import init, * as wasm from "./pkg/halflight_wasm.js";

let ready = null;
function load() {
  if (!ready) ready = instantiate();
  return ready;
}

// wasm-pack's web target locates the .wasm with fetch(new URL(..., import.meta.url)),
// which browsers and Deno resolve and Node does not (fetch on a file: URL is
// "not implemented... yet"). In Node, read the bytes and hand them over; the
// import of node:fs is dynamic so a bundler targeting the browser never sees it.
async function instantiate() {
  const isNode = typeof process !== "undefined" && !!process.versions?.node;
  if (!isNode) return init();
  const { readFile } = await import("node:fs/promises");
  const { fileURLToPath } = await import("node:url");
  const bytes = await readFile(fileURLToPath(new URL("./pkg/halflight_wasm_bg.wasm", import.meta.url)));
  return init({ module_or_path: bytes });
}

export const Filter = Object.freeze({ Box: 0, Lanczos3: 1, Mitchell: 2 });

/** Resample interleaved linear f32. Returns a new Float32Array of dw*dh*ch. */
export async function resample(src, sw, sh, ch, dw, dh, filter = Filter.Box) {
  await load();
  return wasm.resample(src, sw, sh, ch, dw, dh, filter);
}

/** sRGB u8 (e.g. ImageData.data without alpha) to linear f32. */
export async function srgbToLinear(u8) { await load(); return wasm.srgb_to_linear(u8); }

/** Linear f32 to sRGB u8, clamped. */
export async function linearToSrgb(f32) { await load(); return wasm.linear_to_srgb(f32); }

/**
 * The whole job on sRGB u8: decode, resample in linear light, encode.
 * `ch` is 1, 3 or 4; alpha is resampled as a channel, which is correct for
 * unassociated alpha only if you premultiply first.
 */
export async function resize(u8, sw, sh, ch, dw, dh, filter = Filter.Box) {
  await load();
  const lin = wasm.srgb_to_linear(u8);
  const out = wasm.resample(lin, sw, sh, ch, dw, dh, filter);
  return wasm.linear_to_srgb(out);
}
