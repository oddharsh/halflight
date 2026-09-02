# halflight

`halflight` is an image resampler whose input type makes the classic downscale
defect unrepresentable. It takes **planar linear f32** and returns the same.
The kernel is the one everybody ships, and the parity tests prove it; what
differs is that averaging encoded sRGB values, the thing that darkens every
downscaled texture, cannot be written against this API.

A one pixel black and white checkerboard is exactly half black and half
white, so any honest downscale has to average it. The right answer is the
sRGB encoding of *half the light*, **187.5**. Here is what reads it, measured
(`cargo run --release -p conformance -- checkerboard`):

| path | reads |
|---|---:|
| `halflight` | **188.0** |
| `fast_image_resize`, sRGB mapper → U16x3 (opt-in) | 188.0 |
| `fast_image_resize`, default U8x3 | 128.0 |
| `image`, default | 127.0 |
| macOS `sips` | 127.6 |
| `Bun.Image` | 128.0 |
| browser canvas `drawImage` | 127.0 |

Every default on that list averages encoded values. The two that get it right
are the two you have to opt into.

```rust
use halflight::{resample, decode_srgb, encode_srgb, Filter};

// sRGB u8 in, sRGB u8 out, and the average happens in light.
let lin = decode_srgb(&rgb8);                         // u8 -> linear f32, table lookup
let out = resample(&lin, w, h, 3, 900, 600, Filter::Lanczos3);
let rgb8 = encode_srgb(&out);                         // linear f32 -> u8, exact round trip
```

Zero dependencies. `#![forbid(unsafe_code)]`. MSRV 1.75. Also on npm as a
13.8 KB (gzip) wasm module with the same three calls.

## Install

```bash
cargo add halflight
```

```bash
npm i halflight
```

```js
import { resize, Filter } from "halflight";

// ImageData-shaped bytes in, resized bytes out, averaged in linear light.
const out = await resize(rgba, sw, sh, 4, dw, dh, Filter.Lanczos3);
```

The JavaScript surface is `resample`, `srgbToLinear`, `linearToSrgb` and the
convenience `resize`. The wasm loads lazily on first call and works in
browsers, Node, Bun and Deno.

## What it is, and what it is not

`halflight` is **not a different resampler**. Fed identical linear f32, its
kernel agrees with `image`'s and `fast_image_resize`'s to f32 rounding,
enforced in CI on every commit:

| filter | vs `image` 0.25 | vs `fast_image_resize` 6.1 |
|---|---:|---:|
| Box | n/a (image has no Box) | 1.2e-7 |
| Lanczos3 | 3.8e-6 | 1.9e-6 |

It **is** three things the incumbents make optional or get wrong by default,
each with a test that has an analytically known answer rather than a
reference implementation to agree with:

- **The average is over light.** The checkerboard above. The API's input type
  is the enforcement.
- **Support scales with the reduction.** A downscale must low-pass at the
  output Nyquist, so the filter widens by the reduction factor. A flat field
  stays flat to 1e-4 at every scale, including the edges, where weights
  renormalise per output sample so the border does not vignette.
- **Interpolating filters are identity at unit scale.** Box and Lanczos3 return
  the input untouched at 1:1. `image` short-circuits that case to a copy, so
  the property is untestable there; here the kernel actually runs.

Mitchell is deliberately *approximating* (B = 1/3) and blurs at every scale
including 1.0; that is pinned by a test too, so it never reads as a failure of
the identity one.

## How fast

The honest number is the whole job on 8-bit sRGB: decode, resample in linear
light, encode. One process per cell, its own warm-up, best of 5, on a 24 MP
frame (5952×3968) → 900×600. Milliseconds.

| path | Box | Lanczos3 | colour |
|---|---:|---:|---|
| `halflight`, u8 whole job | 43.9 | 123.6 | correct |
| `halflight`, kernel only (f32 in hand) | 23.1 | 99.6 | correct |
| `fast_image_resize`, sRGB mapper → U16x3 → back | 37.5 | 63.0 | correct |
| `fast_image_resize`, kernel only (F32x3) | 19.6 | 102.0 | correct |
| `fast_image_resize`, default U8x3 | 8.4 | 28.6 | **wrong** |
| `image`, kernel only (Rgb32F) | 138.8 | 460.3 | correct |
| `image`, default | 58.8 | 159.4 | **wrong** |

Three things to read off that, and the second is the one a benchmark table
usually hides:

1. **Kernel to kernel, `halflight` and `fast_image_resize` are at parity**, and
   both are 5-6× ahead of `image`.
2. **fir's opt-in correct path beats `halflight`'s whole job**, 1.2× on Box and
   2× on Lanczos3. Its mapper quantises to u16 linear and resizes that in SIMD,
   where a u16 lane is half an f32 lane; `halflight` stays f32 linear, which
   costs no precision in the shadows and pays for it here. The rest of the gap
   is `halflight`'s transfer step, which is scalar today (the encode is one
   `powf` per output sample). That is the roadmap, and
   [how-it-works.md](docs/how-it-works.md) says how much of the gap is which.
3. **The fastest row is the wrong one.** 8.4 ms averages encoded values.

`fast_image_resize` is excellent, and if u16 linear is precision enough for
you its mapper path is the faster correct option today. What `halflight`
offers is that the correct path is the *only* path, at f32, with zero
dependencies and a surface small enough to read in one sitting.

### Real photos

Synthetic noise is photo-shaped, not a photo. The same paths over 46 straight-
out-of-camera JPEGs (24 MP Fujifilm and Leica frames, long edge to 900 px, Box
unless noted), mean of best-of-5 per file:

| path | ms | colour |
|---|---:|---|
| `fast_image_resize`, default U8x3 | 10.9 | **wrong** |
| `fast_image_resize`, sRGB mapper → U16x3 → back | 44.7 | correct |
| `halflight`, u8 whole job | 61.9 | correct |
| `image`, default, Lanczos3 | 230.0 | **wrong** |

`cargo run --release -p conformance -- corpus <dir>` reproduces it on any
directory of JPEG or PNG files.

Methodology, caveats and the wasm size budget are in
[how-it-works.md](docs/how-it-works.md). `pnpm`-style reproducibility:
`cargo run --release -p conformance -- bench`.

## Gamma 2.2

Some sources declare a pure power curve rather than sRGB (a Leica M Monochrom's
ICC profile says Gray Gamma 2.2). `g22_to_linear` / `linear_to_g22` are a
separate pair rather than a parameter on the sRGB one, because the two curves
must never blend: sRGB's linear toe is exactly what makes it not a power law,
and a "close enough" hybrid is how a 4-code shadow error comes back wearing a
fix's name.

## Credits

The kernel, the analytic property tests and the checkerboard measurement were
written for [aadhar.sh](https://aadhar.sh)'s photo pipeline and lived there
first; the story of how the instrument kept being the bug is at
[aadhar.sh/garage/resample](https://aadhar.sh/garage/resample). The parity
oracles are `image` and `fast_image_resize`, which get the kernel right and
made it possible to prove this one does too. The shape of this repository,
a small engine beside a never-published conformance package that holds the
oracles and the bench, is borrowed from
[shadcn-ui/cn](https://github.com/shadcn-ui/cn).

## License

MIT OR Apache-2.0, at your option.
