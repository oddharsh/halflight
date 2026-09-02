# How halflight works

You don't need this to use `halflight`. This is for the curious.

## The short version

Every image resizer averages. sRGB is not light; it is an encoding of perceived
brightness, roughly light to the power 1/2.2. Average the encoded numbers and
you average the wrong thing: a one pixel black and white checkerboard, which
is half light by construction, comes back as sRGB 128 instead of 188, and every
texture in every downscaled photo darkens by the same mechanism.

`halflight`'s only input type is planar linear f32. Decode to light, resample,
encode back. The wrong average is not a default you forgot to override; it
cannot be written.

That's the whole trick. The kernel itself is the same separable convolution
everybody ships, and the parity tests prove it.

## The kernel

`resample` is a separable filter: one horizontal pass, one vertical. A 2D filter
of this family factors, which turns O(support²) per output pixel into
O(support). At a 3.3× reduction that is ~13 taps per pixel instead of ~44.

Three things make it correct, and something in the wild gets each one wrong:

1. **Support scales with the reduction.** On a downscale the filter has to
   low-pass at the *output* Nyquist, so its width in source pixels grows by the
   reduction factor. A kernel evaluated at fixed width is sampling, not
   filtering, and it aliases. (`plan()` widens `support` by `1/scale`.)
2. **Sample centres, not edges.** The output sample `i` sits on source
   coordinate `(i + 0.5) / scale - 0.5`. Get the half pixel wrong and the whole
   image shifts by half an output pixel, which reads as "slightly blurry".
3. **Weights renormalise per output sample.** The window clips at the image
   edge; the surviving weights are rescaled to sum to 1, or the border darkens
   toward zero and shows up as a vignette nobody can explain.

The passes are monomorphised over the channel count for 1 and 3 channels. The
win is not vectorisation so much as what a compile-time channel count removes:
the per-channel walk over the tap list becomes one walk carrying a fixed-width
accumulator, so each weight is loaded once instead of `ch` times and the bounds
checks fold. Measured on 5952×3968 RGB → 900×600 Box: 42.9 → 22.9 ms for the
kernel, output byte-identical.

**Byte-identical is a property of summation order, not luck.** Both shapes
accumulate each channel over the taps in ascending order, and rustc does not
reassociate f32 adds, so the sums are bitwise the same. A SIMD pass that split
the accumulator would break that order. The oracle test in the crate is the
tripwire, and it is bitwise on purpose: an epsilon would let exactly that
rewrite through. If you content-address your outputs, you depend on this.

## Parity

Fed identical linear f32, the kernel agrees with both incumbents' kernels.
Enforced in CI on every commit (`conformance/tests/parity.rs`):

| filter | vs `image` 0.25 | vs `fast_image_resize` 6.1 |
|---|---:|---:|
| Box | (image has no Box) | max abs diff 1.2e-7 |
| Lanczos3 | 3.8e-6 | 1.9e-6 |

Those are f32 rounding. `halflight` is not a different resampler. It is the
same resampler behind an input type that refuses the wrong colour path.

The checkerboard is committed as a test too (`conformance/tests/checkerboard.rs`),
in both directions: `halflight` and fir's opt-in mapper must read ~188, and the
incumbents' *default* paths must read ~128. The second half exists so that the
day one of them changes its default, the README's opening claim goes red rather
than stale.

## What it costs

The honest number is the whole job on 8-bit sRGB: decode, resample in linear
light, encode. Measured with one process per cell, its own warm-up, best of 5.

24 MP frame (5952×3968) → 900×600, ms:

| path | Box | Lanczos3 | colour |
|---|---:|---:|---|
| `halflight` u8 whole job | 43.9 | 123.6 | correct |
| `halflight` kernel, f32 in hand | 23.1 | 99.6 | correct |
| `fast_image_resize` sRGB mapper → U16x3 → back (opt-in) | 37.5 | 63.0 | correct |
| `fast_image_resize` kernel, F32x3 | 19.6 | 102.0 | correct |
| `fast_image_resize` default U8x3 | 8.4 | 28.6 | **wrong** |
| `image` kernel, Rgb32F | 138.8 | 460.3 | correct |
| `image` default | 58.8 | 159.4 | **wrong** |

Read it in three parts:

- **Kernel to kernel on f32, `halflight` and `fast_image_resize` are at
  parity** (23.1 vs 19.6, 99.6 vs 102.0), and both are 5-6× ahead of `image`'s.
- **fir's opt-in correct path is faster than `halflight`'s whole job**, by 1.2×
  on Box and 2× on Lanczos3. Two reasons, both real. Its mapper quantises to
  u16 linear and resizes that, and a u16 lane is half an f32 lane in SIMD, so
  its Lanczos3 at 63 ms beats even its own f32 kernel at 102. And `halflight`'s
  transfer step is scalar: the decode table hoist that took the Box job from
  55.6 to 43.9 ms is in, and the encode is still one `powf` per output sample.
  That gap is the roadmap, not a mystery.
- **The fastest number in the table is the wrong one.** fir's default U8x3
  path at 8.4 ms averages encoded values. That is the trade every default
  makes for you, and the reason this crate's API has no such path.

`halflight` chooses f32 linear over u16 linear deliberately: no quantisation
in the shadows, where sRGB's linear toe puts the fewest codes and where the
encoded-average error is largest. Whether that precision is worth 2× on
Lanczos3 is your call to make; it is not one made for you by a default.

## Benchmark methodology

`cargo run --release -p conformance -- bench` runs every implementation ×
shape × filter in its own child process, with two warm-up calls, and keeps the
best of 5 timed calls. Shared-process harnesses let warm-up, allocator state
and cache residency leak between implementations, and best-of-5 in a child is
the cheap way to not have that argument. The source is deterministic synthetic
noise (`synthetic_rgb`), so the numbers are reproducible; `-- corpus <dir>`
runs the same paths over real photos.

Two honest caveats. Milliseconds wobble a few percent run to run. And the
synthetic source is photo-*shaped*, not a photo; the corpus command exists
because a real sensor frame compresses and caches differently.

## Size

The wasm surface is 29 KB raw, 13.8 KB gzip, built with `opt-level = "s"` and
`wasm-opt`. The crate itself has zero dependencies. CI gates the wasm against a
40 KB gzip budget.
