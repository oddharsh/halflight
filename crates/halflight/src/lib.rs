//! # halflight
//!
//! A separable image resampler that takes **planar linear f32** and returns the
//! same. That signature is the whole design: averaging encoded sRGB values, the
//! defect that darkens every downscaled texture, is not a default you must
//! remember to override here. It is unrepresentable.
//!
//! A one-pixel black and white checkerboard is exactly half black and half
//! white, so any honest downscale has to average it, and the right answer is
//! the sRGB encoding of *half the light*: 187.5. Averaging the encoded values
//! instead gives 127.5. Both major Rust image crates, macOS `sips`, `Bun.Image`
//! and the browser canvas all read within a code of 128 on their default path.
//!
//! ```
//! use halflight::{resample, srgb_to_linear, linear_to_srgb, Filter};
//!
//! // 64x64 checkerboard, one channel, decoded to linear light first.
//! let n = 64;
//! let src: Vec<f32> = (0..n * n)
//!     .map(|i| srgb_to_linear(if ((i / n) + (i % n)) % 2 == 0 { 0 } else { 255 }))
//!     .collect();
//!
//! let out = resample(&src, n, n, 1, 4, 4, Filter::Box);
//! let mean = out.iter().sum::<f32>() / out.len() as f32;
//!
//! assert!((mean - 0.5).abs() < 0.02);          // half the LIGHT
//! assert_eq!(linear_to_srgb(mean), 188);        // which encodes to ~188, not 128
//! ```
//!
//! Three things make a resampler correct, and every one of them is something
//! shipping software gets wrong:
//!
//! 1. **Support scales with the reduction.** On a downscale the filter has to
//!    low-pass at the *output* Nyquist, so its support in source pixels widens
//!    by the reduction factor. A kernel evaluated at fixed width is sampling,
//!    not filtering, and it aliases.
//! 2. **The average is over light.** sRGB encodes perceived brightness, not
//!    light. This crate's input type makes that the caller's decode step, not
//!    the resampler's problem.
//! 3. **Weights are normalised per output sample.** The window clips at the
//!    image edge; the surviving weights must renormalise or the border darkens
//!    toward zero, which shows up as a vignette nobody can explain.
//!
//! The kernel is free of I/O and of any image type. Decode to linear f32 with
//! the transfer functions here (or your own), resample, encode back.

#![forbid(unsafe_code)]
// The accumulation loops are written by index on purpose: the summation ORDER
// is the contract that makes the fixed-channel and dynamic paths bitwise equal,
// and an iterator rewrite is the kind of "cleanup" that would quietly break it.
#![allow(clippy::needless_range_loop)]

/// How the filter weights fall off with distance from the sample centre.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Filter {
    /// Area average. Zero ringing by construction, since no weight is negative,
    /// and exactly the analytic answer when the reduction is an integer factor.
    Box,
    /// Windowed sinc, three lobes. Sharper than Box and it rings, because the
    /// lobes are negative.
    Lanczos3,
    /// Cubic with B=C=1/3. Between the two: mild ringing, less softening.
    /// Approximating rather than interpolating, so it is not identity at unit
    /// scale; see the test that pins that.
    Mitchell,
}

impl Filter {
    /// Half-width in OUTPUT-normalised units, before the reduction factor
    /// widens it.
    fn support(self) -> f32 {
        match self {
            Filter::Box => 0.5,
            Filter::Lanczos3 => 3.0,
            Filter::Mitchell => 2.0,
        }
    }

    fn eval(self, t: f32) -> f32 {
        let t = t.abs();
        match self {
            Filter::Box => {
                // Half-open so a sample exactly on the boundary is counted once
                // rather than by both neighbours.
                if t < 0.5 {
                    1.0
                } else if t == 0.5 {
                    0.5
                } else {
                    0.0
                }
            }
            Filter::Lanczos3 => {
                if t < 1e-7 {
                    1.0
                } else if t < 3.0 {
                    let pt = std::f32::consts::PI * t;
                    (pt.sin() / pt) * ((pt / 3.0).sin() / (pt / 3.0))
                } else {
                    0.0
                }
            }
            Filter::Mitchell => {
                const B: f32 = 1.0 / 3.0;
                const C: f32 = 1.0 / 3.0;
                let t2 = t * t;
                if t < 1.0 {
                    ((12.0 - 9.0 * B - 6.0 * C) * t * t2
                        + (-18.0 + 12.0 * B + 6.0 * C) * t2
                        + (6.0 - 2.0 * B))
                        / 6.0
                } else if t < 2.0 {
                    ((-B - 6.0 * C) * t * t2
                        + (6.0 * B + 30.0 * C) * t2
                        + (-12.0 * B - 48.0 * C) * t
                        + (8.0 * B + 24.0 * C))
                        / 6.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// The taps for one output sample: where they start and what they weigh.
struct Taps {
    first: usize,
    weights: Vec<f32>,
}

/// Weights for every output sample along one axis.
///
/// `center` is the source coordinate the output sample sits on, derived from
/// PIXEL CENTRES rather than edges. Getting that half-pixel wrong shifts the
/// whole image by half an output pixel, which reads as "slightly blurry" and is
/// actually a misalignment.
fn plan(src_len: usize, dst_len: usize, f: Filter) -> Vec<Taps> {
    let scale = dst_len as f32 / src_len as f32;
    // On a downscale the filter widens; on an upscale it does not.
    let widen = if scale < 1.0 { 1.0 / scale } else { 1.0 };
    let support = f.support() * widen;
    let mut out = Vec::with_capacity(dst_len);
    for i in 0..dst_len {
        let center = (i as f32 + 0.5) / scale - 0.5;
        let first = ((center - support).ceil().max(0.0)) as usize;
        let last = ((center + support).floor().min(src_len as f32 - 1.0)) as usize;
        let mut weights = Vec::with_capacity(last.saturating_sub(first) + 1);
        let mut sum = 0.0f32;
        for s in first..=last {
            let w = f.eval((s as f32 - center) / widen);
            weights.push(w);
            sum += w;
        }
        // Renormalise, so a window clipped by the edge still integrates to 1.
        if sum != 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }
        out.push(Taps { first, weights });
    }
    out
}

/// Separable resample of interleaved planar f32 in linear light.
///
/// `src` holds `sw * sh * ch` samples, row-major, channels interleaved. The
/// result holds `dw * dh * ch` in the same layout. Any channel count works;
/// 1 and 3 take a monomorphised fast path.
///
/// Separable because a 2D filter of this family factors into two 1D passes,
/// which turns O(support²) per output pixel into O(support). At a 3.3×
/// reduction that is the difference between ~44 taps and ~13 per pixel.
///
/// The passes are monomorphised over the channel count for 1 and 3, with the
/// dynamic loop kept as the fallback for any other. The win is what a
/// compile-time channel count removes: the per-channel pass over the tap list
/// becomes one pass carrying a fixed-width accumulator, so each weight is
/// loaded once instead of `ch` times and the bounds checks fold.
///
/// Output is BITWISE identical between the fixed and dynamic shapes, and that
/// is a property of the summation order rather than luck: both accumulate each
/// channel over k ascending, and rustc does not reassociate f32 adds. A SIMD
/// pass that split the accumulator would break that order; the oracle test
/// below is the tripwire.
///
/// # Panics
///
/// In debug builds, if `src.len() != sw * sh * ch`.
pub fn resample(src: &[f32], sw: usize, sh: usize, ch: usize, dw: usize, dh: usize, f: Filter) -> Vec<f32> {
    debug_assert_eq!(src.len(), sw * sh * ch);
    match ch {
        1 => resample_fixed::<1>(src, sw, sh, dw, dh, f),
        3 => resample_fixed::<3>(src, sw, sh, dw, dh, f),
        _ => resample_dyn(src, sw, sh, ch, dw, dh, f),
    }
}

fn resample_fixed<const CH: usize>(
    src: &[f32],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    f: Filter,
) -> Vec<f32> {
    // Horizontal first: it shrinks the row length before the vertical pass has
    // to walk it, which is strictly less work when both axes reduce.
    let xplan = plan(sw, dw, f);
    let mut mid = vec![0.0f32; dw * sh * CH];
    for y in 0..sh {
        let row = &src[y * sw * CH..(y + 1) * sw * CH];
        for (x, tap) in xplan.iter().enumerate() {
            let mut acc = [0.0f32; CH];
            for (k, w) in tap.weights.iter().enumerate() {
                let p = &row[(tap.first + k) * CH..(tap.first + k + 1) * CH];
                for c in 0..CH {
                    acc[c] += p[c] * w;
                }
            }
            mid[(y * dw + x) * CH..(y * dw + x + 1) * CH].copy_from_slice(&acc);
        }
    }
    let yplan = plan(sh, dh, f);
    let mut dst = vec![0.0f32; dw * dh * CH];
    for (y, tap) in yplan.iter().enumerate() {
        for x in 0..dw {
            let mut acc = [0.0f32; CH];
            for (k, w) in tap.weights.iter().enumerate() {
                let i = ((tap.first + k) * dw + x) * CH;
                for c in 0..CH {
                    acc[c] += mid[i + c] * w;
                }
            }
            dst[(y * dw + x) * CH..(y * dw + x + 1) * CH].copy_from_slice(&acc);
        }
    }
    dst
}

/// The dynamic-channel loops: the fallback for channel counts other than 1
/// and 3, and the oracle the specialisation is tested against.
fn resample_dyn(src: &[f32], sw: usize, sh: usize, ch: usize, dw: usize, dh: usize, f: Filter) -> Vec<f32> {
    let xplan = plan(sw, dw, f);
    let mut mid = vec![0.0f32; dw * sh * ch];
    for y in 0..sh {
        for (x, tap) in xplan.iter().enumerate() {
            for c in 0..ch {
                let mut acc = 0.0f32;
                for (k, w) in tap.weights.iter().enumerate() {
                    acc += src[(y * sw + tap.first + k) * ch + c] * w;
                }
                mid[(y * dw + x) * ch + c] = acc;
            }
        }
    }
    let yplan = plan(sh, dh, f);
    let mut dst = vec![0.0f32; dw * dh * ch];
    for (y, tap) in yplan.iter().enumerate() {
        for x in 0..dw {
            for c in 0..ch {
                let mut acc = 0.0f32;
                for (k, w) in tap.weights.iter().enumerate() {
                    acc += mid[((tap.first + k) * dw + x) * ch + c] * w;
                }
                dst[(y * dw + x) * ch + c] = acc;
            }
        }
    }
    dst
}

// ── sRGB transfer ─────────────────────────────────────────────────────────────
// Exact round trip at 8 bit: all 256 values return to themselves, verified in
// the tests rather than assumed.

/// The forward transfer takes a u8, so it has exactly 256 answers and does not
/// need to be computed once per sample. On a 40-megapixel frame the `powf` per
/// sample was most of the resize; the table makes it a load. Built once,
/// lazily, from the same expression the table replaces, so there is no second
/// definition of sRGB to keep in step.
static SRGB_LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();

/// The sRGB decode table, 256 entries. Take it once and index it in a loop;
/// `srgb_to_linear` is the same lookup behind a per-call `OnceLock` read, and
/// that atomic load was 30 ms of a 24-megapixel decode when measured
/// (55.6 ms whole job, 23.1 ms of which is the kernel).
#[inline]
pub fn srgb_lut() -> &'static [f32; 256] {
    SRGB_LUT.get_or_init(|| std::array::from_fn(|i| srgb_to_linear_exact(i as u8)))
}

/// sRGB-encoded 8-bit value to linear light in `0.0..=1.0`.
#[inline]
pub fn srgb_to_linear(c: u8) -> f32 {
    srgb_lut()[c as usize]
}

/// Bulk decode: every sample through the table, the lock taken once.
pub fn decode_srgb(src: &[u8]) -> Vec<f32> {
    let lut = srgb_lut();
    src.iter().map(|&c| lut[c as usize]).collect()
}

/// Bulk encode. Exact: every sample goes through `linear_to_srgb`, so the
/// 8-bit round trip stays bit-perfect. This is the slow half of the transfer
/// step (a `powf` per output sample) and the obvious next optimisation; it is
/// left exact here rather than approximated, because an approximation that
/// broke the round-trip test would be a second definition of sRGB.
pub fn encode_srgb(src: &[f32]) -> Vec<u8> {
    src.iter().map(|&l| linear_to_srgb(l)).collect()
}

fn srgb_to_linear_exact(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.040_449_936 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear light to an sRGB-encoded 8-bit value. Clamps to `0.0..=1.0` first,
/// which is where Lanczos overshoot goes.
pub fn linear_to_srgb(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0);
    let s = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

// ── pure gamma 2.2 ────────────────────────────────────────────────────────────
// A separate pair rather than a parameter on the sRGB one, because the two
// curves must never blend: the piecewise linear toe is exactly what makes sRGB
// not-a-power-law, and a "close enough" hybrid is how a max-4-code shadow error
// comes back wearing a fix's name. Gray Gamma 2.2 is what a Leica M Monochrom
// declares in its ICC profile.

static G22_LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();

/// The gamma-2.2 decode table, 256 entries.
#[inline]
pub fn g22_lut() -> &'static [f32; 256] {
    G22_LUT.get_or_init(|| std::array::from_fn(|i| (i as f32 / 255.0).powf(2.2)))
}

/// Gamma-2.2-encoded 8-bit value to linear light.
#[inline]
pub fn g22_to_linear(c: u8) -> f32 {
    g22_lut()[c as usize]
}

/// Linear light to a gamma-2.2-encoded 8-bit value.
pub fn linear_to_g22(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0);
    (l.powf(1.0 / 2.2) * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g22_round_trip_is_exact_at_8_bit() {
        for c in 0u8..=255 {
            assert_eq!(linear_to_g22(g22_to_linear(c)), c, "g22 value {c} did not return");
        }
    }

    /// The two curves must actually differ where it matters, or the second
    /// pair is decoration. Largest gap is in the shadows, where sRGB's linear
    /// toe departs from the power law.
    #[test]
    fn g22_and_srgb_are_distinct_curves() {
        let mut diverge = 0;
        for c in 1u8..=254 {
            if (g22_to_linear(c) - srgb_to_linear(c)).abs() / srgb_to_linear(c).max(1e-9) > 0.01 {
                diverge += 1;
            }
        }
        assert!(
            diverge > 100,
            "curves nearly identical ({diverge} values differ >1%)"
        );
    }

    #[test]
    fn srgb_round_trip_is_exact_at_8_bit() {
        for c in 0u8..=255 {
            assert_eq!(linear_to_srgb(srgb_to_linear(c)), c, "value {c} did not return");
        }
    }

    /// Box and Lanczos3 are INTERPOLATING filters: their weight is 1 at zero
    /// offset and 0 at every other integer, so at unit scale each output sample
    /// reads exactly one input sample. Some libraries short-circuit this case
    /// to a copy, which makes the property untestable there; here the kernel
    /// actually runs at unit scale, which is why the test means something.
    #[test]
    fn unit_scale_is_identity_for_the_interpolating_filters() {
        let src: Vec<f32> = (0..64 * 64).map(|i| (i % 251) as f32 / 251.0).collect();
        for f in [Filter::Box, Filter::Lanczos3] {
            let out = resample(&src, 64, 64, 1, 64, 64, f);
            for (i, (a, b)) in src.iter().zip(out.iter()).enumerate() {
                assert!((a - b).abs() < 1e-5, "{f:?} moved sample {i}: {a} -> {b}");
            }
        }
    }

    /// Mitchell is NOT identity, and that is the filter rather than a defect.
    /// B=1/3 makes it approximating instead of interpolating: its weight at zero
    /// offset is (6-2B)/6 = 0.889, with the remainder spread to the neighbours,
    /// so it blurs slightly at every scale including 1.0.
    #[test]
    fn mitchell_is_approximating_and_so_blurs_at_unit_scale() {
        let src: Vec<f32> = (0..64 * 64).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let out = resample(&src, 64, 64, 1, 64, 64, Filter::Mitchell);
        let moved = src
            .iter()
            .zip(out.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-4)
            .count();
        assert!(
            moved > src.len() / 2,
            "Mitchell behaved as interpolating, which it is not"
        );
    }

    /// A checkerboard is half light, and half light is what must come out.
    #[test]
    fn checkerboard_reduces_to_half_light() {
        let n = 64;
        let src: Vec<f32> = (0..n * n)
            .map(|i| if ((i / n) + (i % n)) % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        for f in [Filter::Box, Filter::Lanczos3, Filter::Mitchell] {
            let out = resample(&src, n, n, 1, 4, 4, f);
            let mean = out.iter().sum::<f32>() / out.len() as f32;
            assert!((mean - 0.5).abs() < 0.02, "{f:?} gave {mean}, expected 0.5");
        }
    }

    /// Weights integrate to 1 everywhere, including where the window clips the
    /// edge. A flat field must stay flat rather than darkening at the border.
    #[test]
    fn a_flat_field_stays_flat_including_the_edges() {
        let src = vec![0.5f32; 100 * 100];
        for f in [Filter::Box, Filter::Lanczos3, Filter::Mitchell] {
            let out = resample(&src, 100, 100, 1, 30, 30, f);
            for (i, v) in out.iter().enumerate() {
                assert!((v - 0.5).abs() < 1e-4, "{f:?} sample {i} drifted to {v}");
            }
        }
    }

    /// The fixed-channel passes must be BITWISE equal to the dynamic loop they
    /// specialise. Bitwise rather than epsilon on purpose: both shapes
    /// accumulate each channel over k ascending and rustc does not reassociate
    /// f32, so exact equality is the contract, and an epsilon would let a
    /// reordering SIMD rewrite slip through the exact gate it needs to hit.
    /// Anyone content-addressing the output depends on this.
    #[test]
    fn fixed_channel_passes_match_the_dynamic_oracle_bitwise() {
        let (sw, sh) = (97, 61); // deliberately awkward, non-square, prime-ish
        for ch in [1usize, 3] {
            let src: Vec<f32> = (0..sw * sh * ch)
                .map(|i| ((i * 2654435761usize) % 1000) as f32 / 999.0)
                .collect();
            for f in [Filter::Box, Filter::Lanczos3, Filter::Mitchell] {
                for (dw, dh) in [(29, 17), (97, 61), (120, 80)] {
                    let a = resample(&src, sw, sh, ch, dw, dh, f);
                    let b = resample_dyn(&src, sw, sh, ch, dw, dh, f);
                    assert_eq!(a.len(), b.len());
                    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                        assert_eq!(x.to_bits(), y.to_bits(), "{f:?} ch={ch} {dw}x{dh} sample {i}");
                    }
                }
            }
        }
    }

    /// Box has no negative lobes, so it cannot overshoot. The other two can, and
    /// this pins which is which rather than leaving it to belief.
    #[test]
    fn box_does_not_ring_and_lanczos_does() {
        let n = 64;
        let src: Vec<f32> = (0..n * n)
            .map(|i| if (i % n) < n / 2 { 0.0 } else { 1.0 })
            .collect();
        let over = |f: Filter| {
            resample(&src, n, n, 1, 16, 16, f)
                .iter()
                .fold(0.0f32, |m, v| m.max((-*v).max(*v - 1.0)))
        };
        assert!(over(Filter::Box) < 1e-6, "Box rang");
        assert!(
            over(Filter::Lanczos3) > 1e-4,
            "Lanczos3 did not ring, so this test proves nothing"
        );
    }

    /// The bulk helpers are the per-sample functions applied in order, bitwise.
    #[test]
    fn bulk_transfer_matches_per_sample_bitwise() {
        let all: Vec<u8> = (0..=255).collect();
        let lin = decode_srgb(&all);
        for (i, &c) in all.iter().enumerate() {
            assert_eq!(lin[i].to_bits(), srgb_to_linear(c).to_bits());
        }
        let back = encode_srgb(&lin);
        assert_eq!(back, all, "bulk round trip is not exact");
    }

    // The table is an optimisation and must not be a second opinion about sRGB.
    #[test]
    fn the_lut_agrees_with_the_expression_it_replaces() {
        for c in 0..=255u8 {
            assert_eq!(
                srgb_to_linear(c).to_bits(),
                srgb_to_linear_exact(c).to_bits(),
                "lut disagrees at {c}"
            );
        }
    }
}
