#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//! Shared fixtures and the incumbents' adapters, so the tests and the bench
//! measure the same code paths.

use fast_image_resize as fir;
use fir::{images::Image, FilterType as FirFilter, PixelType, ResizeAlg, ResizeOptions, Resizer};
use halflight::{resample, Filter};
use image::{imageops::FilterType as ImgFilter, ImageBuffer, Rgb};

/// A 1px checkerboard in sRGB u8, `ch` channels. Exactly half black and half
/// white by construction, so its correct reduction is one number.
pub fn checkerboard(n: usize, ch: usize) -> Vec<u8> {
    let mut v = vec![0u8; n * n * ch];
    for y in 0..n {
        for x in 0..n {
            let s = if (x + y) % 2 == 0 { 0 } else { 255 };
            for c in 0..ch {
                v[(y * n + x) * ch + c] = s;
            }
        }
    }
    v
}

/// Deterministic photo-shaped noise: a soft gradient with per-pixel texture.
/// Not a photo, and not meant to stand in for one; it is what the bench runs
/// on when no corpus directory is given, so the numbers are reproducible.
pub fn synthetic_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut v = vec![0u8; w * h * 3];
    let mut s: u32 = 0x9E37_79B9;
    for y in 0..h {
        for x in 0..w {
            let g = (x as f32 / w as f32) * 180.0 + (y as f32 / h as f32) * 60.0;
            for c in 0..3 {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let n = (s % 41) as f32 - 20.0;
                v[(y * w + x) * 3 + c] = (g + n + c as f32 * 7.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    v
}

pub fn interior_mean(px: &[u8], w: usize, h: usize, ch: usize, margin: usize) -> f64 {
    let (mut t, mut n) = (0f64, 0usize);
    for y in margin..h - margin {
        for x in margin..w - margin {
            for c in 0..ch {
                t += px[(y * w + x) * ch + c] as f64;
                n += 1;
            }
        }
    }
    t / n as f64
}

// ── halflight, end to end on u8 sRGB ────────────────────────────────────────
// Decode with the LUT, resample in linear f32, encode back. This is the whole
// job, and it is the number that competes with fir's mapper path.

pub fn halflight_u8(src: &[u8], sw: usize, sh: usize, ch: usize, dw: usize, dh: usize, f: Filter) -> Vec<u8> {
    let out = resample(&halflight::decode_srgb(src), sw, sh, ch, dw, dh, f);
    halflight::encode_srgb(&out)
}

/// The kernel alone, on linear f32 already in hand. Comparable to feeding the
/// incumbents f32 directly.
pub fn halflight_f32(
    src: &[f32],
    sw: usize,
    sh: usize,
    ch: usize,
    dw: usize,
    dh: usize,
    f: Filter,
) -> Vec<f32> {
    resample(src, sw, sh, ch, dw, dh, f)
}

// ── image 0.25 ──────────────────────────────────────────────────────────────

pub fn image_filter(f: Filter) -> ImgFilter {
    match f {
        // image has no Box; Triangle is its nearest non-ringing choice.
        Filter::Box => ImgFilter::Triangle,
        Filter::Lanczos3 => ImgFilter::Lanczos3,
        Filter::Mitchell => ImgFilter::CatmullRom,
    }
}

/// image's default path: u8 in, u8 out, averaging encoded values.
pub fn image_u8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, f: Filter) -> Vec<u8> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(sw as u32, sh as u32, src.to_vec()).unwrap();
    image::imageops::resize(&img, dw as u32, dh as u32, image_filter(f)).into_raw()
}

/// image fed linear f32 directly: the kernel alone, gamma-correct because the
/// caller made it so. This is the parity oracle for the kernel.
pub fn image_f32(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize, f: Filter) -> Vec<f32> {
    let img: ImageBuffer<Rgb<f32>, Vec<f32>> =
        ImageBuffer::from_raw(sw as u32, sh as u32, src.to_vec()).unwrap();
    image::imageops::resize(&img, dw as u32, dh as u32, image_filter(f)).into_raw()
}

// ── fast_image_resize 6 ─────────────────────────────────────────────────────

pub fn fir_filter(f: Filter) -> FirFilter {
    match f {
        Filter::Box => FirFilter::Box,
        Filter::Lanczos3 => FirFilter::Lanczos3,
        Filter::Mitchell => FirFilter::Mitchell,
    }
}

fn fir_opts(f: Filter) -> ResizeOptions {
    ResizeOptions::new().resize_alg(ResizeAlg::Convolution(fir_filter(f)))
}

/// fir's default path: U8x3 in, U8x3 out, SIMD, averaging encoded values.
pub fn fir_u8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, f: Filter) -> Vec<u8> {
    let src_img = Image::from_vec_u8(sw as u32, sh as u32, src.to_vec(), PixelType::U8x3).unwrap();
    let mut dst = Image::new(dw as u32, dh as u32, PixelType::U8x3);
    Resizer::new().resize(&src_img, &mut dst, &fir_opts(f)).unwrap();
    dst.into_vec()
}

/// fir's CORRECT path, opt-in: sRGB u8 -> linear u16 through its own mapper,
/// resize U16x3, map back. This is what "gamma-correct with fir" costs.
pub fn fir_u8_srgb_mapped(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, f: Filter) -> Vec<u8> {
    let mapper = fir::create_srgb_mapper();
    let src_img = Image::from_vec_u8(sw as u32, sh as u32, src.to_vec(), PixelType::U8x3).unwrap();
    let mut src_lin = Image::new(sw as u32, sh as u32, PixelType::U16x3);
    mapper.forward_map(&src_img, &mut src_lin).unwrap();
    let mut dst_lin = Image::new(dw as u32, dh as u32, PixelType::U16x3);
    Resizer::new()
        .resize(&src_lin, &mut dst_lin, &fir_opts(f))
        .unwrap();
    let mut dst = Image::new(dw as u32, dh as u32, PixelType::U8x3);
    mapper.backward_map(&dst_lin, &mut dst).unwrap();
    dst.into_vec()
}

/// fir fed linear f32 directly (F32x3): kernel-only parity oracle.
pub fn fir_f32(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize, f: Filter) -> Vec<f32> {
    let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let src_img = Image::from_vec_u8(sw as u32, sh as u32, bytes, PixelType::F32x3).unwrap();
    let mut dst = Image::new(dw as u32, dh as u32, PixelType::F32x3);
    Resizer::new().resize(&src_img, &mut dst, &fir_opts(f)).unwrap();
    dst.into_vec()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect()
}
