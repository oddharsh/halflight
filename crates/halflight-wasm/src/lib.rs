//! The JavaScript surface. Same contract as the crate: planar interleaved f32
//! in linear light, in and out. The two sRGB helpers exist so a caller can go
//! from `ImageData`-shaped bytes to linear and back without writing the
//! transfer function in JS, which is the step everyone gets wrong.

use wasm_bindgen::prelude::*;

fn filter(id: u8) -> halflight::Filter {
    match id {
        0 => halflight::Filter::Box,
        1 => halflight::Filter::Lanczos3,
        2 => halflight::Filter::Mitchell,
        _ => halflight::Filter::Box,
    }
}

/// Resample interleaved linear f32. `filter`: 0 Box, 1 Lanczos3, 2 Mitchell.
#[wasm_bindgen]
pub fn resample(
    src: &[f32],
    sw: usize,
    sh: usize,
    ch: usize,
    dw: usize,
    dh: usize,
    filter_id: u8,
) -> Vec<f32> {
    halflight::resample(src, sw, sh, ch, dw, dh, filter(filter_id))
}

/// sRGB u8 samples to linear f32, one for one. Alpha, if you pass it, is
/// treated as a colour channel; decode RGB and alpha separately if that matters.
#[wasm_bindgen]
pub fn srgb_to_linear(src: &[u8]) -> Vec<f32> {
    halflight::decode_srgb(src)
}

/// Linear f32 to sRGB u8, clamped.
#[wasm_bindgen]
pub fn linear_to_srgb(src: &[f32]) -> Vec<u8> {
    halflight::encode_srgb(src)
}
