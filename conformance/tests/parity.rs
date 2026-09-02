//! Fed identical linear f32, the kernel must agree with both incumbents'
//! kernels. This is the claim that halflight is not a different resampler,
//! only one whose input type refuses the wrong colour path.

use conformance::*;
use halflight::Filter;

fn linear_field(w: usize, h: usize) -> Vec<f32> {
    synthetic_rgb(w, h)
        .iter()
        .map(|&c| halflight::srgb_to_linear(c))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn lanczos3_matches_image_and_fir_kernels_on_linear_f32() {
    let (sw, sh, dw, dh) = (640, 427, 200, 133);
    let src = linear_field(sw, sh);
    let ours = halflight_f32(&src, sw, sh, 3, dw, dh, Filter::Lanczos3);
    let img = image_f32(&src, sw, sh, dw, dh, Filter::Lanczos3);
    let fir = fir_f32(&src, sw, sh, dw, dh, Filter::Lanczos3);
    let d_img = max_abs_diff(&ours, &img);
    let d_fir = max_abs_diff(&ours, &fir);
    eprintln!("Lanczos3 on linear f32, max |diff| vs image {d_img:.2e}, vs fast_image_resize {d_fir:.2e}");
    assert!(d_img < 1e-4, "kernel diverges from image's by {d_img}");
    assert!(
        d_fir < 1e-4,
        "kernel diverges from fast_image_resize's by {d_fir}"
    );
}

#[test]
fn box_matches_fir_on_linear_f32() {
    let (sw, sh, dw, dh) = (640, 427, 160, 107);
    let src = linear_field(sw, sh);
    let ours = halflight_f32(&src, sw, sh, 3, dw, dh, Filter::Box);
    let fir = fir_f32(&src, sw, sh, dw, dh, Filter::Box);
    let d = max_abs_diff(&ours, &fir);
    eprintln!("Box on linear f32, max |diff| vs fast_image_resize {d:.2e}");
    assert!(d < 1e-4, "Box diverges from fast_image_resize's by {d}");
}
