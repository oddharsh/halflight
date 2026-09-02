//! The README's opening measurement, committed as a test so it cannot rot.
//! A 1px checkerboard must reduce to sRGB 187.5. The incumbents' default
//! paths read ~127.5, because they average encoded values.

use conformance::*;
use halflight::Filter;

const N: usize = 1024;
const OUT: usize = 64;
const IDEAL: f64 = 187.5;

#[test]
fn halflight_reads_half_the_light() {
    let src = checkerboard(N, 3);
    for f in [Filter::Box, Filter::Lanczos3, Filter::Mitchell] {
        let out = halflight_u8(&src, N, N, 3, OUT, OUT, f);
        let m = interior_mean(&out, OUT, OUT, 3, 2);
        eprintln!("halflight {f:?}: {m:.1}");
        assert!(
            (m - IDEAL).abs() < 1.5,
            "halflight {f:?}: {m:.1}, expected ~{IDEAL}"
        );
    }
}

#[test]
fn fir_mapped_path_agrees_once_you_opt_in() {
    let src = checkerboard(N, 3);
    let out = fir_u8_srgb_mapped(&src, N, N, OUT, OUT, Filter::Box);
    let m = interior_mean(&out, OUT, OUT, 3, 2);
    eprintln!("fast_image_resize srgb mapper: {m:.1}");
    assert!((m - IDEAL).abs() < 1.5, "fir mapped: {m:.1}, expected ~{IDEAL}");
}

/// The incumbents' DEFAULT paths. Asserted as wrong rather than merely
/// printed, so the day one of them changes its default this test says so and
/// the README's opener gets rewritten instead of going stale.
#[test]
fn the_default_paths_average_encoded_values() {
    let src = checkerboard(N, 3);
    let img = interior_mean(&image_u8(&src, N, N, OUT, OUT, Filter::Lanczos3), OUT, OUT, 3, 2);
    let fir = interior_mean(&fir_u8(&src, N, N, OUT, OUT, Filter::Box), OUT, OUT, 3, 2);
    eprintln!("default paths: image {img:.1}  fast_image_resize {fir:.1}  (correct {IDEAL})");
    assert!(
        (img - 127.5).abs() < 2.0,
        "image's default path read {img:.1}; if it moved toward {IDEAL} the README is out of date"
    );
    assert!((fir - 127.5).abs() < 2.0, "fast_image_resize's default path read {fir:.1}; if it moved toward {IDEAL} the README is out of date");
}
