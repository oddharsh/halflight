use halflight::{g22_lut16, srgb_lut16};

/// This integration-test binary has one test, so neither global table can be
/// warmed by another test before the small-stack initialization probe runs.
#[test]
fn u16_tables_initialize_on_a_small_stack_and_preserve_every_sample() {
    std::thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn(|| {
            let srgb = srgb_lut16();
            let g22 = g22_lut16();
            assert_eq!(srgb.len(), 65536);
            assert_eq!(g22.len(), 65536);

            // Preserve zenc's original scalar formulas as an independent
            // compatibility oracle. Check all codes, including the sRGB toe.
            for c in 0..=u16::MAX {
                let s = c as f32 / 65535.0;
                let expected_srgb = if s <= 0.040_449_936 {
                    s / 12.92
                } else {
                    ((s + 0.055) / 1.055).powf(2.4)
                };
                assert_eq!(srgb[c as usize].to_bits(), expected_srgb.to_bits(), "sRGB {c}");
                assert_eq!(g22[c as usize].to_bits(), s.powf(2.2).to_bits(), "gamma 2.2 {c}");
            }
        })
        .expect("start small-stack thread")
        .join()
        .expect("initialize both tables and compare their values");
}
