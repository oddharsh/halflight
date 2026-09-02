#![allow(clippy::type_complexity, clippy::needless_range_loop)]
//! The bench driver. One implementation × one shape per child process, its own
//! warm-up, best of N timed runs. Shared-process harnesses let warm-up, cache
//! state and allocator behaviour leak between implementations; a child per pair
//! is the cheap way to not have that argument.
//!
//!   conformance bench                 # the matrix, prints a markdown table
//!   conformance bench --runs 7
//!   conformance run <impl> <shape> <filter> [--runs N]   # one cell, in-process
//!   conformance checkerboard          # the README opener, every path
//!   conformance corpus <dir>          # real photos: mean ms per impl over the dir

use conformance::*;
use halflight::{srgb_to_linear, Filter};
use std::process::Command;
use std::time::Instant;

const IMPLS: &[&str] = &[
    "halflight/u8",  // decode LUT + kernel + encode: the whole job
    "halflight/f32", // kernel only, linear f32 in hand
    "fir/u8-mapped", // fir's correct path: srgb mapper, u16 resize
    "fir/f32",       // fir kernel only on F32x3
    "fir/u8",        // fir default: WRONG colour, fastest
    "image/f32",     // image kernel only on Rgb32F
    "image/u8",      // image default: WRONG colour
];
// (label, sw, sh, dw, dh). The first is a 24 MP camera frame to a 600px tile,
// the geometry this crate was written for. The second is fast_image_resize's
// own benchmark geometry, so its README table can be read against this one.
const SHAPES: &[(&str, usize, usize, usize, usize)] = &[
    ("24MP->900x600", 5952, 3968, 900, 600),
    ("fir-geometry 4928x3279->852x567", 4928, 3279, 852, 567),
];
const FILTERS: &[(&str, Filter)] = &[("box", Filter::Box), ("lanczos3", Filter::Lanczos3)];

fn parse_filter(s: &str) -> Filter {
    match s {
        "box" => Filter::Box,
        "lanczos3" => Filter::Lanczos3,
        "mitchell" => Filter::Mitchell,
        _ => panic!("filter {s}"),
    }
}

fn time_one(imp: &str, sw: usize, sh: usize, dw: usize, dh: usize, f: Filter, runs: usize) -> f64 {
    let src = synthetic_rgb(sw, sh);
    let lin: Vec<f32> = src.iter().map(|&c| srgb_to_linear(c)).collect();
    let go = || -> usize {
        match imp {
            "halflight/u8" => halflight_u8(&src, sw, sh, 3, dw, dh, f).len(),
            "halflight/f32" => halflight_f32(&lin, sw, sh, 3, dw, dh, f).len(),
            "fir/u8-mapped" => fir_u8_srgb_mapped(&src, sw, sh, dw, dh, f).len(),
            "fir/f32" => fir_f32(&lin, sw, sh, dw, dh, f).len(),
            "fir/u8" => fir_u8(&src, sw, sh, dw, dh, f).len(),
            "image/f32" => image_f32(&lin, sw, sh, dw, dh, f).len(),
            "image/u8" => image_u8(&src, sw, sh, dw, dh, f).len(),
            _ => panic!("impl {imp}"),
        }
    };
    go();
    go(); // warm-up: page in, populate LUTs, let the allocator settle
    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let t = Instant::now();
        let n = go();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        assert!(n > 0);
        if ms < best {
            best = ms;
        }
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let runs = args
        .iter()
        .position(|a| a == "--runs")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(5usize);
    match args.get(1).map(String::as_str) {
        Some("run") => {
            let imp = &args[2];
            let shape = SHAPES.iter().find(|s| s.0 == args[3]).expect("shape");
            let f = parse_filter(&args[4]);
            println!(
                "{:.3}",
                time_one(imp, shape.1, shape.2, shape.3, shape.4, f, runs)
            );
        }
        Some("bench") => {
            let exe = std::env::current_exe().unwrap();
            for (label, sw, sh, dw, dh) in SHAPES {
                println!("\n### {label}  (best of {runs}, one process per cell, ms)\n");
                println!(
                    "| implementation | {} |",
                    FILTERS.iter().map(|f| f.0).collect::<Vec<_>>().join(" | ")
                );
                println!("|---|{}", "---:|".repeat(FILTERS.len()));
                for imp in IMPLS {
                    let mut cells = vec![];
                    for (fname, _) in FILTERS {
                        let out = Command::new(&exe)
                            .args(["run", imp, label, fname, "--runs", &runs.to_string()])
                            .output()
                            .unwrap();
                        let ms: f64 = String::from_utf8_lossy(&out.stdout)
                            .trim()
                            .parse()
                            .unwrap_or(f64::NAN);
                        cells.push(format!("{ms:.1}"));
                    }
                    println!("| `{imp}` | {} |", cells.join(" | "));
                }
                let _ = (sw, sh, dw, dh);
            }
        }
        Some("checkerboard") => {
            let n = 1024;
            let o = 64;
            let src = checkerboard(n, 3);
            let m = |v: Vec<u8>| interior_mean(&v, o, o, 3, 2);
            println!("1px checkerboard {n}x{n} -> {o}x{o}, interior mean (correct: 187.5)\n");
            println!("| path | reads |\n|---|---:|");
            println!(
                "| `halflight` (Box) | {:.1} |",
                m(halflight_u8(&src, n, n, 3, o, o, Filter::Box))
            );
            println!(
                "| `fast_image_resize` srgb mapper, U16x3 (opt-in) | {:.1} |",
                m(fir_u8_srgb_mapped(&src, n, n, o, o, Filter::Box))
            );
            println!(
                "| `fast_image_resize` default U8x3 | {:.1} |",
                m(fir_u8(&src, n, n, o, o, Filter::Box))
            );
            println!(
                "| `image` default, Lanczos3 | {:.1} |",
                m(image_u8(&src, n, n, o, o, Filter::Lanczos3))
            );
        }
        Some("corpus") => {
            let dir = &args[2];
            let mut files: Vec<_> = std::fs::read_dir(dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    matches!(
                        p.extension()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_ascii_lowercase())
                            .as_deref(),
                        Some("jpg" | "jpeg" | "png")
                    )
                })
                .collect();
            files.sort();
            println!("corpus: {} files in {dir}\n", files.len());
            let mut tot = std::collections::BTreeMap::<&str, (f64, usize)>::new();
            for p in &files {
                let img = match image::open(p) {
                    Ok(i) => i.to_rgb8(),
                    Err(_) => continue,
                };
                let (sw, sh) = (img.width() as usize, img.height() as usize);
                let (dw, dh) = if sw >= sh {
                    (900, (sh * 900 + sw / 2) / sw)
                } else {
                    ((sw * 600 + sh / 2) / sh, 600)
                };
                let src = img.into_raw();
                let lin: Vec<f32> = src.iter().map(|&c| srgb_to_linear(c)).collect();
                let cases: Vec<(&str, Box<dyn Fn() -> usize>)> = vec![
                    (
                        "halflight/u8",
                        Box::new(|| halflight_u8(&src, sw, sh, 3, dw, dh, Filter::Box).len()),
                    ),
                    (
                        "fir/u8-mapped",
                        Box::new(|| fir_u8_srgb_mapped(&src, sw, sh, dw, dh, Filter::Box).len()),
                    ),
                    (
                        "fir/u8",
                        Box::new(|| fir_u8(&src, sw, sh, dw, dh, Filter::Box).len()),
                    ),
                    (
                        "image/u8 (lanczos3)",
                        Box::new(|| image_u8(&src, sw, sh, dw, dh, Filter::Lanczos3).len()),
                    ),
                ];
                let _ = &lin;
                for (name, go) in cases {
                    go();
                    let mut best = f64::INFINITY;
                    for _ in 0..runs {
                        let t = Instant::now();
                        go();
                        best = best.min(t.elapsed().as_secs_f64() * 1e3);
                    }
                    let e = tot.entry(name).or_insert((0.0, 0));
                    e.0 += best;
                    e.1 += 1;
                }
            }
            println!("| implementation | mean best-of-{runs} ms | files |\n|---|---:|---:|");
            for (name, (ms, n)) in tot {
                println!("| `{name}` | {:.1} | {n} |", ms / n as f64);
            }
        }
        _ => eprintln!("usage: conformance bench|run|checkerboard|corpus"),
    }
}
