# Development

```bash
cargo test -p halflight                    # the crate: 11 tests + the doctest
cargo test -p conformance -- --nocapture   # gates, with the numbers printed
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check

cargo run --release -p conformance -- checkerboard        # the README opener, every path
cargo run --release -p conformance -- bench --runs 5      # the matrix, one process per cell
cargo run --release -p conformance -- corpus <dir>        # real photos (jpg/png)

cd packages/halflight && npm run build && npm test         # wasm + the JS surface
```

## Layout

| path | what |
|---|---|
| `crates/halflight` | the crate. Zero dependencies. Kernel, sRGB and gamma-2.2 transfer, tests. |
| `crates/halflight-wasm` | the `wasm-bindgen` surface for the npm package. Never published to crates.io. |
| `packages/halflight` | the npm package: a Node-aware loader over the wasm-pack output. |
| `conformance` | never published. The incumbents' adapters, the checkerboard and parity gates, the bench driver. |

## Rules that are load-bearing

- **The kernel's accumulation order is the contract.** `resample_fixed` and
  `resample_dyn` must stay bitwise equal, and the test asserts bits, not an
  epsilon. A SIMD rewrite has to preserve per-channel ascending summation or
  re-mint every content-addressed output anyone has built on this.
- **The transfer tables are not a second opinion about sRGB.** The LUT is
  built from the exact expression it replaces and a test asserts they agree
  bitwise at all 256 values. Any faster encode must keep the 8-bit round trip
  exact, or it is a different colour space wearing sRGB's name.
- **The checkerboard test asserts the incumbents are wrong.** If `image` or
  `fast_image_resize` change their default to a linear-light path, that test
  goes red on purpose: the README's opening claim has to be rewritten, not
  left to rot.

## Toolchain

`rust-toolchain.toml` pins the development toolchain, and CI reads the pin
out of that file so it is declared once. `clippy -D warnings` is a gate
against that fixed lint set; a `stable` that moves under it is how the seed's
first CI run went red with no code change. MSRV is 1.75 and is checked on its
own toolchain in CI.

`Cargo.lock` is lockfile version 3 and must stay there: the MSRV job's cargo
1.75 cannot read version 4, which newer cargo writes for a fresh lock. cargo
keeps an existing older version rather than bumping it, so this only matters
if the file is ever deleted and regenerated.

The wasm build wants `wasm32-unknown-unknown`, `wasm-pack`, and `wasm-opt`
(binaryen) on the path.
