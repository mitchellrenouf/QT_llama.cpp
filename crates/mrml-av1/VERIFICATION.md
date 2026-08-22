# AV1 contribution verification

The codec is incomplete; this evidence covers only the code currently present.
It must not be used as evidence of AV1 bitstream conformance.

## 2026-08-22 Windows

Host: `x86_64-pc-windows-gnullvm`, Rust nightly 1.100.0

```powershell
rustup run nightly-x86_64-pc-windows-gnullvm cargo test -p mrml-av1 --release --no-default-features --locked
rustup run nightly-x86_64-pc-windows-gnullvm cargo test -p mrml-av1 --release --features nvidia --locked
rustup run nightly-x86_64-pc-windows-gnullvm cargo clippy -p mrml-av1 --all-targets --all-features --locked --no-deps -- -D warnings
```

Results: 265 CPU tests passed; 260 NVIDIA-feature tests passed in the preceding
run; crate-local Clippy passed with warnings denied after the latest CPU
changes. No MSVC, Visual Studio, Windows SDK, CUDA
Toolkit, or external LLVM was used.

## 2026-08-22 Linux

Host: WSL2 `x86_64-unknown-linux-gnu`, Rust nightly 1.100.0

```bash
cargo test -p mrml-av1 --release --no-default-features --locked
cargo test -p mrml-av1 --release --features nvidia --locked
cargo clippy -p mrml-av1 --all-targets --all-features --locked --no-deps -- -D warnings
```

Results: 265 CPU tests passed after the latest CPU changes; 260
NVIDIA-feature tests and crate-local Clippy passed in the preceding run.

## Authoritative AOM vectors

The `decode_ivf` example was run against files from AOMedia's official
`aom-test-data` bucket, with SHA-1 values checked against libaom's
`test/test-data.sha1` manifest.

For independent black-box comparison, AOMedia Project AV1 Decoder 3.14.1 was
run in WSL without linking it into MRML. Its first two 384-byte planar I420
outputs have MD5 `6353b245c305a5f4f2845ee7ad2b128b` and
`f4b0078dfbc8b581fa959d4512b9940a`; MRML now matches both files byte for byte.
The first q7 output is 152064 bytes with MD5
`fbd569613f9a2e52075566dce8e2af6d`. An isolated accounting/inspection build under ignored `target/`
confirms that the 16x16 first frame has two non-skipped `BLOCK_16X8` leaves,
six plane transforms, `TX_16X8` luma transforms, and `DCT_DCT` transform type.
The first leaf uses luma `SMOOTH_H_PRED` and chroma `UV_D45_PRED`; the second
uses luma and chroma DC. MRML now agrees through syntax and reconstruction for
both frames. No reference implementation source or library is incorporated
into the crate.

- `av1-1-b8-01-size-16x16.ivf` (`838388fb...`): both packets decode, pass
  strict tile termination, and produce byte-exact reference frames. The final
  reconstruction discrepancy was an intra-DC availability error: a lower
  chroma transform averaged a synthesized left edge even though only its above
  edge was available.
- `av1-1-b8-00-quantizer-07.ivf` (`7f8113cd...`): its complete first frame now
  decodes after using the filter-intra directional equivalent to select the
  intra transform-type CDF. Correcting warped-prediction phase reduction then
  advances the second frame through its first affine global-motion block.
  Clipping variable-transform lookup and inter prediction at the cropped right
  edge now advances that frame through 211 blocks and 734 transform blocks
  before the next tile-entropy divergence, so this remains a failing
  conformance target.

The remaining q7 failure is explicit evidence that decoder conformance is not
complete.
The run exposed and fixed a coefficient escape error: the normative Golomb
value is one-based, and `coeff_base_eob` likewise contributes its zero-based
symbol plus one. It also exposed an EOB-CDF selection error that used the
configured transform type instead of the decoded transform type. Subsequent
runs exposed and fixed cropped-frame transform
footprint handling, inter-neighbor transform contexts, and an unbounded luma
transform-block-skip context. A syntax audit additionally fixed `ReadDeltas`:
the first leaf clears it, and only a skipped leaf equal to the complete
superblock suppresses delta-Q/LF syntax; skipped children still consume it as
required by sections 5.11.7, 5.11.12, 5.11.13, and 5.11.18. These two vectors
do not signal delta-Q and therefore retain the failure signatures above.
The same syntax audit fixed skipped intra blocks incorrectly reading a
transform-depth symbol even though section 5.11.16 gates that syntax on
`!skip`; a dedicated regression test covers the corrected CDF-preserving path.
Primary-reference CDF loading now also resets the adaptation-count entry of
every non-coefficient, motion-vector, and coefficient CDF while preserving its
probability thresholds, as required by section 7.4. The present vector failures
do not select a saved primary context, so their signatures remain unchanged.
Coefficient decoding now also retains the single CDF family selected by
`init_coeff_cdfs` for the whole frame and across primary-reference loads rather
than selecting among default banks at each block. A regression test verifies
that a later qindex argument cannot switch the active adaptive family.
Subsampled coefficient-neighbor storage now ceiling-divides odd luma MI
dimensions, so the final chroma row and column remain addressable as required
by AV1 plane geometry. A focused odd-edge regression covers this boundary; the
coefficient spans themselves now clip at cropped frame edges rather than
rejecting the final transform. Escaped levels are also reduced to the
normative 20-bit magnitude before DC-category and cumulative-context updates.
Strict termination errors retain leaf, skip, transform, nonzero-transform, and
EOB-position counts to localize subsequent conformance failures.

## Security review scope

- Portable library source contains no direct `std` or `alloc` imports; the
  command-line conformance example uses `std` for file I/O only.
- Dependencies are restricted to `core` and original workspace crates.
- The portable codec contains no unsafe code.
- OBU, IVF, Annex-B, metadata, tile, entropy, frame-parameter, global-motion,
  film-grain, arithmetic-coded partition traversal, mode contexts, prediction,
  transform, coefficient-level, deblocking, CDEF, super-resolution, and
  reconstruction inputs use checked bounds, precision ranges, and allocation
  boundaries.
- NVDEC/NVENC discovery is isolated behind the `nvidia` feature and uses the
  existing platform dynamic-library boundary. It does not yet create sessions
  or accept device pointers.
- Fuzzing and passing official AOM conformance-vector runs remain required
  before any decoder-conformance claim.

No performance claim is recorded while authoritative conformance vectors still
fail.
