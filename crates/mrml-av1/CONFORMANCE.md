# AV1 conformance ledger

This crate is a clean-room, CPU-only implementation. `cargo check -p mrml-av1
--all-targets` is the minimum gate. A feature is only marked complete after
positive vectors, malformed-input tests, and comparison with AOM conformance
vectors have passed.

| AV1 specification area | Decode | Encode | Notes |
| --- | --- | --- | --- |
| 5.2 low-overhead bitstream | implemented, vectors pending | partial | Mandatory internal OBU sizes, strict length boundaries, and reusable encoder stream assembly; unsized OBUs accepted only through Annex B outer lengths |
| 5.3 OBU header | implemented, vectors pending | partial | Extension validation, canonical generation, reserved-OBU skipping, and reserved payload validity checks implemented |
| 5.8 metadata OBU | implemented, vectors pending | pending | Typed HDR CLL/MDCV, ITU-T T.35 envelope, scalability structures/temporal dependencies, and bit-accurate timecode; unknown types retained |
| 5.4 sequence header | implemented, vectors pending | partial | Timing, selectable operating points with temporal/spatial filtering and extension-flag invariants, nonzero decoder tick validation, decoder model, color/tool flags, and reduced/non-reduced encoder generation |
| IVF transport (non-normative) | complete | complete | Strict packet count and bounds |
| 5.9 uncompressed frame header | partial | pending | Major syntax, frame-ID marking, short reference signaling, identical ordinary/redundant header-copy state, decoder-model timestamps, and film grain parsed; further conformance vectors remain |
| 5.10 tile info / 5.11 tile groups | implemented, vectors pending | pending | Uniform/non-uniform layouts, strict group bounds, independent per-tile CDF initialization, live block-CDF access, designated context-update-tile retention, and end-to-end tile traversal |
| 5.12 tile list OBU | implemented, vectors pending | n/a | Strict 512-entry/128-anchor limits, exact coded-tile bounds, external anchor API, anchor-format checks, large-scale-tile constraints, per-entry LAST anchor substitution, camera-tile reconstruction, raster output assembly, and no reference update |
| 5.11 quantization | implemented, vectors pending | pending | Lossless state derived per segment |
| 5.12 segmentation | implemented, vectors pending | pending | All eight features and inherited state parsed |
| 5.13 delta parameters | implemented, vectors pending | pending | Delta-Q and delta loop-filter syntax |
| 5.14 loop filter parameters | implemented, vectors pending | pending | Includes reference/mode deltas and inherited state |
| 5.15 CDEF parameters | implemented, vectors pending | pending | |
| 5.16 loop restoration parameters | implemented, vectors pending | pending | |
| 5.17 TX mode / reference mode | implemented, vectors pending | pending | |
| 5.18 skip mode / global motion | partial | pending | Skip derivation and global parameters parsed; conformance vectors pending |
| 5.19 film grain parameters | implemented, vectors pending | pending | Includes reference inheritance, normative Gaussian/AR pattern generation, stripe overlap, scaling lookup, output clipping, and output-only synthesis |
| 5.9.15 tile group syntax | partial, vectors pending | pending | Ordered multi-OBU tile-group accumulation with gap/overlap/duplicate rejection and allocated pending-frame grid/reconstruction planes; arithmetic-coded recursive traversal and normative initial CDF state for implemented symbols; per-MI block/reference/MV/palette state; common block prefix, reference decisions, intra/inter mode syntax, palette maps, quantization matrices, motion-vector stacks, adaptive transform syntax, coefficient token decoding, dequantization, prediction, inverse transforms, residual addition, and context updates are connected in the concrete block/tile loop; independent-vector coverage remains |
| 8.2 symbol coding process | decoder implemented, vectors pending | encoder groundwork | Decoder range normalization, literals, NS, CDF adaptation, and trailing validation; inverse adaptive range encoder round-trips symbols/literals/NS with identical CDF state and emits byte-aligned tile termination accepted by the strict decoder |
| 7.11 prediction | partial, vectors pending | n/a | Intra families, edge filtering, CfL, decoded palette-map plane materialization, motion scaling, compound blending, separable inter convolution, normative sub-pel and warped-filter banks, affine warp estimation, motion-mode contexts, SIMPLE/local/global dispatch, and OBMC neighbor traversal are connected to block reconstruction; independent-vector coverage remains |
| 7.12 reconstruction | partial | n/a | Checked 8/10/12-bit planes, residual placement, clipping, and public frame conversion |
| 7.13 inverse transforms | implemented, vectors pending | n/a | DCT, ADST, identity, WHT, 2D transform process, transform-type components, and normative transform-set selection/inversion |
| 7.14 deblocking | partial, vectors pending | n/a | Normative edge masks, level derivation, narrow/8-tap/16-tap filters, and frame-edge traversal are integrated |
| 7.15 CDEF | partial, vectors pending | n/a | Normative direction/variance detection, constraint, directional block filtering, region availability, output clipping, and per-tile frame traversal are integrated |
| 8 decoding process | integrated, vectors pending | n/a | Tile reconstruction flows through deblocking, CDEF, super-resolution, restoration, output-only film grain, reference refresh, saved CDF/grid state, and display gating |
| Annex A profiles and levels | partial | partial | Profile/color validation implemented |
| Annex B length-delimited format | complete | complete | Strict nested temporal/frame/OBU bounds |
| Annex C decoder model | partial | n/a | Sequence timing/model parameters, frame-presentation times, and per-operating-point buffer-removal times are parsed and retained on displayed frames; fullness/conformance simulation remains |

## Optional acceleration backends

| Backend | Status | Build behavior |
| --- | --- | --- |
| CPU decoder | in progress | Always available; normative fallback |
| NVDEC AV1 | loader/bootstrap complete, session ABI pending | `nvidia` feature; runtime-loaded `nvcuvid` and CUDA driver with real `cuInit(0)` availability validation |
| CPU encoder | syntax groundwork | Always available; sequence/frame headers, uniform tiling, OBU assembly, standardized metadata, scalability dependencies, adaptive arithmetic symbol/literal/NS encoding, and validated tile termination; coded-block policy remains pending |
| NVENC AV1 | loader/bootstrap complete, session ABI pending | `nvidia` feature; runtime-loaded NVENCODE API with maximum driver API-version negotiation |

## Completion definition

“Entire AV1 spec supported” means all normative bitstream syntax accepted or
rejected exactly as specified, 8/10/12-bit profile support, 4:0:0/4:2:0/4:2:2/
4:4:4 output, spatial and temporal layers, tiles, super-resolution, all intra
and inter prediction modes, transforms, loop filters, restoration, film grain,
metadata, reference-frame state, error bounds, and passing the official AOM
conformance suite. Encoder completion additionally requires legal syntax for
every coding tool; compression-quality optimization is a separate milestone.
