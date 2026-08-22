# mrml-opus

Original, Rust-only, allocation-free RFC 6716 Opus work for MRML. Packet
framing, standards-compliant DTX, stateful mono/stereo SILK decoding, and the
CELT entropy, allocation, PVQ, energy, anti-collapse, transform, and synthesis
paths are implemented. The public decoder accepts SILK, CELT, and hybrid
packets, including their shared hybrid range stream. The SILK path includes
multi-interval frames, asymmetric in-band LBRR/FEC parsing, side suppression,
concealment, channel conversion, and output at each Opus API sample rate. The
public decoder exposes one-transition regular-plus-LBRR decoding and stateful
loss concealment for SILK, CELT, and hybrid frames. It also exposes the final
constituent frame's entropy range for `OPUS_GET_FINAL_RANGE` interoperability
checks; DTX and packet-loss concealment report zero.

A public PCM encoder emits 10/20/40/60 ms mono or stereo narrowband,
medium-band, and wideband SILK packets; 2.5/5/10/20 ms mono or stereo
narrowband, wideband, superwideband, and fullband CELT packets; and 20 ms
Hybrid packets. CELT analysis supports long and
detected-transient short blocks, theta-coupled joint/intensity stereo, and
direct-MDCT conformance baselines. Automatic per-channel bitrate thresholds
select SILK, hybrid, or CELT. Explicit mono and stereo SILK encoding
continuously controls 20 ms packet sizes across the tested 6 to 80 kbit/s
range, mono hybrid encoding
from 116 kbit/s, mono CELT from 48 kbit/s, and stereo CELT from 80 kbit/s.
Representative packets from every encoded SILK and CELT bandwidth have also
been accepted by an independent system libopus decoder. Explicit and automatic low-rate CELT and hybrid packet generation now honor
the requested packet budget and round-trip through this decoder; perceptual
analysis and independent interoperability coverage at those rates remain incomplete. Hybrid packets signal
redundancy absence, while Hybrid and SILK-only decoding byte-isolates and
power-complementarily cross-laps embedded 5 ms CELT transition frames with
mode-transition state resets. Hybrid encoding can embed a caller-supplied,
TOC-less 5 ms CELT redundancy frame with normative signaling and rate-budget
separation. Automatic CELT-to-Hybrid beginning redundancy is emitted; reverse or
delayed transitions, SILK-only redundancy emission, and independent
interoperability-vector coverage remain incomplete. The crate
never labels a private audio format as Opus.
