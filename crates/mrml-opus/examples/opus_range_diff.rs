//! Black-box final-range differential minimizer for Opus packets.
//!
//! This diagnostic deliberately observes only the public libopus decoder API;
//! it is not part of `mrml-opus` and contributes no implementation code.

#[cfg(target_os = "linux")]
mod linux {
    use core::ffi::{c_int, c_uchar, c_void};

    use mrml_opus::{Decoder, Packet};

    const OPUS_OK: c_int = 0;
    const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;

    #[link(name = "opus")]
    unsafe extern "C" {
        fn opus_decoder_create(sample_rate: c_int, channels: c_int, error: *mut c_int)
        -> *mut c_void;
        fn opus_decoder_destroy(decoder: *mut c_void);
        fn opus_decode(
            decoder: *mut c_void,
            data: *const c_uchar,
            length: c_int,
            pcm: *mut i16,
            frame_size: c_int,
            decode_fec: c_int,
        ) -> c_int;
        #[link_name = "opus_decoder_ctl"]
        fn opus_decoder_get_u32(
            decoder: *mut c_void,
            request: c_int,
            value: *mut u32,
        ) -> c_int;
    }

    fn decode_reference(packet: &[u8]) -> Option<u32> {
        let mut error = 0;
        // SAFETY: The returned decoder is checked and destroyed in this scope.
        let decoder = unsafe { opus_decoder_create(48_000, 2, &mut error) };
        if decoder.is_null() || error != OPUS_OK {
            return None;
        }
        let mut pcm = [0i16; 11_520];
        // SAFETY: All pointers reference live buffers of the declared lengths.
        let decoded = unsafe {
            opus_decode(
                decoder,
                packet.as_ptr(),
                c_int::try_from(packet.len()).ok()?,
                pcm.as_mut_ptr(),
                5_760,
                0,
            )
        };
        let mut range = 0;
        let status = if decoded >= 0 {
            // SAFETY: The request writes one u32 to the supplied live pointer.
            unsafe { opus_decoder_get_u32(decoder, OPUS_GET_FINAL_RANGE_REQUEST, &mut range) }
        } else {
            decoded
        };
        // SAFETY: `decoder` was returned by `opus_decoder_create` above.
        unsafe { opus_decoder_destroy(decoder) };
        (status == OPUS_OK).then_some(range)
    }

    fn decode_mrml(packet: &[u8]) -> Option<u32> {
        let parsed = Packet::parse(packet).ok()?;
        let samples = 48_000usize
            .checked_mul(parsed.frame_duration_us as usize)?
            .checked_div(1_000_000)?
            .checked_mul(parsed.frame_count as usize)?;
        let mut decoder = Decoder::new(2).ok()?;
        let mut pcm = [0i16; 11_520];
        decoder.decode(packet, &mut pcm[..samples * 2], 48_000).ok()?;
        Some(decoder.final_range())
    }

    fn mismatch(packet: &[u8]) -> Option<(u32, u32)> {
        let reference = decode_reference(packet)?;
        let mrml = decode_mrml(packet)?;
        (reference != mrml).then_some((reference, mrml))
    }

    fn minimize(mut packet: Vec<u8>, verbose: bool) -> Option<(Vec<u8>, u32, u32)> {
        let (reference, mrml) = mismatch(&packet)?;
        print_packet(&packet, reference, mrml);

        let mut changed = true;
        while changed {
            changed = false;
            for index in 1..packet.len() {
                let original = packet[index];
                if original == 0 {
                    continue;
                }
                packet[index] = 0;
                if mismatch(&packet).is_some() {
                    changed = true;
                    if verbose {
                        println!("keep_zero index={index}");
                    }
                } else {
                    packet[index] = original;
                }
            }
        }
        for length in 2..packet.len() {
            if mismatch(&packet[..length]).is_some_and(|(reference, mrml)| {
                reference != 0 && mrml != 0
            }) {
                if verbose {
                    println!("truncate length={length}");
                }
                packet.truncate(length);
                break;
            }
        }
        let (reference, mrml) = mismatch(&packet)?;
        Some((packet, reference, mrml))
    }

    fn parse_hex(text: &str) -> Result<Vec<u8>, String> {
        if !text.len().is_multiple_of(2) {
            return Err("hex input must contain complete bytes".to_owned());
        }
        (0..text.len())
            .step_by(2)
            .map(|offset| {
                u8::from_str_radix(&text[offset..offset + 2], 16)
                    .map_err(|_| format!("invalid hex at character {offset}"))
            })
            .collect()
    }

    fn print_packet(packet: &[u8], reference: u32, mrml: u32) {
        let hex = packet.iter().fold(String::new(), |mut output, byte| {
            use core::fmt::Write;
            let _ = write!(output, "{byte:02X}");
            output
        });
        println!(
            "bytes={} reference={reference:#010x} mrml={mrml:#010x} packet={hex}",
            packet.len()
        );
    }

    pub fn main() -> Result<(), String> {
        let mut verbose = false;
        let mut packet_hex = None;
        for argument in std::env::args().skip(1) {
            if argument == "--verbose" {
                verbose = true;
            } else if argument == "--help" || argument == "-h" {
                return Ok(println!("usage: opus_range_diff <packet-hex> [--verbose]"));
            } else if packet_hex.is_none() {
                packet_hex = Some(argument);
            } else {
                return Err(format!("unexpected argument: {argument}"));
            }
        }
        let input = packet_hex.ok_or_else(|| {
            "usage: opus_range_diff <packet-hex> [--verbose]".to_owned()
        })?;
        let packet = parse_hex(&input)?;
        let (minimal, reference, mrml) = minimize(packet, verbose)
            .ok_or_else(|| "packet is rejected by a decoder or final ranges already match".to_owned())?;
        print_packet(&minimal, reference, mrml);
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), String> {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() -> Result<(), String> {
    Err("opus_range_diff requires Linux and the system libopus development library".to_owned())
}
