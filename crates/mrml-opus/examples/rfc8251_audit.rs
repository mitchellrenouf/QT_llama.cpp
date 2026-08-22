use std::path::{Path, PathBuf};

use mrml_opus::{Decoder, Packet};

#[derive(Clone, Copy, Default)]
struct Totals {
    packets: usize,
    decoded: usize,
    exact_ranges: usize,
}

fn audit(path: &Path) -> Result<Totals, String> {
    let data = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut offset = 0usize;
    let mut totals = Totals::default();
    let mut decoder = Decoder::new(2).map_err(|error| format!("decoder: {error}"))?;
    let mut pcm = [0i16; 11_520];
    while offset < data.len() {
        if data.len() - offset < 8 {
            return Err(format!(
                "{}: truncated record header at byte {offset}",
                path.display()
            ));
        }
        let length = u32::from_be_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| "record length".to_owned())?,
        ) as usize;
        let expected_range = u32::from_be_bytes(
            data[offset + 4..offset + 8]
                .try_into()
                .map_err(|_| "final range".to_owned())?,
        );
        let packet_start = offset + 8;
        let packet_end = packet_start
            .checked_add(length)
            .ok_or_else(|| "record length overflow".to_owned())?;
        let packet = data
            .get(packet_start..packet_end)
            .ok_or_else(|| format!("{}: truncated packet", path.display()))?;
        totals.packets += 1;
        if let Ok(parsed) = Packet::parse(packet) {
            let frames = 48_000usize
                .checked_mul(parsed.frame_duration_us as usize)
                .and_then(|value| value.checked_div(1_000_000))
                .and_then(|value| value.checked_mul(parsed.frame_count as usize))
                .ok_or_else(|| "decoded frame count overflow".to_owned())?;
            if decoder
                .decode(packet, &mut pcm[..frames * 2], 48_000)
                .is_ok()
            {
                totals.decoded += 1;
                totals.exact_ranges += usize::from(decoder.final_range() == expected_range);
            }
        }
        offset = packet_end;
    }
    Ok(totals)
}

fn main() -> Result<(), String> {
    let mut arguments = std::env::args_os().skip(1);
    let directory =
        PathBuf::from(arguments.next().ok_or_else(|| {
            "usage: rfc8251_audit <vector-directory> [--require-exact]".to_owned()
        })?);
    let require_exact = arguments.any(|argument| argument == "--require-exact");
    let mut all = Totals::default();
    for number in 1..=12 {
        let path = directory.join(format!("testvector{number:02}.bit"));
        let totals = audit(&path)?;
        println!(
            "{number:02}: packets={} decoded={} exact_ranges={}",
            totals.packets, totals.decoded, totals.exact_ranges
        );
        all.packets += totals.packets;
        all.decoded += totals.decoded;
        all.exact_ranges += totals.exact_ranges;
    }
    println!(
        "all: packets={} decoded={} exact_ranges={}",
        all.packets, all.decoded, all.exact_ranges
    );
    if all.decoded != all.packets || (require_exact && all.exact_ranges != all.packets) {
        return Err("RFC 8251 conformance requirement not met".to_owned());
    }
    Ok(())
}
