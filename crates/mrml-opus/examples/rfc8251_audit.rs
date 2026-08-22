use std::path::{Path, PathBuf};

use mrml_opus::{Decoder, Packet};

#[derive(Clone, Copy, Default)]
struct Totals {
    packets: usize,
    decoded: usize,
    exact_ranges: usize,
}

fn audit(path: &Path, report_first: bool) -> Result<Totals, String> {
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
                let actual_range = decoder.final_range();
                if report_first
                    && actual_range != expected_range
                    && totals.exact_ranges + 1 == totals.packets
                {
                    let hex = packet.iter().fold(String::new(), |mut text, byte| {
                        use std::fmt::Write;
                        let _ = write!(text, "{byte:02X}");
                        text
                    });
                    println!(
                        "{}: first mismatch packet={} expected={expected_range:#010x} actual={actual_range:#010x} bytes={hex}",
                        path.display(),
                        totals.packets
                    );
                }
                totals.exact_ranges += usize::from(actual_range == expected_range);
            }
        }
        offset = packet_end;
    }
    Ok(totals)
}

fn main() -> Result<(), String> {
    let mut arguments = std::env::args_os().skip(1);
    let input = PathBuf::from(arguments.next().ok_or_else(|| {
        "usage: rfc8251_audit <vector-directory|vector.bit> [--require-exact] [--report-first]"
            .to_owned()
    })?);
    let mut require_exact = false;
    let mut report_first = false;
    for argument in arguments {
        if argument == "--require-exact" {
            require_exact = true;
        } else if argument == "--report-first" {
            report_first = true;
        } else {
            return Err(format!(
                "unrecognized option: {}",
                argument.to_string_lossy()
            ));
        }
    }
    let mut all = Totals::default();
    let paths: Vec<(String, PathBuf)> = if input.is_file() {
        let label = input
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("vector")
            .to_owned();
        vec![(label, input)]
    } else {
        (1..=12)
            .map(|number| {
                (
                    format!("{number:02}"),
                    input.join(format!("testvector{number:02}.bit")),
                )
            })
            .collect()
    };
    for (label, path) in paths {
        let totals = audit(&path, report_first)?;
        println!(
            "{label}: packets={} decoded={} exact_ranges={}",
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
