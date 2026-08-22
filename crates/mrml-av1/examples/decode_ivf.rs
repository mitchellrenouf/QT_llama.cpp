//! Small conformance harness for exercising the decoder against IVF vectors.

use std::{env, fs, process};

fn main() {
    let Some(path) = env::args_os().nth(1) else {
        eprintln!("usage: decode_ivf <vector.ivf>");
        process::exit(2);
    };
    let output_directory = env::args_os().nth(2);
    let data = fs::read(&path).unwrap_or_else(|error| {
        eprintln!("failed to read {path:?}: {error}");
        process::exit(2);
    });
    if data.len() < 32 || &data[..4] != b"DKIF" {
        eprintln!("invalid IVF header");
        process::exit(1);
    }
    let header_len = usize::from(u16::from_le_bytes([data[6], data[7]]));
    let mut decoder = mrml_av1::Decoder::new();
    let mut offset = header_len;
    let mut packet = 0;
    let mut displayed = 0;
    while offset < data.len() {
        if data.len() - offset < 12 {
            eprintln!("truncated IVF packet header at {offset}");
            process::exit(1);
        }
        let size = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        let payload = offset + 12;
        let end = payload.checked_add(size).unwrap();
        let mut obu_offset = payload;
        let mut obu_index = 0;
        while obu_offset < end {
            let header = data[obu_offset];
            let header_bytes = 1 + usize::from(header & 4 != 0);
            let mut cursor = obu_offset + header_bytes;
            let mut obu_size = 0usize;
            let mut shift = 0;
            loop {
                let byte = data[cursor];
                cursor += 1;
                obu_size |= usize::from(byte & 0x7f) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            let obu_end = cursor + obu_size;
            eprintln!(
                "packet {packet} OBU {obu_index}: type {} bytes {}",
                (header >> 3) & 15,
                obu_end - obu_offset
            );
            if (header >> 3) & 15 == 6 {
                print_frame_header_diagnostics(&decoder, &data[cursor..obu_end]);
            }
            let frames = decoder
                .decode_obus(&data[obu_offset..obu_end])
                .unwrap_or_else(|error| {
                    eprintln!(
                        "packet {packet} OBU {obu_index} at IVF offset {obu_offset} failed: {error:?}"
                    );
                    process::exit(1);
                });
            if (header >> 3) & 15 == 1 {
                let sequence = decoder.sequence().unwrap();
                eprintln!(
                    "sequence: {}x{} {:?} {}-bit sb128={} monochrome={}",
                    sequence.max_width,
                    sequence.max_height,
                    sequence.chroma_sampling,
                    sequence.bit_depth,
                    sequence.use_128x128_superblock,
                    sequence.monochrome
                );
            }
            for frame in frames {
                println!(
                    "frame {displayed}: {}x{} {:?} {}-bit",
                    frame.width, frame.height, frame.chroma_sampling, frame.bit_depth
                );
                if let Some(directory) = &output_directory {
                    let mut planar =
                        Vec::with_capacity(frame.y.len() + frame.u.len() + frame.v.len());
                    planar.extend_from_slice(&frame.y);
                    planar.extend_from_slice(&frame.u);
                    planar.extend_from_slice(&frame.v);
                    fs::create_dir_all(directory).unwrap();
                    fs::write(
                        std::path::Path::new(directory).join(format!("frame-{displayed:04}.yuv")),
                        planar,
                    )
                    .unwrap();
                }
                displayed += 1;
            }
            obu_offset = obu_end;
            obu_index += 1;
        }
        offset = end;
        packet += 1;
    }
    println!("decoded {displayed} displayed frames from {packet} packets");
}

fn print_frame_header_diagnostics(decoder: &mrml_av1::Decoder, payload: &[u8]) {
    let Some(sequence) = decoder.sequence() else {
        return;
    };
    let references = [mrml_av1::frame_header::ReferenceInfo::default(); 8];
    let Ok(frame_header) =
        mrml_av1::frame_header::parse(payload, sequence, &references, None, 0, 0)
    else {
        return;
    };
    let Some(layout) = frame_header.tile_layout.as_ref() else {
        return;
    };
    eprintln!(
        "header: {} bits, q={}, tx={:?}, delta-q={}, tiles={}x{} sb columns={:?} rows={:?}",
        frame_header.bits_consumed,
        frame_header.quantization.base_q_idx,
        frame_header.tx_mode,
        frame_header.delta_params.delta_q_present,
        layout.columns(),
        layout.rows(),
        &layout.column_starts_sb[..],
        &layout.row_starts_sb[..],
    );
}
