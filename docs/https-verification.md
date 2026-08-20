# HTTPS implementation verification

Verified on 2026-08-20 for the TLS, HTTP, model-download, and HTTPS-server
changes.

## Platforms and commands

Windows used nightly `x86_64-pc-windows-gnullvm` with Rust's bundled MinGW and
LLD components:

```powershell
cargo +nightly-x86_64-pc-windows-gnullvm test --workspace
cargo +nightly-x86_64-pc-windows-gnullvm build --workspace --release
cargo +nightly-x86_64-pc-windows-gnullvm clippy --workspace --all-targets
cargo +nightly-x86_64-pc-windows-gnullvm bench -p mrml-tensor --bench hot_paths -- --test
```

The workspace run passed except that its first sandboxed attempt to launch
headless Edge could not expose a DevTools port. Re-running that exact test with
GUI-process permission passed. All non-browser tests passed on the first full
run.

Linux used a clean Arch Linux WSL2 installation and its native nightly GNU
host:

```bash
cargo test --workspace
cargo build --workspace --release
cargo bench -p mrml-tensor --bench hot_paths -- --test
```

The complete Linux suite and release build passed.

## Protocol and security checks

- ML-KEM-768 was checked against NIST ACVP key-generation, encapsulation, and
  decapsulation vectors. X25519, HKDF, SHA-2, SHA-3, SHAKE, AES-GCM,
  ChaCha20-Poly1305, and RSA verification retain their independent standard
  vectors and tamper tests.
- A live Hugging Face handshake was required to negotiate standardized
  X25519MLKEM768. A real HTTPS model-CDN range request was bounded to 1 KiB and
  passed on Windows and Linux.
- A fresh localhost RSA certificate was used for client/server interoperability
  on Windows and Linux. The test covered certificate-chain and hostname
  validation, RSA-PSS CertificateVerify, both Finished messages, hybrid key
  agreement, and encrypted application data in both directions.
- Model SHA3-512 sidecars reject tampered cached models. Resume checkpoints are
  accepted only after their partial-file digest matches; redirects remove
  authorization and cookie headers when the origin changes.
- TLS records, HTTP response headers, redirect counts, request headers, request
  bodies, and model API responses have explicit size limits. AEAD failures,
  malformed DER/PEM, invalid certificates, conflicting Content-Length fields,
  request transfer encodings, truncated bodies, and invalid resume responses
  are rejected.
- RSA private-exponent selection does not branch on exponent bits and loaded
  private exponents are overwritten on drop. OS cryptographic randomness is
  used for TLS, X25519, ML-KEM, RSA-PSS salts, and AEAD nonces/keys.
- `cargo tree --workspace --edges normal` and `Cargo.lock` contain workspace
  crates only. Source scanning found no `std`/`alloc` crate imports and no
  program path invoking curl or wget.

The static Clippy run completed without errors. It reports existing style and
documentation warnings, including missing `# Safety` sections on several
pre-existing public platform FFI functions; these warnings were not introduced
by the HTTPS changes. Unsafe code remains concentrated in the original native
platform, allocator, SIMD, and accelerator layers.

## Performance evidence

The existing tensor hot-path benchmark completed on both systems. This change
does not alter tensor kernels. Representative optimized comparisons were:

| Check | Windows | Arch Linux |
| --- | ---: | ---: |
| vocabulary top-40 speedup | 5.96x | 12.29x |
| batched RoPE speedup | 7.40x | 6.54x |
| Q8 vocabulary-dot speedup | 4.57x | 4.64x |
| Q4 8K KV scan | 752.4 us | 723.2 us |

## Provenance limitation

Mechanical repository checks can prove the absence of external Cargo sources,
forbidden crate imports, and non-Rust implementation files. They cannot prove
the authorship history of every pre-existing line. Contributors remain
responsible for the original-work declaration required by the README and CC0.
