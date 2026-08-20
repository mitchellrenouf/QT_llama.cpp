use crate::{Certificate, CertificateError};
use mrml_runtime::Vector;

pub fn verify_server_chain(hostname: &str, chain: &[&[u8]]) -> Result<(), CertificateError> {
    let leaf_der = *chain.first().ok_or(CertificateError::Malformed)?;
    let leaf = Certificate::parse(leaf_der)?; leaf.verify_hostname(hostname)?; leaf.verify_time_now()?;
    for pair in chain.windows(2) {
        let child = Certificate::parse(pair[0])?; let issuer = Certificate::parse(pair[1])?;
        issuer.require_ca()?; issuer.verify_time_now()?; child.verify_signed_by(&issuer)?;
    }
    let last = Certificate::parse(*chain.last().ok_or(CertificateError::Malformed)?)?;
    let mut trusted = false; let mut store_available = false;
    if let Some(path) = mrml_runtime::environment_variable("MRML_CA_BUNDLE") {
        let bundle = mrml_runtime::read_file(&path).map_err(|_| CertificateError::TrustStoreUnavailable)?; store_available = true;
        for_each_pem_certificate(&bundle, |der| { if let Ok(root)=Certificate::parse(der) { if last.verify_signed_by(&root).is_ok() { trusted=true; return false; } } true })?;
    }
    #[cfg(windows)] {
        if mrml_runtime::visit_root_certificates(|der| {
            if let Ok(root) = Certificate::parse(der) {
                if last.verify_signed_by(&root).is_ok() { trusted = true; return false; }
            } true
        }) { store_available = true; }
    }
    #[cfg(unix)] {
        let paths = ["/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/ca-bundle.pem"];
        let mut bundle = None; for path in paths { if let Ok(bytes) = mrml_runtime::read_file(path) { bundle = Some(bytes); break; } }
        if let Some(bundle)=bundle { store_available=true; for_each_pem_certificate(&bundle, |der| {
            if let Ok(root) = Certificate::parse(der) {
                if last.verify_signed_by(&root).is_ok() { trusted = true; return false; }
            } true
        })?; }
    }
    if trusted { Ok(()) } else if store_available { Err(CertificateError::UntrustedIssuer) } else { Err(CertificateError::TrustStoreUnavailable) }
}

#[allow(dead_code)]
fn base64_value(byte: u8) -> Option<u8> { match byte { b'A'..=b'Z'=>Some(byte-b'A'),b'a'..=b'z'=>Some(byte-b'a'+26),b'0'..=b'9'=>Some(byte-b'0'+52),b'+'=>Some(62),b'/'=>Some(63),_=>None } }

#[allow(dead_code)]
fn decode_base64(input: &[u8], output: &mut Vector<u8>) -> Result<(), CertificateError> {
    let mut group=[0u8;4]; let mut used=0; let mut ended=false;
    for byte in input.iter().copied() {
        if byte.is_ascii_whitespace() { continue; } if ended { return Err(CertificateError::Malformed); }
        group[used]=byte; used+=1;
        if used==4 {
            let a=base64_value(group[0]).ok_or(CertificateError::Malformed)? as u32; let b=base64_value(group[1]).ok_or(CertificateError::Malformed)? as u32;
            let c=if group[2]==b'=' {0}else{base64_value(group[2]).ok_or(CertificateError::Malformed)? as u32}; let d=if group[3]==b'=' {0}else{base64_value(group[3]).ok_or(CertificateError::Malformed)? as u32};
            output.try_push(((a<<2)|(b>>4)) as u8).map_err(|_|CertificateError::Malformed)?;
            if group[2]!=b'=' { output.try_push(((b<<4)|(c>>2)) as u8).map_err(|_|CertificateError::Malformed)?; }
            if group[3]!=b'=' { output.try_push(((c<<6)|d) as u8).map_err(|_|CertificateError::Malformed)?; }
            if group[2]==b'=' || group[3]==b'=' { ended=true; } used=0;
        }
    }
    if used==0 { Ok(()) } else { Err(CertificateError::Malformed) }
}

#[allow(dead_code)]
fn for_each_pem_certificate(bundle:&[u8],mut visitor:impl FnMut(&[u8])->bool)->Result<(),CertificateError>{
    const BEGIN:&[u8]=b"-----BEGIN CERTIFICATE-----"; const END:&[u8]=b"-----END CERTIFICATE-----"; let mut position=0;
    while let Some(start)=find(&bundle[position..],BEGIN) { let content_start=position+start+BEGIN.len(); let end=find(&bundle[content_start..],END).ok_or(CertificateError::Malformed)?+content_start; let mut der=Vector::new(); decode_base64(&bundle[content_start..end],&mut der)?; if !visitor(&der){break;} position=end+END.len(); }
    Ok(())
}
#[allow(dead_code)]
fn find(haystack:&[u8],needle:&[u8])->Option<usize>{haystack.windows(needle.len()).position(|part|part==needle)}

#[cfg(test)] mod tests { use super::*; #[test] fn base64_decoder_is_strict(){let mut out=Vector::new();decode_base64(b"TUlNRQ==",&mut out).unwrap();assert_eq!(&out[..],b"MIME");assert!(decode_base64(b"A===",&mut Vector::new()).is_err());} }
