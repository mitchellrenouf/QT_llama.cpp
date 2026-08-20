use mrml_crypto::rsa_pkcs1_sha256_verify;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CertificateError { Malformed, UnsupportedAlgorithm, InvalidSignature, HostnameMismatch, NotCertificateAuthority, NotYetValid, Expired, ClockUnavailable, TrustStoreUnavailable, UntrustedIssuer }

#[derive(Clone, Copy)]
struct Element<'a> { tag: u8, value: &'a [u8], encoded: &'a [u8] }

struct Reader<'a> { bytes: &'a [u8], position: usize }
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, position: 0 } }
    fn next(&mut self) -> Result<Element<'a>, CertificateError> {
        let start = self.position; let tag = *self.bytes.get(self.position).ok_or(CertificateError::Malformed)?; self.position += 1;
        let first = *self.bytes.get(self.position).ok_or(CertificateError::Malformed)?; self.position += 1;
        let length = if first & 0x80 == 0 { first as usize } else {
            let count = (first & 0x7f) as usize;
            if count == 0 || count > core::mem::size_of::<usize>() || self.position + count > self.bytes.len() { return Err(CertificateError::Malformed); }
            if self.bytes[self.position] == 0 { return Err(CertificateError::Malformed); }
            let mut value = 0usize; for byte in &self.bytes[self.position..self.position + count] { value = value.checked_mul(256).and_then(|v| v.checked_add(*byte as usize)).ok_or(CertificateError::Malformed)?; }
            self.position += count; if value < 128 { return Err(CertificateError::Malformed); } value
        };
        let end = self.position.checked_add(length).ok_or(CertificateError::Malformed)?;
        if end > self.bytes.len() { return Err(CertificateError::Malformed); }
        let value = &self.bytes[self.position..end]; self.position = end;
        Ok(Element { tag, value, encoded: &self.bytes[start..end] })
    }
    fn finish(self) -> Result<(), CertificateError> { if self.position == self.bytes.len() { Ok(()) } else { Err(CertificateError::Malformed) } }
}

const SHA256_WITH_RSA: &[u8] = &[0x2a,0x86,0x48,0x86,0xf7,0x0d,0x01,0x01,0x0b];
const RSA_ENCRYPTION: &[u8] = &[0x2a,0x86,0x48,0x86,0xf7,0x0d,0x01,0x01,0x01];
const SUBJECT_ALT_NAME: &[u8] = &[0x55,0x1d,0x11];
const BASIC_CONSTRAINTS: &[u8] = &[0x55,0x1d,0x13];

fn algorithm(element: Element<'_>, expected: &[u8]) -> Result<(), CertificateError> {
    if element.tag != 0x30 { return Err(CertificateError::Malformed); }
    let mut reader = Reader::new(element.value); let oid = reader.next()?;
    if oid.tag != 0x06 || oid.value != expected { return Err(CertificateError::UnsupportedAlgorithm); }
    if reader.position < reader.bytes.len() { let parameter = reader.next()?; if parameter.tag != 0x05 || !parameter.value.is_empty() { return Err(CertificateError::Malformed); } }
    reader.finish()
}

pub struct Certificate<'a> {
    tbs: &'a [u8], modulus: &'a [u8], exponent: &'a [u8], signature: &'a [u8], extensions: &'a [u8], not_before: u64, not_after: u64,
}

impl<'a> Certificate<'a> {
    pub fn parse(der: &'a [u8]) -> Result<Self, CertificateError> {
        let mut outer = Reader::new(der); let certificate = outer.next()?; outer.finish()?;
        if certificate.tag != 0x30 { return Err(CertificateError::Malformed); }
        let mut fields = Reader::new(certificate.value); let tbs = fields.next()?; let signature_algorithm = fields.next()?; let signature = fields.next()?; fields.finish()?;
        if tbs.tag != 0x30 || signature.tag != 0x03 || signature.value.first() != Some(&0) { return Err(CertificateError::Malformed); }
        algorithm(signature_algorithm, SHA256_WITH_RSA)?;
        let mut body = Reader::new(tbs.value);
        let first = body.next()?; if first.tag != 0xa0 { return Err(CertificateError::Malformed); }
        let serial = body.next()?; if serial.tag != 0x02 || serial.value.is_empty() { return Err(CertificateError::Malformed); }
        algorithm(body.next()?, SHA256_WITH_RSA)?;
        let issuer = body.next()?; if issuer.tag != 0x30 { return Err(CertificateError::Malformed); }
        let validity = body.next()?; if validity.tag != 0x30 { return Err(CertificateError::Malformed); }
        let mut times = Reader::new(validity.value); let not_before = certificate_time(times.next()?)?; let not_after = certificate_time(times.next()?)?; times.finish()?;
        if not_after < not_before { return Err(CertificateError::Malformed); }
        let subject = body.next()?; if subject.tag != 0x30 { return Err(CertificateError::Malformed); }
        let spki = body.next()?; if spki.tag != 0x30 { return Err(CertificateError::Malformed); }
        let mut public = Reader::new(spki.value); algorithm(public.next()?, RSA_ENCRYPTION)?; let bits = public.next()?; public.finish()?;
        if bits.tag != 0x03 || bits.value.first() != Some(&0) { return Err(CertificateError::Malformed); }
        let mut key_outer = Reader::new(&bits.value[1..]); let key = key_outer.next()?; key_outer.finish()?; if key.tag != 0x30 { return Err(CertificateError::Malformed); }
        let mut key_fields = Reader::new(key.value); let modulus = key_fields.next()?; let exponent = key_fields.next()?; key_fields.finish()?;
        if modulus.tag != 0x02 || exponent.tag != 0x02 || modulus.value.is_empty() || exponent.value.is_empty() { return Err(CertificateError::Malformed); }
        let modulus = if modulus.value[0] == 0 { &modulus.value[1..] } else { modulus.value };
        let exponent = if exponent.value[0] == 0 { &exponent.value[1..] } else { exponent.value };
        let mut extensions = &[][..];
        while body.position < body.bytes.len() { let element = body.next()?; if element.tag == 0xa3 { extensions = element.value; } else if element.tag != 0x81 && element.tag != 0x82 { return Err(CertificateError::Malformed); } }
        Ok(Self { tbs: tbs.encoded, modulus, exponent, signature: &signature.value[1..], extensions, not_before, not_after })
    }

    pub fn verify_signed_by(&self, issuer: &Certificate<'_>) -> Result<(), CertificateError> {
        rsa_pkcs1_sha256_verify(issuer.modulus, issuer.exponent, self.tbs, self.signature).map_err(|_| CertificateError::InvalidSignature)
    }

    fn extension(&self, wanted: &[u8]) -> Result<Option<&'a [u8]>, CertificateError> {
        if self.extensions.is_empty() { return Ok(None); }
        let mut wrapper = Reader::new(self.extensions); let sequence = wrapper.next()?; wrapper.finish()?; if sequence.tag != 0x30 { return Err(CertificateError::Malformed); }
        let mut entries = Reader::new(sequence.value);
        while entries.position < entries.bytes.len() {
            let entry = entries.next()?; if entry.tag != 0x30 { return Err(CertificateError::Malformed); } let mut fields = Reader::new(entry.value);
            let oid = fields.next()?; if oid.tag != 0x06 { return Err(CertificateError::Malformed); }
            let mut value = fields.next()?; if value.tag == 0x01 { value = fields.next()?; }
            fields.finish()?; if value.tag != 0x04 { return Err(CertificateError::Malformed); }
            if oid.value == wanted { return Ok(Some(value.value)); }
        } Ok(None)
    }

    pub fn require_ca(&self) -> Result<(), CertificateError> {
        let value = self.extension(BASIC_CONSTRAINTS)?.ok_or(CertificateError::NotCertificateAuthority)?;
        let mut outer = Reader::new(value); let sequence = outer.next()?; outer.finish()?; if sequence.tag != 0x30 { return Err(CertificateError::Malformed); }
        let mut fields = Reader::new(sequence.value); let ca = fields.next()?;
        if ca.tag == 0x01 && ca.value == [0xff] { Ok(()) } else { Err(CertificateError::NotCertificateAuthority) }
    }

    pub fn verify_hostname(&self, hostname: &str) -> Result<(), CertificateError> {
        let value = self.extension(SUBJECT_ALT_NAME)?.ok_or(CertificateError::HostnameMismatch)?;
        let mut outer = Reader::new(value); let names = outer.next()?; outer.finish()?; if names.tag != 0x30 { return Err(CertificateError::Malformed); }
        let mut entries = Reader::new(names.value);
        while entries.position < entries.bytes.len() { let name = entries.next()?; if name.tag == 0x82 && dns_matches(name.value, hostname.as_bytes()) { return Ok(()); } }
        Err(CertificateError::HostnameMismatch)
    }

    pub fn verify_time(&self, now: u64) -> Result<(), CertificateError> {
        if now < self.not_before { Err(CertificateError::NotYetValid) } else if now > self.not_after { Err(CertificateError::Expired) } else { Ok(()) }
    }

    pub fn verify_time_now(&self) -> Result<(), CertificateError> { self.verify_time(mrml_runtime::unix_time_seconds().ok_or(CertificateError::ClockUnavailable)?) }
}

fn decimal(bytes: &[u8]) -> Result<u32, CertificateError> {
    let mut value = 0u32; for byte in bytes { if !byte.is_ascii_digit() { return Err(CertificateError::Malformed); } value = value * 10 + (byte - b'0') as u32; } Ok(value)
}

fn certificate_time(element: Element<'_>) -> Result<u64, CertificateError> {
    let (year, rest) = match (element.tag, element.value.len()) {
        (0x17, 13) => { let short = decimal(&element.value[..2])?; (if short >= 50 { 1900 + short } else { 2000 + short }, &element.value[2..]) },
        (0x18, 15) => (decimal(&element.value[..4])?, &element.value[4..]),
        _ => return Err(CertificateError::Malformed),
    };
    if rest[10] != b'Z' { return Err(CertificateError::Malformed); }
    let month=decimal(&rest[..2])?; let day=decimal(&rest[2..4])?; let hour=decimal(&rest[4..6])?; let minute=decimal(&rest[6..8])?; let second=decimal(&rest[8..10])?;
    if !(1..=12).contains(&month) || day == 0 || day > days_in_month(year,month) || hour > 23 || minute > 59 || second > 59 { return Err(CertificateError::Malformed); }
    let mut y=year as i64; let m=month as i64; y -= (m <= 2) as i64; let era=if y>=0 { y } else { y-399 }/400; let yoe=y-era*400; let shifted=m+if m>2 {-3} else {9}; let doy=(153*shifted+2)/5+day as i64-1; let days=era*146097+yoe*365+yoe/4-yoe/100+doy-719468;
    if days < 0 { return Err(CertificateError::Malformed); }
    Ok(days as u64*86400+hour as u64*3600+minute as u64*60+second as u64)
}

fn days_in_month(year:u32,month:u32)->u32 { match month { 1|3|5|7|8|10|12=>31,4|6|9|11=>30,2=>if year%4==0&&(year%100!=0||year%400==0){29}else{28},_=>0 } }

fn dns_matches(pattern: &[u8], hostname: &[u8]) -> bool {
    if pattern.iter().any(|b| !b.is_ascii()) || hostname.iter().any(|b| !b.is_ascii()) { return false; }
    if pattern.starts_with(b"*.") {
        let Some(dot) = hostname.iter().position(|byte| *byte == b'.') else { return false; };
        dot != 0 && hostname[dot + 1..].eq_ignore_ascii_case(&pattern[2..])
    } else { hostname.eq_ignore_ascii_case(pattern) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn wildcard_matches_exactly_one_label() {
        assert!(dns_matches(b"*.example.com", b"api.example.com")); assert!(!dns_matches(b"*.example.com", b"a.b.example.com")); assert!(!dns_matches(b"*.example.com", b"example.com"));
    }
    #[test] fn der_rejects_nonminimal_and_truncated_lengths() {
        assert!(Reader::new(&[0x30,0x81,0x01,0]).next().is_err()); assert!(Reader::new(&[0x30,2,0]).next().is_err());
    }
    #[test] fn external_public_chain_when_configured() {
        let Some(directory) = mrml_runtime::environment_variable("MRML_TLS_FIXTURES") else { return; };
        let leaf_bytes = mrml_runtime::read_file(&mrml_runtime::join_path(&directory, "leaf.der")).unwrap();
        let issuer_bytes = mrml_runtime::read_file(&mrml_runtime::join_path(&directory, "intermediate.der")).unwrap();
        let leaf = Certificate::parse(&leaf_bytes).unwrap(); let issuer = Certificate::parse(&issuer_bytes).unwrap();
        leaf.verify_hostname("huggingface.co").unwrap(); leaf.verify_signed_by(&issuer).unwrap(); issuer.require_ca().unwrap();
        leaf.verify_time_now().unwrap(); issuer.verify_time_now().unwrap();
        crate::verify_server_chain("huggingface.co", &[&leaf_bytes, &issuer_bytes]).unwrap();
        assert_eq!(leaf.verify_hostname("attacker.example"), Err(CertificateError::HostnameMismatch));
    }
    #[test] fn parses_utc_and_generalized_certificate_times() {
        assert_eq!(certificate_time(Element{tag:0x17,value:b"700101000000Z",encoded:&[]}).unwrap(),0);
        assert_eq!(certificate_time(Element{tag:0x18,value:b"20000229000000Z",encoded:&[]}).unwrap(),951782400);
        assert!(certificate_time(Element{tag:0x18,value:b"21000229000000Z",encoded:&[]}).is_err());
    }
}
