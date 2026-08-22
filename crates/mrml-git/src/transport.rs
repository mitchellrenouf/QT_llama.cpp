use core::fmt;
use mrml_runtime::{Text, Vector};
use crate::ObjectId;

const MAX_PACKET: usize = 65_520;
const MAX_REFS: usize = 1_000_000;
const MAX_RESPONSE: usize = 1024 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteRef { pub name: Text, pub id: ObjectId }

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Advertisement { pub refs: Vector<RemoteRef>, pub capabilities: Vector<Text> }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransportError { Truncated, PacketLength, InvalidHex, InvalidUtf8, InvalidObjectId, InvalidReference, TooManyRefs, TooLarge, RemoteError, MissingPack }

pub fn encode_packet(payload: &[u8], output: &mut Vector<u8>) -> Result<(), TransportError> {
    let length = payload.len().checked_add(4).ok_or(TransportError::TooLarge)?;
    if !(4..=MAX_PACKET).contains(&length) { return Err(TransportError::PacketLength); }
    let hex = hex4(length as u16); output.extend(hex); output.extend(payload.iter().copied()); Ok(())
}

pub fn encode_flush(output: &mut Vector<u8>) { output.extend(*b"0000"); }

pub fn decode_packets(source: &[u8]) -> Result<Vector<Option<Vector<u8>>>, TransportError> {
    let mut cursor=0usize;let mut packets=Vector::new();
    while cursor<source.len(){let header=source.get(cursor..cursor+4).ok_or(TransportError::Truncated)?;cursor+=4;let length=parse_hex4(header)?;if length==0{packets.push(None);continue;}if length<4{return Err(TransportError::PacketLength);}let payload=length-4;if payload>MAX_PACKET-4{return Err(TransportError::PacketLength);}let end=cursor.checked_add(payload).ok_or(TransportError::TooLarge)?;let bytes=source.get(cursor..end).ok_or(TransportError::Truncated)?;let mut owned=Vector::new();owned.extend(bytes.iter().copied());packets.push(Some(owned));cursor=end;}Ok(packets)
}

pub fn parse_advertisement(source: &[u8]) -> Result<Advertisement, TransportError> {
    let packets=decode_packets(source)?;let mut refs=Vector::new();let mut capabilities=Vector::new();let mut first=true;
    for packet in packets {let Some(bytes)=packet else{break};let text=core::str::from_utf8(&bytes).map_err(|_|TransportError::InvalidUtf8)?.trim_end_matches('\n');if text.starts_with("ERR "){return Err(TransportError::RemoteError);}let(reference,caps)=if first{match text.split_once('\0'){Some((left,right))=>(left,Some(right)),None=>(text,None)}}else{(text,None)};first=false;let(id,name)=reference.split_once(' ').ok_or(TransportError::InvalidReference)?;validate_ref(name)?;let id=ObjectId::parse(id).ok_or(TransportError::InvalidObjectId)?;if refs.len()>=MAX_REFS{return Err(TransportError::TooManyRefs);}refs.push(RemoteRef{name:name.into(),id});if let Some(caps)=caps{for cap in caps.split_whitespace(){if cap.chars().any(char::is_control){return Err(TransportError::InvalidReference);}capabilities.push(cap.into());}}
    }Ok(Advertisement{refs,capabilities})
}

pub fn fetch_request(wants:&[ObjectId], have:&[ObjectId], capabilities:&[&str])->Result<Vector<u8>,TransportError>{if wants.is_empty(){return Err(TransportError::InvalidObjectId);}let mut output=Vector::new();for(index,id)in wants.iter().enumerate(){let mut line=mrml_runtime::mrml_format!("want {id}");if index==0{for capability in capabilities{if capability.is_empty()||capability.chars().any(|c|c.is_control()||c.is_whitespace()){return Err(TransportError::InvalidReference);}line.push(' ');line.push_str(capability);}}line.push('\n');encode_packet(line.as_bytes(),&mut output)?;}encode_flush(&mut output);for id in have{encode_packet(mrml_runtime::mrml_format!("have {id}\n").as_bytes(),&mut output)?;}encode_packet(b"done\n",&mut output)?;Ok(output)}

pub fn extract_pack_response(source:&[u8],side_band:bool)->Result<Vector<u8>,TransportError>{
    if !side_band {let start=source.windows(4).position(|window|window==b"PACK").ok_or(TransportError::MissingPack)?;let mut pack=Vector::new();pack.extend(source[start..].iter().copied());return Ok(pack);}
    let packets=decode_packets(source)?;let mut pack=Vector::new();for packet in packets{let Some(bytes)=packet else{continue};if bytes.is_empty(){continue;}match bytes[0]{1=>{if pack.len().checked_add(bytes.len()-1).is_none_or(|n|n>MAX_RESPONSE){return Err(TransportError::TooLarge);}pack.extend(bytes[1..].iter().copied());},2=>{},3=>return Err(TransportError::RemoteError),_=>{let text=core::str::from_utf8(&bytes).unwrap_or("");if !matches!(text,"NAK\n")&&!text.starts_with("ACK "){return Err(TransportError::InvalidReference);}}}}if pack.starts_with(b"PACK"){Ok(pack)}else{Err(TransportError::MissingPack)}}

fn validate_ref(name:&str)->Result<(),TransportError>{if !name.starts_with("refs/")&&name!="HEAD"{return Err(TransportError::InvalidReference);}if name.is_empty()||name.contains("..")||name.contains("@{")||name.contains(['\\',' ','~','^',':','?','*','['])||name.chars().any(char::is_control){Err(TransportError::InvalidReference)}else{Ok(())}}
fn parse_hex4(bytes:&[u8])->Result<usize,TransportError>{if bytes.len()!=4{return Err(TransportError::Truncated);}let mut value=0usize;for byte in bytes{value=value*16+match byte{b'0'..=b'9'=>(*byte-b'0')as usize,b'a'..=b'f'=>(*byte-b'a'+10)as usize,b'A'..=b'F'=>(*byte-b'A'+10)as usize,_=>return Err(TransportError::InvalidHex)};}Ok(value)}
fn hex4(value:u16)->[u8;4]{const H:&[u8;16]=b"0123456789abcdef";[H[((value>>12)&15)as usize],H[((value>>8)&15)as usize],H[((value>>4)&15)as usize],H[(value&15)as usize]]}
impl fmt::Display for TransportError{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{f.write_str(match self{Self::Truncated=>"truncated Git protocol response",Self::PacketLength=>"invalid Git pkt-line length",Self::InvalidHex=>"invalid Git pkt-line hexadecimal length",Self::InvalidUtf8=>"Git protocol text is not UTF-8",Self::InvalidObjectId=>"invalid advertised object ID",Self::InvalidReference=>"invalid advertised reference or capability",Self::TooManyRefs=>"too many advertised references",Self::TooLarge=>"Git protocol response exceeds limit",Self::RemoteError=>"remote Git service reported an error",Self::MissingPack=>"Git response did not contain a pack"})}}
impl core::error::Error for TransportError{}

#[cfg(test)]mod tests{use super::*;
 #[test]fn parses_capability_advertisement(){let id="1111111111111111111111111111111111111111";let line=mrml_runtime::mrml_format!("{id} refs/heads/main\0multi_ack side-band-64k\n");let mut wire=Vector::new();encode_packet(line.as_bytes(),&mut wire).unwrap();encode_flush(&mut wire);let ad=parse_advertisement(&wire).unwrap();assert_eq!(ad.refs[0].name,"refs/heads/main");assert_eq!(ad.capabilities[1],"side-band-64k");}
 #[test]fn builds_fetch_and_extracts_sideband_pack(){let id=ObjectId::parse("2222222222222222222222222222222222222222").unwrap();let request=fetch_request(&[id],&[],&["side-band-64k"]).unwrap();assert!(request.windows(5).any(|v|v==b"want "));let mut response=Vector::new();encode_packet(b"NAK\n",&mut response).unwrap();encode_packet(b"\x01PACKdata",&mut response).unwrap();encode_flush(&mut response);assert_eq!(&*extract_pack_response(&response,true).unwrap(),b"PACKdata");}
 #[test]fn rejects_malformed_lengths_and_error_band(){assert_eq!(decode_packets(b"0003"),Err(TransportError::PacketLength));let mut response=Vector::new();encode_packet(b"\x03denied",&mut response).unwrap();assert_eq!(extract_pack_response(&response,true),Err(TransportError::RemoteError));}
}
