use core::fmt;
use mrml_runtime::{TcpStream,Vector};
use crate::{EncryptedPacketReader,EncryptedPacketWriter,Identification,ProtocolError,decode_plain_packet,encode_plain_packet,parse_identification};

const CLIENT_ID:&[u8]=b"SSH-2.0-mrml_0.4\r\n";
const MAX_BANNER:usize=50*257;

pub struct SshWire{stream:TcpStream,reader:Option<EncryptedPacketReader>,writer:Option<EncryptedPacketWriter>}
impl SshWire{
 pub fn connect(host:&str,port:u16)->Result<(Self,Identification),WireError>{let mut stream=TcpStream::connect_host(host,port).map_err(|_|WireError::Connect)?;stream.set_read_timeout_millis(15_000).map_err(|_|WireError::Timeout)?;stream.set_write_timeout_millis(15_000).map_err(|_|WireError::Timeout)?;stream.write_all(CLIENT_ID).map_err(|_|WireError::Write)?;let mut banner=Vector::new();loop{if banner.len()>=MAX_BANNER{return Err(WireError::Protocol(ProtocolError::InvalidIdentification));}let mut byte=[0];stream.read_exact(&mut byte).map_err(|_|WireError::Read)?;banner.push(byte[0]);match parse_identification(&banner){Ok((id,_))=>return Ok((Self{stream,reader:None,writer:None},id)),Err(ProtocolError::Truncated)=>{},Err(error)=>return Err(WireError::Protocol(error))}}
 }
 pub fn enable_encryption(&mut self,reader:EncryptedPacketReader,writer:EncryptedPacketWriter){self.reader=Some(reader);self.writer=Some(writer);}
 pub fn send(&mut self,payload:&[u8])->Result<(),WireError>{let packet=match self.writer.as_mut(){Some(writer)=>writer.encode(payload),None=>encode_plain_packet(payload,8)}.map_err(WireError::Protocol)?;self.stream.write_all(&packet).map_err(|_|WireError::Write)}
 pub fn receive(&mut self)->Result<Vector<u8>,WireError>{if let Some(reader)=self.reader.as_mut(){let mut first=[0u8;16];self.stream.read_exact(&mut first).map_err(|_|WireError::Read)?;let total=reader.wire_length(&first).map_err(WireError::Protocol)?;let mut wire=Vector::new();wire.extend(first);wire.try_resize(total,0).map_err(|_|WireError::Protocol(ProtocolError::Length))?;self.stream.read_exact(&mut wire[16..]).map_err(|_|WireError::Read)?;reader.decode(&wire).map(|pair|pair.0).map_err(WireError::Protocol)}else{let mut header=[0u8;4];self.stream.read_exact(&mut header).map_err(|_|WireError::Read)?;let length=u32::from_be_bytes(header)as usize;if length<6||length>1024*1024+256{return Err(WireError::Protocol(ProtocolError::InvalidPacket));}let total=length+4;let mut wire=Vector::new();wire.extend(header);wire.try_resize(total,0).map_err(|_|WireError::Protocol(ProtocolError::Length))?;self.stream.read_exact(&mut wire[4..]).map_err(|_|WireError::Read)?;let(payload,_)=decode_plain_packet(&wire,8).map_err(WireError::Protocol)?;let mut owned=Vector::new();owned.extend(payload.iter().copied());Ok(owned)}}
}

#[derive(Clone,Copy,Debug,Eq,PartialEq)]pub enum WireError{Connect,Timeout,Read,Write,Protocol(ProtocolError)}
impl fmt::Display for WireError{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{match self{Self::Protocol(error)=>write!(f,"{error}"),Self::Connect=>f.write_str("failed to connect SSH TCP stream"),Self::Timeout=>f.write_str("failed to configure SSH timeout"),Self::Read=>f.write_str("failed to read SSH stream"),Self::Write=>f.write_str("failed to write SSH stream")}}}
impl core::error::Error for WireError{}

#[cfg(test)]mod tests{use super::*;use mrml_runtime::TcpListener;
 #[test]fn loopback_exchanges_identification_and_plain_packet(){let listener=TcpListener::bind([127,0,0,1],0).unwrap();let port=listener.local_port().unwrap();assert!(mrml_runtime::spawn_detached(move||{let mut stream=listener.accept().unwrap();let mut id=[0u8;CLIENT_ID.len()];stream.read_exact(&mut id).unwrap();assert_eq!(&id,CLIENT_ID);stream.write_all(b"banner\r\nSSH-2.0-test_server\r\n").unwrap();let mut header=[0u8;4];stream.read_exact(&mut header).unwrap();let length=u32::from_be_bytes(header)as usize;let mut packet=Vector::new();packet.extend(header);packet.try_resize(length+4,0).unwrap();stream.read_exact(&mut packet[4..]).unwrap();assert_eq!(decode_plain_packet(&packet,8).unwrap().0,b"hello");let response=encode_plain_packet(b"world",8).unwrap();stream.write_all(&response).unwrap();}).is_ok());let(mut wire,id)=SshWire::connect("127.0.0.1",port).unwrap();assert_eq!(id.software,"test_server");wire.send(b"hello").unwrap();assert_eq!(&*wire.receive().unwrap(),b"world");}
}
