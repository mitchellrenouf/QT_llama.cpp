use core::fmt;
use mrml_runtime::{TcpStream,Vector};
use crate::{BinaryReader,BinaryWriter,ChannelEvent,ChannelState,EncryptedPacketReader,EncryptedPacketWriter,GitService,Identification,ProtocolError,RsaPrivateKey,build_channel_data,build_channel_eof,build_channel_open,build_exec_request,build_kex_init,build_publickey_auth,build_window_adjust,decode_plain_packet,derive_exchange_keys,encode_plain_packet,negotiate_kex_init,parse_channel_event,parse_identification,verify_rsa_sha2_256};

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

pub struct AuthenticatedSsh{pub wire:SshWire,pub session_id:[u8;32]}
impl AuthenticatedSsh{
 pub fn connect(host:&str,port:u16,user:&str,key:&RsaPrivateKey,expected_host_key:&[u8])->Result<Self,WireError>{
  let(mut wire,server_id)=SshWire::connect(host,port)?;
  let client_kex=build_kex_init().map_err(WireError::Protocol)?;wire.send(&client_kex)?;
  let server_kex=wire.receive()?;negotiate_kex_init(&server_kex).map_err(WireError::Protocol)?;
  let mut secret=[0u8;32];mrml_runtime::fill_random(&mut secret).map_err(|_|WireError::Protocol(ProtocolError::Entropy))?;
  let client_public=mrml_crypto::x25519_public(secret);
  let mut init=BinaryWriter::new();init.byte(30);init.string(&client_public).map_err(WireError::Protocol)?;wire.send(&init.finish())?;
  let reply=wire.receive()?;let mut reader=BinaryReader::new(&reply);if reader.byte().map_err(WireError::Protocol)?!=31{return Err(WireError::Protocol(ProtocolError::InvalidPacket));}let host_key=reader.string().map_err(WireError::Protocol)?;let server_public_slice=reader.string().map_err(WireError::Protocol)?;let server_public:[u8;32]=server_public_slice.try_into().map_err(|_|WireError::Protocol(ProtocolError::InvalidPublicKey))?;let signature=reader.string().map_err(WireError::Protocol)?;if reader.remaining()!=0||!constant_time_equal(host_key,expected_host_key){return Err(WireError::Protocol(ProtocolError::Authentication));}
  let client_id=&CLIENT_ID[..CLIENT_ID.len()-2];let server_line=identification_line(&server_id);let transcript=transcript(client_id,&server_line,&client_kex,&server_kex,host_key,&client_public,&server_public).map_err(WireError::Protocol)?;
  let keys=derive_exchange_keys(secret,server_public,&[&transcript],None).map_err(WireError::Protocol)?;
  verify_rsa_sha2_256(host_key,&keys.exchange_hash,signature).map_err(WireError::Protocol)?;
  wire.send(&[21])?;let newkeys=wire.receive()?;if &*newkeys!=[21]{return Err(WireError::Protocol(ProtocolError::InvalidPacket));}
  let mut client_key=[0u8;16];client_key.copy_from_slice(&keys.client_key[..16]);let mut server_key=[0u8;16];server_key.copy_from_slice(&keys.server_key[..16]);
  wire.enable_encryption(EncryptedPacketReader::new(server_key,keys.server_iv,keys.server_mac),EncryptedPacketWriter::new(client_key,keys.client_iv,keys.client_mac));
  let mut service=BinaryWriter::new();service.byte(5);service.string(b"ssh-userauth").map_err(WireError::Protocol)?;wire.send(&service.finish())?;let accepted=wire.receive()?;let mut accepted_reader=BinaryReader::new(&accepted);if accepted_reader.byte().map_err(WireError::Protocol)?!=6||accepted_reader.text().map_err(WireError::Protocol)?!="ssh-userauth"||accepted_reader.remaining()!=0{return Err(WireError::Protocol(ProtocolError::Authentication));}
  let auth=build_publickey_auth(&keys.exchange_hash,user,key).map_err(WireError::Protocol)?;wire.send(&auth)?;let response=wire.receive()?;if response.first().copied()!=Some(52){return Err(WireError::Protocol(ProtocolError::Authentication));}
  Ok(Self{wire,session_id:keys.exchange_hash})
 }
}

pub struct GitChannel{ssh:AuthenticatedSsh,state:ChannelState}
impl GitChannel{
 pub fn open(mut ssh:AuthenticatedSsh,service:GitService,path:&str)->Result<Self,WireError>{ssh.wire.send(&build_channel_open(0).map_err(WireError::Protocol)?)?;let opened=parse_channel_event(&ssh.wire.receive()?).map_err(WireError::Protocol)?;let state=ChannelState::confirmed(0,&opened).map_err(WireError::Protocol)?;ssh.wire.send(&build_exec_request(state.remote,service,path).map_err(WireError::Protocol)?)?;match parse_channel_event(&ssh.wire.receive()?).map_err(WireError::Protocol)?{ChannelEvent::Success(id)if id==state.local=>Ok(Self{ssh,state}),_=>Err(WireError::Protocol(ProtocolError::Authentication))}}
 pub fn send_all(&mut self,mut bytes:&[u8])->Result<(),WireError>{while !bytes.is_empty(){if self.state.send_window==0{match self.receive_event()?{ChannelEvent::WindowAdjusted{..}=>continue,_=>return Err(WireError::Protocol(ProtocolError::InvalidPacket))}}let count=bytes.len().min(self.state.send_packet as usize).min(self.state.send_window as usize);self.state.accept_send(count).map_err(WireError::Protocol)?;self.ssh.wire.send(&build_channel_data(self.state.remote,&bytes[..count]).map_err(WireError::Protocol)?)?;bytes=&bytes[count..];}Ok(())}
 pub fn send_eof(&mut self)->Result<(),WireError>{self.ssh.wire.send(&build_channel_eof(self.state.remote))}
 pub fn receive_event(&mut self)->Result<ChannelEvent,WireError>{let event=parse_channel_event(&self.ssh.wire.receive()?).map_err(WireError::Protocol)?;match &event{ChannelEvent::WindowAdjusted{recipient,bytes}if *recipient==self.state.local=>self.state.add_window(*bytes).map_err(WireError::Protocol)?,ChannelEvent::Data{recipient,bytes}|ChannelEvent::ExtendedData{recipient,bytes,..}if *recipient==self.state.local=>{self.state.accept_receive(bytes.len()).map_err(WireError::Protocol)?;if self.state.receive_window<1024*1024{let replenish=2*1024*1024-self.state.receive_window;self.ssh.wire.send(&build_window_adjust(self.state.remote,replenish).map_err(WireError::Protocol)?)?;self.state.receive_window+=replenish;}},ChannelEvent::Close(recipient)if *recipient==self.state.local=>self.state.closed=true,_=>{}}Ok(event)}
}

fn identification_line(id:&Identification)->Vector<u8>{let mut line=mrml_runtime::mrml_format!("SSH-{}-{}",id.protocol,id.software);if let Some(comments)=&id.comments{line.push(' ');line.push_str(comments);}let mut bytes=Vector::new();bytes.extend(line.bytes());bytes}
fn transcript(client_id:&[u8],server_id:&[u8],client_kex:&[u8],server_kex:&[u8],host_key:&[u8],client_public:&[u8],server_public:&[u8])->Result<Vector<u8>,ProtocolError>{let mut w=BinaryWriter::new();for value in [client_id,server_id,client_kex,server_kex,host_key,client_public,server_public]{w.string(value)?;}Ok(w.finish())}
fn constant_time_equal(left:&[u8],right:&[u8])->bool{let same_length=left.len()==right.len();let mut difference=0u8;let length=left.len().max(right.len());for index in 0..length{difference|=left.get(index).copied().unwrap_or(0)^right.get(index).copied().unwrap_or(0);}same_length&&difference==0}

#[derive(Clone,Copy,Debug,Eq,PartialEq)]pub enum WireError{Connect,Timeout,Read,Write,Protocol(ProtocolError)}
impl fmt::Display for WireError{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{match self{Self::Protocol(error)=>write!(f,"{error}"),Self::Connect=>f.write_str("failed to connect SSH TCP stream"),Self::Timeout=>f.write_str("failed to configure SSH timeout"),Self::Read=>f.write_str("failed to read SSH stream"),Self::Write=>f.write_str("failed to write SSH stream")}}}
impl core::error::Error for WireError{}

#[cfg(test)]mod tests{use super::*;use mrml_runtime::TcpListener;
 #[test]fn loopback_exchanges_identification_and_plain_packet(){let listener=TcpListener::bind([127,0,0,1],0).unwrap();let port=listener.local_port().unwrap();assert!(mrml_runtime::spawn_detached(move||{let mut stream=listener.accept().unwrap();let mut id=[0u8;CLIENT_ID.len()];stream.read_exact(&mut id).unwrap();assert_eq!(&id,CLIENT_ID);stream.write_all(b"banner\r\nSSH-2.0-test_server\r\n").unwrap();let mut header=[0u8;4];stream.read_exact(&mut header).unwrap();let length=u32::from_be_bytes(header)as usize;let mut packet=Vector::new();packet.extend(header);packet.try_resize(length+4,0).unwrap();stream.read_exact(&mut packet[4..]).unwrap();assert_eq!(decode_plain_packet(&packet,8).unwrap().0,b"hello");let response=encode_plain_packet(b"world",8).unwrap();stream.write_all(&response).unwrap();}).is_ok());let(mut wire,id)=SshWire::connect("127.0.0.1",port).unwrap();assert_eq!(id.software,"test_server");wire.send(b"hello").unwrap();assert_eq!(&*wire.receive().unwrap(),b"world");}
}
