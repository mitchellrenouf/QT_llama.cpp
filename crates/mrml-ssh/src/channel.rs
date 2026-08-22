use mrml_runtime::Vector;
use crate::{BinaryReader, BinaryWriter, ProtocolError};

const INITIAL_WINDOW:u32=2*1024*1024;
const MAX_PACKET:u32=32*1024;

#[derive(Clone,Copy,Debug,Eq,PartialEq)]pub enum GitService{UploadPack,ReceivePack}
impl GitService{fn command(self)->&'static str{match self{Self::UploadPack=>"git-upload-pack",Self::ReceivePack=>"git-receive-pack"}}}

#[derive(Clone,Debug,Eq,PartialEq)]pub enum ChannelEvent{OpenConfirmed{recipient:u32,sender:u32,window:u32,packet:u32},OpenFailed{recipient:u32,reason:u32},WindowAdjusted{recipient:u32,bytes:u32},Data{recipient:u32,bytes:Vector<u8>},ExtendedData{recipient:u32,kind:u32,bytes:Vector<u8>},Success(u32),Failure(u32),ExitStatus{recipient:u32,status:u32},Eof(u32),Close(u32)}

#[derive(Clone,Debug,Eq,PartialEq)]pub struct ChannelState{pub local:u32,pub remote:u32,pub send_window:u32,pub send_packet:u32,pub receive_window:u32,pub closed:bool}
impl ChannelState{
 pub fn confirmed(local:u32,event:&ChannelEvent)->Result<Self,ProtocolError>{match *event{ChannelEvent::OpenConfirmed{recipient,sender,window,packet}if recipient==local&&packet!=0&&packet<=1024*1024=>Ok(Self{local,remote:sender,send_window:window,send_packet:packet,receive_window:INITIAL_WINDOW,closed:false}),_=>Err(ProtocolError::InvalidPacket)}}
 pub fn accept_send(&mut self,length:usize)->Result<(),ProtocolError>{let length=u32::try_from(length).map_err(|_|ProtocolError::Length)?;if self.closed||length>self.send_packet||length>self.send_window{return Err(ProtocolError::Length);}self.send_window-=length;Ok(())}
 pub fn add_window(&mut self,bytes:u32)->Result<(),ProtocolError>{self.send_window=self.send_window.checked_add(bytes).ok_or(ProtocolError::Length)?;Ok(())}
 pub fn accept_receive(&mut self,length:usize)->Result<(),ProtocolError>{let length=u32::try_from(length).map_err(|_|ProtocolError::Length)?;self.receive_window=self.receive_window.checked_sub(length).ok_or(ProtocolError::Length)?;Ok(())}
}

pub fn build_service_request(name:&str)->Result<Vector<u8>,ProtocolError>{if !matches!(name,"ssh-userauth"|"ssh-connection"){return Err(ProtocolError::InvalidPacket);}let mut w=BinaryWriter::new();w.byte(5);w.string(name.as_bytes())?;Ok(w.finish())}
pub fn build_channel_open(sender:u32)->Result<Vector<u8>,ProtocolError>{let mut w=BinaryWriter::new();w.byte(90);w.string(b"session")?;w.u32(sender);w.u32(INITIAL_WINDOW);w.u32(MAX_PACKET);Ok(w.finish())}
pub fn build_exec_request(recipient:u32,service:GitService,path:&str)->Result<Vector<u8>,ProtocolError>{validate_path(path)?;let command=mrml_runtime::mrml_format!("{} '{}'",service.command(),path);let mut w=BinaryWriter::new();w.byte(98);w.u32(recipient);w.string(b"exec")?;w.boolean(true);w.string(command.as_bytes())?;Ok(w.finish())}
pub fn build_channel_data(recipient:u32,data:&[u8])->Result<Vector<u8>,ProtocolError>{if data.is_empty()||data.len()>MAX_PACKET as usize{return Err(ProtocolError::Length);}let mut w=BinaryWriter::new();w.byte(94);w.u32(recipient);w.string(data)?;Ok(w.finish())}
pub fn build_channel_eof(recipient:u32)->Vector<u8>{simple(96,recipient)}
pub fn build_channel_close(recipient:u32)->Vector<u8>{simple(97,recipient)}
pub fn build_window_adjust(recipient:u32,bytes:u32)->Result<Vector<u8>,ProtocolError>{if bytes==0{return Err(ProtocolError::Length);}let mut w=BinaryWriter::new();w.byte(93);w.u32(recipient);w.u32(bytes);Ok(w.finish())}
fn simple(kind:u8,recipient:u32)->Vector<u8>{let mut w=BinaryWriter::new();w.byte(kind);w.u32(recipient);w.finish()}

pub fn parse_channel_event(message:&[u8])->Result<ChannelEvent,ProtocolError>{let mut r=BinaryReader::new(message);let kind=r.byte()?;let event=match kind{91=>ChannelEvent::OpenConfirmed{recipient:r.u32()?,sender:r.u32()?,window:r.u32()?,packet:r.u32()?},92=>{let recipient=r.u32()?;let reason=r.u32()?;r.string()?;r.string()?;ChannelEvent::OpenFailed{recipient,reason}},93=>ChannelEvent::WindowAdjusted{recipient:r.u32()?,bytes:r.u32()?},94=>{let recipient=r.u32()?;ChannelEvent::Data{recipient,bytes:copy(r.string()?)}},95=>{let recipient=r.u32()?;let kind=r.u32()?;ChannelEvent::ExtendedData{recipient,kind,bytes:copy(r.string()?)}},99=>ChannelEvent::Success(r.u32()?),100=>ChannelEvent::Failure(r.u32()?),96=>ChannelEvent::Eof(r.u32()?),97=>ChannelEvent::Close(r.u32()?),98=>{let recipient=r.u32()?;if r.text()? != "exit-status"||!r.boolean()?{return Err(ProtocolError::InvalidPacket);}ChannelEvent::ExitStatus{recipient,status:r.u32()?}},_=>return Err(ProtocolError::InvalidPacket)};if r.remaining()!=0{return Err(ProtocolError::InvalidPacket);}Ok(event)}
fn copy(value:&[u8])->Vector<u8>{let mut out=Vector::new();out.extend(value.iter().copied());out}
fn validate_path(path:&str)->Result<(),ProtocolError>{if path.is_empty()||path.len()>4096||path.starts_with('-')||path.contains('\'')||path.chars().any(|c|c.is_control()){return Err(ProtocolError::InvalidPacket);}if path.bytes().any(|b|!(b.is_ascii_alphanumeric()||matches!(b,b'/'|b'.'|b'_'|b'-'|b'~'))){return Err(ProtocolError::InvalidPacket);}Ok(())}

#[cfg(test)]mod tests{use super::*;
 #[test]fn exec_is_shell_injection_safe(){let message=build_exec_request(7,GitService::UploadPack,"owner/repo.git").unwrap();let mut r=BinaryReader::new(&message);assert_eq!(r.byte(),Ok(98));assert_eq!(r.u32(),Ok(7));assert_eq!(r.text(),Ok("exec"));assert_eq!(r.boolean(),Ok(true));assert_eq!(r.text(),Ok("git-upload-pack 'owner/repo.git'"));assert!(build_exec_request(7,GitService::UploadPack,"repo';evil").is_err());}
 #[test]fn parses_data_and_enforces_windows(){let mut w=BinaryWriter::new();w.byte(94);w.u32(3);w.string(b"PACK").unwrap();assert_eq!(parse_channel_event(&w.finish()),Ok(ChannelEvent::Data{recipient:3,bytes:Vector::from(*b"PACK")}));let event=ChannelEvent::OpenConfirmed{recipient:1,sender:9,window:10,packet:8};let mut state=ChannelState::confirmed(1,&event).unwrap();assert_eq!(state.accept_send(8),Ok(()));assert!(state.accept_send(3).is_err());state.add_window(5).unwrap();assert_eq!(state.accept_send(5),Ok(()));}
 #[test]fn rejects_trailing_and_unknown_messages(){assert_eq!(parse_channel_event(&[255]),Err(ProtocolError::InvalidPacket));let mut eof=build_channel_eof(1);eof.push(0);assert_eq!(parse_channel_event(&eof),Err(ProtocolError::InvalidPacket));}
 #[test]fn window_adjust_is_bounded_and_addressed(){assert!(build_window_adjust(4,0).is_err());let message=build_window_adjust(4,99).unwrap();assert_eq!(parse_channel_event(&message),Ok(ChannelEvent::WindowAdjusted{recipient:4,bytes:99}));}
}
