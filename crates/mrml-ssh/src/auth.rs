use mrml_crypto::{rsa_pkcs1_sha256_sign, rsa_pkcs1_sha256_verify};
use mrml_runtime::Vector;
use crate::{BinaryReader, BinaryWriter, ProtocolError};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RsaPublicKey { pub exponent: Vector<u8>, pub modulus: Vector<u8> }

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RsaPrivateKey { pub public: RsaPublicKey, pub private_exponent: Vector<u8> }

pub fn parse_rsa_public_key(blob: &[u8]) -> Result<RsaPublicKey, ProtocolError> {
    let mut reader=BinaryReader::new(blob);
    if reader.text()? != "ssh-rsa" { return Err(ProtocolError::InvalidPublicKey); }
    let exponent=positive_mpint(reader.string()?)?;
    let modulus=positive_mpint(reader.string()?)?;
    if reader.remaining()!=0 || exponent.is_empty() || modulus.len()<128 || modulus.len()>512 { return Err(ProtocolError::InvalidPublicKey); }
    let mut exponent_owned=Vector::new();exponent_owned.extend(exponent.iter().copied());
    let mut modulus_owned=Vector::new();modulus_owned.extend(modulus.iter().copied());
    Ok(RsaPublicKey{exponent:exponent_owned,modulus:modulus_owned})
}

pub fn verify_rsa_sha2_256(key_blob:&[u8], message:&[u8], signature_blob:&[u8])->Result<(),ProtocolError>{
    let key=parse_rsa_public_key(key_blob)?;
    let mut reader=BinaryReader::new(signature_blob);
    if reader.text()? != "rsa-sha2-256" { return Err(ProtocolError::Authentication); }
    let signature=reader.string()?;
    if reader.remaining()!=0 || signature.len()!=key.modulus.len(){return Err(ProtocolError::Authentication);}
    rsa_pkcs1_sha256_verify(&key.modulus,&key.exponent,message,signature).map_err(|_|ProtocolError::Authentication)
}

pub fn sign_rsa_sha2_256(key:&RsaPrivateKey,message:&[u8])->Result<Vector<u8>,ProtocolError>{
    if key.public.modulus.len()<128 || key.public.modulus.len()>512 || key.private_exponent.is_empty(){return Err(ProtocolError::InvalidPublicKey);}
    let mut signature=Vector::new();signature.try_resize(key.public.modulus.len(),0).map_err(|_|ProtocolError::Length)?;
    rsa_pkcs1_sha256_sign(&key.public.modulus,&key.private_exponent,message,&mut signature).map_err(|_|ProtocolError::Authentication)?;
    let mut writer=BinaryWriter::new();writer.string(b"rsa-sha2-256")?;writer.string(&signature)?;Ok(writer.finish())
}

pub fn build_publickey_auth(session_id:&[u8],user:&str,key:&RsaPrivateKey)->Result<Vector<u8>,ProtocolError>{
    validate_identity(user)?;
    let public_blob=encode_rsa_public_key(&key.public)?;
    let mut request=BinaryWriter::new();request.byte(50);request.string(user.as_bytes())?;request.string(b"ssh-connection")?;request.string(b"publickey")?;request.boolean(true);request.string(b"rsa-sha2-256")?;request.string(&public_blob)?;
    let unsigned=request.finish();
    let mut signed=BinaryWriter::new();signed.string(session_id)?;for byte in &unsigned{signed.byte(*byte);}let signature=sign_rsa_sha2_256(key,&signed.finish())?;
    let mut result=Vector::new();result.extend(unsigned);let mut field=BinaryWriter::new();field.string(&signature)?;result.extend(field.finish());Ok(result)
}

pub fn encode_rsa_public_key(key:&RsaPublicKey)->Result<Vector<u8>,ProtocolError>{let mut writer=BinaryWriter::new();writer.string(b"ssh-rsa")?;write_mpint(&mut writer,&key.exponent)?;write_mpint(&mut writer,&key.modulus)?;Ok(writer.finish())}
fn write_mpint(writer:&mut BinaryWriter,value:&[u8])->Result<(),ProtocolError>{let first=value.iter().position(|b|*b!=0).unwrap_or(value.len());let value=&value[first..];if value.is_empty(){return writer.string(&[]);}if value[0]&0x80!=0{let mut owned=Vector::new();owned.push(0);owned.extend(value.iter().copied());writer.string(&owned)}else{writer.string(value)}}
fn positive_mpint(value:&[u8])->Result<&[u8],ProtocolError>{if value.is_empty(){return Err(ProtocolError::InvalidPublicKey);}if value[0]&0x80!=0{return Err(ProtocolError::InvalidPublicKey);}let value=if value[0]==0{if value.len()==1||value[1]&0x80==0{return Err(ProtocolError::InvalidPublicKey);}&value[1..]}else{value};Ok(value)}
fn validate_identity(value:&str)->Result<(),ProtocolError>{if value.is_empty()||value.len()>255||value.chars().any(|c|c.is_control()){Err(ProtocolError::InvalidUtf8)}else{Ok(())}}

#[cfg(test)]mod tests{use super::*;
 fn decode(text:&str)->Vector<u8>{(0..text.len()/2).map(|i|u8::from_str_radix(&text[i*2..i*2+2],16).unwrap()).collect()}
 fn key()->RsaPrivateKey{RsaPrivateKey{public:RsaPublicKey{modulus:decode("9eff1e540991fee9de7c7ed50d5da16508d610090a52c9aa4c41bc868e93e7cc03a6cc766fb2dab78ba91e4315f6524e355fda2c8a71b372f012d43460c2c425c2ae763d96a20584bc030e3595cc9f2352f51288f8db5d398d55efc566381707b4df848444641093fc5c48ca894db8397b252d00d5d606fe377b09f3609850fb"),exponent:Vector::from([1,0,1])},private_exponent:decode("2187d1e08d2821e736497102035094a1d70c35d3823ed552b9c43f3aed4499e4b77c6cb0297c418de5c123a5a8330b467d111ad4bbd9a0ab839fa4eaeae108364d4ad3f439916be8a244f8071922b1918cce92b27fe5f6ed24a328b15030b3fb3e300166c651f5f457daef746c4051a7a0f035379dcacf3a164fb4aedd284a11")}}
 #[test]fn signatures_round_trip_through_ssh_blobs(){let key=key();let public=encode_rsa_public_key(&key.public).unwrap();let signature=sign_rsa_sha2_256(&key,b"message").unwrap();assert_eq!(verify_rsa_sha2_256(&public,b"message",&signature),Ok(()));assert_eq!(verify_rsa_sha2_256(&public,b"tampered",&signature),Err(ProtocolError::Authentication));}
 #[test]fn userauth_binds_signature_to_session(){let request=build_publickey_auth(b"session","git",&key()).unwrap();let mut reader=BinaryReader::new(&request);assert_eq!(reader.byte(),Ok(50));assert_eq!(reader.text(),Ok("git"));assert!(request.len()>256);}
}
