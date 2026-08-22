use crate::MachineCode;

const ELF_HEADER_BYTES: usize = 64;
const ELF_SECTION_HEADER_BYTES: usize = 64;
const ELF_SYMBOL_BYTES: usize = 24;
const ELF_SECTION_COUNT: usize = 5;
const ELF_SECTION_NAMES: &[u8] = b"\0.text\0.symtab\0.strtab\0.shstrtab\0";

const COFF_HEADER_BYTES: usize = 20;
const COFF_SECTION_HEADER_BYTES: usize = 40;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectError {
    EmptySymbol,
    SymbolContainsNul,
    OutputTooSmall,
    OffsetOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectFile<const MAX_BYTES: usize> {
    bytes: [u8; MAX_BYTES],
    length: usize,
}

impl<const MAX_BYTES: usize> ObjectFile<MAX_BYTES> {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.length]
    }

    pub const fn len(&self) -> usize {
        self.length
    }

    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }
}

struct Writer<const MAX_BYTES: usize> {
    bytes: [u8; MAX_BYTES],
    position: usize,
}

impl<const MAX_BYTES: usize> Writer<MAX_BYTES> {
    const fn new() -> Self {
        Self {
            bytes: [0; MAX_BYTES],
            position: 0,
        }
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), ObjectError> {
        let end = self
            .position
            .checked_add(bytes.len())
            .ok_or(ObjectError::OffsetOverflow)?;
        let output = self
            .bytes
            .get_mut(self.position..end)
            .ok_or(ObjectError::OutputTooSmall)?;
        output.copy_from_slice(bytes);
        self.position = end;
        Ok(())
    }

    fn zeros(&mut self, length: usize) -> Result<(), ObjectError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or(ObjectError::OffsetOverflow)?;
        self.bytes
            .get(self.position..end)
            .ok_or(ObjectError::OutputTooSmall)?;
        self.position = end;
        Ok(())
    }

    fn align(&mut self, alignment: usize) -> Result<(), ObjectError> {
        let remainder = self.position % alignment;
        if remainder == 0 {
            Ok(())
        } else {
            self.zeros(alignment - remainder)
        }
    }

    fn u16(&mut self, value: u16) -> Result<(), ObjectError> {
        self.write(&value.to_le_bytes())
    }

    fn i16(&mut self, value: i16) -> Result<(), ObjectError> {
        self.write(&value.to_le_bytes())
    }

    fn u32(&mut self, value: u32) -> Result<(), ObjectError> {
        self.write(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> Result<(), ObjectError> {
        self.write(&value.to_le_bytes())
    }

    fn patch_u64(&mut self, offset: usize, value: u64) -> Result<(), ObjectError> {
        let end = offset.checked_add(8).ok_or(ObjectError::OffsetOverflow)?;
        self.bytes
            .get_mut(offset..end)
            .ok_or(ObjectError::OutputTooSmall)?
            .copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn finish(self) -> ObjectFile<MAX_BYTES> {
        ObjectFile {
            bytes: self.bytes,
            length: self.position,
        }
    }
}

fn validate_symbol(symbol: &str) -> Result<(), ObjectError> {
    if symbol.is_empty() {
        Err(ObjectError::EmptySymbol)
    } else if symbol.as_bytes().contains(&0) {
        Err(ObjectError::SymbolContainsNul)
    } else {
        Ok(())
    }
}

pub fn emit_elf64_x86_64<const MAX_OBJECT_BYTES: usize, const MAX_CODE_BYTES: usize>(
    symbol: &str,
    code: &MachineCode<MAX_CODE_BYTES>,
) -> Result<ObjectFile<MAX_OBJECT_BYTES>, ObjectError> {
    validate_symbol(symbol)?;
    let mut writer = Writer::new();

    writer.write(b"\x7fELF")?;
    writer.write(&[2, 1, 1, 0, 0])?;
    writer.zeros(7)?;
    writer.u16(1)?;
    writer.u16(62)?;
    writer.u32(1)?;
    writer.u64(0)?;
    writer.u64(0)?;
    writer.u64(0)?;
    writer.u32(0)?;
    writer.u16(ELF_HEADER_BYTES as u16)?;
    writer.u16(0)?;
    writer.u16(0)?;
    writer.u16(ELF_SECTION_HEADER_BYTES as u16)?;
    writer.u16(ELF_SECTION_COUNT as u16)?;
    writer.u16(4)?;

    let text_offset = writer.position;
    writer.write(code.bytes())?;
    writer.align(8)?;
    let symbol_offset = writer.position;
    writer.zeros(ELF_SYMBOL_BYTES)?;
    writer.u32(1)?;
    writer.write(&[0x12, 0])?;
    writer.u16(1)?;
    writer.u64(0)?;
    writer.u64(to_u64(code.len())?)?;
    let string_offset = writer.position;
    writer.write(&[0])?;
    writer.write(symbol.as_bytes())?;
    writer.write(&[0])?;
    let string_size = writer.position - string_offset;
    let section_name_offset = writer.position;
    writer.write(ELF_SECTION_NAMES)?;
    let section_name_size = writer.position - section_name_offset;
    writer.align(8)?;
    let section_headers_offset = writer.position;
    writer.patch_u64(40, to_u64(section_headers_offset)?)?;

    writer.zeros(ELF_SECTION_HEADER_BYTES)?;
    elf_section(&mut writer, 1, 1, 6, text_offset, code.len(), 0, 0, 16, 0)?;
    elf_section(
        &mut writer,
        7,
        2,
        0,
        symbol_offset,
        ELF_SYMBOL_BYTES * 2,
        3,
        1,
        8,
        ELF_SYMBOL_BYTES,
    )?;
    elf_section(
        &mut writer,
        15,
        3,
        0,
        string_offset,
        string_size,
        0,
        0,
        1,
        0,
    )?;
    elf_section(
        &mut writer,
        23,
        3,
        0,
        section_name_offset,
        section_name_size,
        0,
        0,
        1,
        0,
    )?;
    Ok(writer.finish())
}

#[allow(clippy::too_many_arguments)]
fn elf_section<const MAX_BYTES: usize>(
    writer: &mut Writer<MAX_BYTES>,
    name: u32,
    section_type: u32,
    flags: u64,
    offset: usize,
    size: usize,
    link: u32,
    info: u32,
    alignment: u64,
    entry_size: usize,
) -> Result<(), ObjectError> {
    writer.u32(name)?;
    writer.u32(section_type)?;
    writer.u64(flags)?;
    writer.u64(0)?;
    writer.u64(to_u64(offset)?)?;
    writer.u64(to_u64(size)?)?;
    writer.u32(link)?;
    writer.u32(info)?;
    writer.u64(alignment)?;
    writer.u64(to_u64(entry_size)?)
}

pub fn emit_x86_64_coff<const MAX_OBJECT_BYTES: usize, const MAX_CODE_BYTES: usize>(
    symbol: &str,
    code: &MachineCode<MAX_CODE_BYTES>,
) -> Result<ObjectFile<MAX_OBJECT_BYTES>, ObjectError> {
    validate_symbol(symbol)?;
    let raw_data_offset = COFF_HEADER_BYTES + COFF_SECTION_HEADER_BYTES;
    let symbol_offset = raw_data_offset
        .checked_add(code.len())
        .ok_or(ObjectError::OffsetOverflow)?;
    let mut writer = Writer::new();

    writer.u16(0x8664)?;
    writer.u16(1)?;
    writer.u32(0)?;
    writer.u32(to_u32(symbol_offset)?)?;
    writer.u32(1)?;
    writer.u16(0)?;
    writer.u16(0)?;

    writer.write(b".text\0\0\0")?;
    writer.u32(0)?;
    writer.u32(0)?;
    writer.u32(to_u32(code.len())?)?;
    writer.u32(to_u32(raw_data_offset)?)?;
    writer.u32(0)?;
    writer.u32(0)?;
    writer.u16(0)?;
    writer.u16(0)?;
    writer.u32(0x6000_0020)?;
    writer.write(code.bytes())?;

    if symbol.len() <= 8 {
        writer.write(symbol.as_bytes())?;
        writer.zeros(8 - symbol.len())?;
    } else {
        writer.u32(0)?;
        writer.u32(4)?;
    }
    writer.u32(0)?;
    writer.i16(1)?;
    writer.u16(0x20)?;
    writer.write(&[2, 0])?;
    if symbol.len() <= 8 {
        writer.u32(4)?;
    } else {
        let string_table_size = 4usize
            .checked_add(symbol.len())
            .and_then(|size| size.checked_add(1))
            .ok_or(ObjectError::OffsetOverflow)?;
        writer.u32(to_u32(string_table_size)?)?;
        writer.write(symbol.as_bytes())?;
        writer.write(&[0])?;
    }
    Ok(writer.finish())
}

fn to_u32(value: usize) -> Result<u32, ObjectError> {
    u32::try_from(value).map_err(|_| ObjectError::OffsetOverflow)
}

fn to_u64(value: usize) -> Result<u64, ObjectError> {
    u64::try_from(value).map_err(|_| ObjectError::OffsetOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Item, NoConstants, Parser, compile_x86_64_constant_function};

    fn code() -> MachineCode<16> {
        let module = Parser::new("#[unsafe(no_mangle)] pub extern \"C\" fn answer() -> u64 { 42 }")
            .parse_module::<2, 2>()
            .unwrap();
        let Some(Item::Function(function)) = module.items()[0] else {
            panic!("expected function")
        };
        compile_x86_64_constant_function::<_, 16, 2, 8>(&function, &NoConstants).unwrap()
    }

    fn u16_at(bytes: &[u8], offset: usize) -> u16 {
        u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
    }

    fn u32_at(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    fn u64_at(bytes: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    }

    #[test]
    fn emits_structurally_complete_elf64_relocatable_object() {
        let code = code();
        let object = emit_elf64_x86_64::<512, 16>("answer", &code).unwrap();
        let bytes = object.bytes();
        assert_eq!(&bytes[..9], b"\x7fELF\x02\x01\x01\x00\x00");
        assert_eq!(u16_at(bytes, 16), 1);
        assert_eq!(u16_at(bytes, 18), 62);
        assert_eq!(u16_at(bytes, 60), 5);
        assert_eq!(u16_at(bytes, 62), 4);
        assert_eq!(&bytes[64..75], code.bytes());
        let sections = u64_at(bytes, 40) as usize;
        assert_eq!(sections + 5 * ELF_SECTION_HEADER_BYTES, object.len());
        assert_eq!(u32_at(bytes, sections + ELF_SECTION_HEADER_BYTES), 1);
        assert_eq!(u64_at(bytes, sections + ELF_SECTION_HEADER_BYTES + 32), 11);
    }

    #[test]
    fn emits_coff_with_short_and_long_external_symbols() {
        let code = code();
        let short = emit_x86_64_coff::<256, 16>("answer", &code).unwrap();
        let bytes = short.bytes();
        assert_eq!(u16_at(bytes, 0), 0x8664);
        assert_eq!(u16_at(bytes, 2), 1);
        assert_eq!(u32_at(bytes, 8), 71);
        assert_eq!(&bytes[60..71], code.bytes());
        assert_eq!(&bytes[71..77], b"answer");
        assert_eq!(u32_at(bytes, 89), 4);

        let long = emit_x86_64_coff::<256, 16>("answer_to_everything", &code).unwrap();
        let bytes = long.bytes();
        assert_eq!(u32_at(bytes, 71), 0);
        assert_eq!(u32_at(bytes, 75), 4);
        assert_eq!(u32_at(bytes, 89), 25);
        assert_eq!(&bytes[93..113], b"answer_to_everything");
        assert_eq!(bytes[113], 0);
    }

    #[test]
    fn object_emission_is_deterministic_and_bounded() {
        let code = code();
        let first = emit_elf64_x86_64::<512, 16>("answer", &code).unwrap();
        let second = emit_elf64_x86_64::<512, 16>("answer", &code).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            emit_elf64_x86_64::<64, 16>("answer", &code),
            Err(ObjectError::OutputTooSmall)
        );
        assert_eq!(
            emit_x86_64_coff::<32, 16>("answer", &code),
            Err(ObjectError::OutputTooSmall)
        );
    }

    #[test]
    fn rejects_names_that_cannot_enter_a_string_table() {
        let code = code();
        assert_eq!(
            emit_elf64_x86_64::<512, 16>("", &code),
            Err(ObjectError::EmptySymbol)
        );
        assert_eq!(
            emit_x86_64_coff::<256, 16>("bad\0name", &code),
            Err(ObjectError::SymbolContainsNul)
        );
    }
}
