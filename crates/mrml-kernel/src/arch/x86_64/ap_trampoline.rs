use super::PAGE_SIZE;

const PAGE_BYTES: usize = PAGE_SIZE as usize;
const LONG_MODE_OFFSET: usize = 80;
const GDTR_OFFSET: usize = 160;
const GDT_OFFSET: usize = 168;
const GDT_CODE_SELECTOR: u16 = 8;
const GDT_DATA_SELECTOR: u16 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApTrampolineError {
    InvalidPhysicalPage,
    InvalidPageTable,
    InvalidEntry,
    InvalidStack,
    InvalidCpu,
}

/// One sealed 4 KiB INIT/SIPI target. Firmware or the BSP writes this page
/// while NX, then remaps it read/execute before publishing the SIPI. The page
/// contains no writable runtime state; each AP receives its CPU index in RDI.
#[repr(C, align(4096))]
pub struct ApTrampolineImage {
    bytes: [u8; PAGE_BYTES],
    physical: u64,
}

impl ApTrampolineImage {
    pub fn new(
        physical: u64,
        page_table: u64,
        entry: u64,
        stack_top: u64,
        cpu_index: usize,
    ) -> Result<Self, ApTrampolineError> {
        if !(PAGE_SIZE..0x10_0000).contains(&physical) || !physical.is_multiple_of(PAGE_SIZE) {
            return Err(ApTrampolineError::InvalidPhysicalPage);
        }
        if page_table == 0
            || page_table > u64::from(u32::MAX)
            || !page_table.is_multiple_of(PAGE_SIZE)
        {
            return Err(ApTrampolineError::InvalidPageTable);
        }
        if entry == 0 || !canonical(entry) {
            return Err(ApTrampolineError::InvalidEntry);
        }
        if stack_top < 16 || !canonical(stack_top) || !stack_top.is_multiple_of(16) {
            return Err(ApTrampolineError::InvalidStack);
        }
        let cpu = u64::try_from(cpu_index)
            .ok()
            .filter(|value| *value < 256)
            .ok_or(ApTrampolineError::InvalidCpu)?;

        let mut bytes = [0u8; PAGE_BYTES];
        let mut cursor = 0usize;
        emit(&mut bytes, &mut cursor, &[0xfa, 0xfc]); // cli; cld
        emit(&mut bytes, &mut cursor, &[0x8c, 0xc8]); // mov ax, cs
        emit(
            &mut bytes,
            &mut cursor,
            &[0x8e, 0xd8, 0x8e, 0xc0, 0x8e, 0xd0],
        );
        emit(&mut bytes, &mut cursor, &[0xbc, 0xf0, 0x0f]); // mov sp, 0xff0
        emit(&mut bytes, &mut cursor, &[0x66, 0x0f, 0x01, 0x16]); // lgdt [disp16]
        emit(&mut bytes, &mut cursor, &(GDTR_OFFSET as u16).to_le_bytes());
        emit(&mut bytes, &mut cursor, &[0x0f, 0x20, 0xe0]); // mov eax, cr4
        emit(
            &mut bytes,
            &mut cursor,
            &[0x66, 0x0d, 0x20, 0x00, 0x00, 0x00],
        ); // PAE
        emit(&mut bytes, &mut cursor, &[0x0f, 0x22, 0xe0]); // mov cr4, eax
        emit(&mut bytes, &mut cursor, &[0x66, 0xb8]); // mov eax, cr3 value
        emit(&mut bytes, &mut cursor, &(page_table as u32).to_le_bytes());
        emit(&mut bytes, &mut cursor, &[0x0f, 0x22, 0xd8]); // mov cr3, eax
        emit(
            &mut bytes,
            &mut cursor,
            &[0x66, 0xb9, 0x80, 0x00, 0x00, 0xc0],
        ); // EFER
        emit(&mut bytes, &mut cursor, &[0x0f, 0x32]); // rdmsr
        emit(
            &mut bytes,
            &mut cursor,
            &[0x66, 0x0d, 0x00, 0x01, 0x00, 0x00],
        ); // LME
        emit(&mut bytes, &mut cursor, &[0x0f, 0x30]); // wrmsr
        emit(&mut bytes, &mut cursor, &[0x0f, 0x20, 0xc0]); // mov eax, cr0
        emit(
            &mut bytes,
            &mut cursor,
            &[0x66, 0x0d, 0x01, 0x00, 0x00, 0x80],
        ); // PE|PG
        emit(&mut bytes, &mut cursor, &[0x0f, 0x22, 0xc0]); // mov cr0, eax
        emit(&mut bytes, &mut cursor, &[0x66, 0xea]); // far jmp ptr16:32
        let long_physical = physical + LONG_MODE_OFFSET as u64;
        emit(
            &mut bytes,
            &mut cursor,
            &(long_physical as u32).to_le_bytes(),
        );
        emit(&mut bytes, &mut cursor, &GDT_CODE_SELECTOR.to_le_bytes());
        while cursor < LONG_MODE_OFFSET {
            bytes[cursor] = 0x90;
            cursor += 1;
        }

        emit(&mut bytes, &mut cursor, &[0x66, 0xb8]);
        emit(&mut bytes, &mut cursor, &GDT_DATA_SELECTOR.to_le_bytes());
        emit(
            &mut bytes,
            &mut cursor,
            &[0x8e, 0xd8, 0x8e, 0xc0, 0x8e, 0xd0],
        );
        emit(&mut bytes, &mut cursor, &[0x48, 0xbc]);
        emit(&mut bytes, &mut cursor, &stack_top.to_le_bytes());
        emit(&mut bytes, &mut cursor, &[0x48, 0xbf]);
        emit(&mut bytes, &mut cursor, &cpu.to_le_bytes());
        emit(&mut bytes, &mut cursor, &[0x48, 0xb8]);
        emit(&mut bytes, &mut cursor, &entry.to_le_bytes());
        emit(&mut bytes, &mut cursor, &[0xff, 0xe0]);

        bytes[GDTR_OFFSET..GDTR_OFFSET + 2].copy_from_slice(&23u16.to_le_bytes());
        bytes[GDTR_OFFSET + 2..GDTR_OFFSET + 6]
            .copy_from_slice(&((physical + GDT_OFFSET as u64) as u32).to_le_bytes());
        bytes[GDT_OFFSET + 8..GDT_OFFSET + 16]
            .copy_from_slice(&0x00af_9a00_0000_ffffu64.to_le_bytes());
        bytes[GDT_OFFSET + 16..GDT_OFFSET + 24]
            .copy_from_slice(&0x00cf_9200_0000_ffffu64.to_le_bytes());
        Ok(Self { bytes, physical })
    }

    pub const fn physical(&self) -> u64 {
        self.physical
    }

    pub const fn startup_vector(&self) -> u8 {
        (self.physical >> 12) as u8
    }

    pub fn bytes(&self) -> &[u8; PAGE_BYTES] {
        &self.bytes
    }
}

fn emit(output: &mut [u8; PAGE_BYTES], cursor: &mut usize, bytes: &[u8]) {
    let end = *cursor + bytes.len();
    output[*cursor..end].copy_from_slice(bytes);
    *cursor = end;
}

const fn canonical(address: u64) -> bool {
    let high = address >> 48;
    let sign = (address >> 47) & 1;
    (sign == 0 && high == 0) || (sign == 1 && high == 0xffff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_binds_page_table_entry_stack_and_cpu() {
        let image = ApTrampolineImage::new(
            0x8000,
            0x20_0000,
            0xffff_8000_0010_0000,
            0xffff_8000_0021_0000,
            7,
        )
        .unwrap();
        assert_eq!(image.startup_vector(), 8);
        assert_eq!(&image.bytes()[33..37], &0x20_0000u32.to_le_bytes());
        assert_eq!(
            &image.bytes()[92..100],
            &0xffff_8000_0021_0000u64.to_le_bytes()
        );
        assert_eq!(&image.bytes()[102..110], &7u64.to_le_bytes());
        assert_eq!(
            &image.bytes()[112..120],
            &0xffff_8000_0010_0000u64.to_le_bytes()
        );
        assert_eq!(
            &image.bytes()[GDT_OFFSET + 8..GDT_OFFSET + 16],
            &0x00af_9a00_0000_ffffu64.to_le_bytes()
        );
    }

    #[test]
    fn rejects_alias_prone_or_unbootable_inputs() {
        assert_eq!(
            ApTrampolineImage::new(0, 0x2000, 0x1000, 0x2000, 0).err(),
            Some(ApTrampolineError::InvalidPhysicalPage)
        );
        assert_eq!(
            ApTrampolineImage::new(0x8000, 0x2001, 0x1000, 0x2000, 0).err(),
            Some(ApTrampolineError::InvalidPageTable)
        );
        assert_eq!(
            ApTrampolineImage::new(0x8000, 0x2000, 0x0000_8000_0000_0000, 0x2000, 0).err(),
            Some(ApTrampolineError::InvalidEntry)
        );
        assert_eq!(
            ApTrampolineImage::new(0x8000, 0x2000, 0x1000, 0x2008, 0).err(),
            Some(ApTrampolineError::InvalidStack)
        );
        assert_eq!(
            ApTrampolineImage::new(0x8000, 0x2000, 0x1000, 0x2000, 256).err(),
            Some(ApTrampolineError::InvalidCpu)
        );
    }
}
