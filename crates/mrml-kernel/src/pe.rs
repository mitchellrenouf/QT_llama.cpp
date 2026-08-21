use crate::{FrameAllocator, MemoryError, PhysAddr};

const DOS_MAGIC: u16 = 0x5a4d;
const PE_SIGNATURE: u32 = 0x0000_4550;
const MACHINE_X86_64: u16 = 0x8664;
const OPTIONAL_MAGIC_PE32_PLUS: u16 = 0x020b;
const COFF_EXECUTABLE: u16 = 0x0002;
const DLL_NX_COMPAT: u16 = 0x0100;
const SECTION_EXECUTE: u32 = 0x2000_0000;
const SECTION_READ: u32 = 0x4000_0000;
const SECTION_WRITE: u32 = 0x8000_0000;
pub const MAX_PE_SECTIONS: usize = 32;
pub const MAX_PE_IMAGE_BYTES: u32 = 512 * 1024 * 1024;
pub const MAX_PE_RELOCATIONS: usize = 65_536;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PeError {
    Truncated,
    InvalidDosHeader,
    InvalidSignature,
    UnsupportedMachine,
    InvalidCoffHeader,
    InvalidOptionalHeader,
    InvalidAlignment,
    InvalidImageSize,
    TooManySections,
    InvalidSection,
    OverlappingSections,
    WritableExecutableSection,
    EntryPointNotExecutable,
    InvalidDestination,
    UnsupportedImports,
    InvalidRelocations,
    RelocationsRequired,
    RelocationOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PeLoadRegion {
    virtual_address: u32,
    pages: u32,
    readable: bool,
    writable: bool,
    executable: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PeAllocatedRegion {
    load: PeLoadRegion,
    physical_start: PhysAddr,
}

impl PeAllocatedRegion {
    pub const fn load(self) -> PeLoadRegion {
        self.load
    }
    pub const fn physical_start(self) -> PhysAddr {
        self.physical_start
    }
    #[cfg(test)]
    pub(crate) const fn test_value(load: PeLoadRegion, physical_start: PhysAddr) -> Self {
        Self {
            load,
            physical_start,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PeAllocationError {
    InvalidPlan(PeError),
    Memory(MemoryError),
    OutputTooSmall,
}

impl PeLoadRegion {
    #[cfg(test)]
    pub(crate) const fn test_value(
        virtual_address: u32,
        pages: u32,
        writable: bool,
        executable: bool,
    ) -> Self {
        Self {
            virtual_address,
            pages,
            readable: true,
            writable,
            executable,
        }
    }
    pub const fn virtual_address(self) -> u32 {
        self.virtual_address
    }
    pub const fn pages(self) -> u32 {
        self.pages
    }
    pub const fn readable(self) -> bool {
        self.readable
    }
    pub const fn writable(self) -> bool {
        self.writable
    }
    pub const fn executable(self) -> bool {
        self.executable
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PeSection {
    virtual_address: u32,
    virtual_size: u32,
    file_offset: u32,
    file_size: u32,
    characteristics: u32,
}

impl PeSection {
    pub const fn virtual_address(self) -> u32 {
        self.virtual_address
    }
    pub const fn virtual_size(self) -> u32 {
        self.virtual_size
    }
    pub const fn file_offset(self) -> u32 {
        self.file_offset
    }
    pub const fn file_size(self) -> u32 {
        self.file_size
    }
    pub const fn readable(self) -> bool {
        self.characteristics & SECTION_READ != 0
    }
    pub const fn writable(self) -> bool {
        self.characteristics & SECTION_WRITE != 0
    }
    pub const fn executable(self) -> bool {
        self.characteristics & SECTION_EXECUTE != 0
    }
}

pub struct PeImage<'a> {
    encoded: &'a [u8],
    image_base: u64,
    entry_rva: u32,
    image_size: u32,
    headers_size: u32,
    section_table: usize,
    section_count: usize,
    relocation_rva: u32,
    relocation_size: u32,
}

impl<'a> PeImage<'a> {
    pub fn parse(encoded: &'a [u8]) -> Result<Self, PeError> {
        Self::parse_with_limit(encoded, MAX_PE_IMAGE_BYTES)
    }

    pub fn parse_with_limit(encoded: &'a [u8], maximum_image_bytes: u32) -> Result<Self, PeError> {
        if maximum_image_bytes == 0 || maximum_image_bytes > MAX_PE_IMAGE_BYTES {
            return Err(PeError::InvalidImageSize);
        }
        if read_u16(encoded, 0)? != DOS_MAGIC {
            return Err(PeError::InvalidDosHeader);
        }
        let pe = read_u32(encoded, 0x3c)? as usize;
        if pe < 0x40 || read_u32(encoded, pe)? != PE_SIGNATURE {
            return Err(PeError::InvalidSignature);
        }
        if read_u16(encoded, pe + 4)? != MACHINE_X86_64 {
            return Err(PeError::UnsupportedMachine);
        }
        let section_count = read_u16(encoded, pe + 6)? as usize;
        if section_count == 0 || section_count > MAX_PE_SECTIONS {
            return Err(PeError::TooManySections);
        }
        if read_u16(encoded, pe + 22)? & COFF_EXECUTABLE == 0 {
            return Err(PeError::InvalidCoffHeader);
        }
        let optional_size = read_u16(encoded, pe + 20)? as usize;
        if optional_size < 112 {
            return Err(PeError::InvalidOptionalHeader);
        }
        let optional = pe.checked_add(24).ok_or(PeError::Truncated)?;
        let section_table = optional
            .checked_add(optional_size)
            .ok_or(PeError::Truncated)?;
        let table_bytes = section_count.checked_mul(40).ok_or(PeError::Truncated)?;
        let table_end = section_table
            .checked_add(table_bytes)
            .ok_or(PeError::Truncated)?;
        encoded
            .get(section_table..table_end)
            .ok_or(PeError::Truncated)?;
        if read_u16(encoded, optional)? != OPTIONAL_MAGIC_PE32_PLUS
            || read_u16(encoded, optional + 70)? & DLL_NX_COMPAT == 0
        {
            return Err(PeError::InvalidOptionalHeader);
        }
        let entry_rva = read_u32(encoded, optional + 16)?;
        let image_base = read_u64(encoded, optional + 24)?;
        let section_alignment = read_u32(encoded, optional + 32)?;
        let file_alignment = read_u32(encoded, optional + 36)?;
        let image_size = read_u32(encoded, optional + 56)?;
        let headers_size = read_u32(encoded, optional + 60)?;
        let directory_count = read_u32(encoded, optional + 108)?;
        let (import_rva, import_size) =
            directory(encoded, optional, optional_size, directory_count, 1)?;
        if import_rva != 0 || import_size != 0 {
            return Err(PeError::UnsupportedImports);
        }
        let (relocation_rva, relocation_size) =
            directory(encoded, optional, optional_size, directory_count, 5)?;
        if section_alignment < 4096
            || !section_alignment.is_power_of_two()
            || !(512..=65_536).contains(&file_alignment)
            || !file_alignment.is_power_of_two()
            || headers_size == 0
            || headers_size % file_alignment != 0
            || headers_size as usize > encoded.len()
            || table_end > headers_size as usize
        {
            return Err(PeError::InvalidAlignment);
        }
        if entry_rva == 0
            || image_size == 0
            || image_size > maximum_image_bytes
            || encoded.len() > maximum_image_bytes as usize
            || image_size % section_alignment != 0
            || headers_size > image_size
            || image_base % 65_536 != 0
            || image_base.checked_add(image_size as u64).is_none()
        {
            return Err(PeError::InvalidImageSize);
        }
        let image = Self {
            encoded,
            image_base,
            entry_rva,
            image_size,
            headers_size,
            section_table,
            section_count,
            relocation_rva,
            relocation_size,
        };
        let mut entry_is_executable = false;
        for left_index in 0..section_count {
            let left = image.section(left_index)?;
            validate_section(
                left,
                encoded.len(),
                headers_size,
                image_size,
                section_alignment,
                file_alignment,
            )?;
            let left_end = section_end(left)?;
            if left.executable() && entry_rva >= left.virtual_address && entry_rva < left_end {
                entry_is_executable = true;
            }
            for right_index in 0..left_index {
                let right = image.section(right_index)?;
                if overlaps(
                    left.virtual_address,
                    left_end,
                    right.virtual_address,
                    section_end(right)?,
                ) || (left.file_size != 0
                    && right.file_size != 0
                    && overlaps(
                        left.file_offset,
                        file_end(left)?,
                        right.file_offset,
                        file_end(right)?,
                    ))
                {
                    return Err(PeError::OverlappingSections);
                }
            }
        }
        if !entry_is_executable {
            return Err(PeError::EntryPointNotExecutable);
        }
        image.validate_relocations()?;
        Ok(image)
    }

    pub const fn image_base(&self) -> u64 {
        self.image_base
    }
    pub const fn entry_rva(&self) -> u32 {
        self.entry_rva
    }
    pub const fn image_size(&self) -> u32 {
        self.image_size
    }
    pub const fn headers_size(&self) -> u32 {
        self.headers_size
    }
    pub const fn section_count(&self) -> usize {
        self.section_count
    }
    pub const fn load_region_count(&self) -> usize {
        self.section_count + 1
    }
    /// Returns the final mapping plan. Region zero covers PE headers and is
    /// always read-only and non-executable; each later region mirrors one
    /// validated section. The caller must populate memory under NX mappings,
    /// then atomically replace them with these permissions.
    pub fn load_region(&self, index: usize) -> Result<PeLoadRegion, PeError> {
        if index == 0 {
            return Ok(PeLoadRegion {
                virtual_address: 0,
                pages: pages_for(self.headers_size)?,
                readable: true,
                writable: false,
                executable: false,
            });
        }
        let section = self.section(index.checked_sub(1).ok_or(PeError::InvalidSection)?)?;
        Ok(PeLoadRegion {
            virtual_address: section.virtual_address,
            pages: pages_for(section.virtual_size)?,
            readable: section.readable(),
            writable: section.writable(),
            executable: section.executable(),
        })
    }
    /// Reserves physical backing for every header/section run. Any allocation
    /// failure is boot-fatal because the monotonic early allocator deliberately
    /// does not roll back or reuse partially reserved frames.
    pub fn allocate_load_regions(
        &self,
        allocator: &mut FrameAllocator<'_>,
        output: &mut [Option<PeAllocatedRegion>],
    ) -> Result<usize, PeAllocationError> {
        let count = self.load_region_count();
        if output.len() < count {
            return Err(PeAllocationError::OutputTooSmall);
        }
        output.fill(None);
        for (index, slot) in output[..count].iter_mut().enumerate() {
            let load = self
                .load_region(index)
                .map_err(PeAllocationError::InvalidPlan)?;
            let physical_start = allocator
                .allocate_contiguous(load.pages as u64, 1)
                .map_err(PeAllocationError::Memory)?;
            *slot = Some(PeAllocatedRegion {
                load,
                physical_start,
            });
        }
        Ok(count)
    }
    /// Materializes the preferred-base image without allocation. The buffer
    /// must be exactly `SizeOfImage`; exact sizing prevents stale adjacent
    /// bytes from accidentally becoming part of the executable measurement.
    pub fn materialize(&self, destination: &mut [u8]) -> Result<u64, PeError> {
        self.materialize_at(destination, self.image_base)
    }
    pub fn materialize_at(&self, destination: &mut [u8], load_base: u64) -> Result<u64, PeError> {
        if destination.len() != self.image_size as usize {
            return Err(PeError::InvalidDestination);
        }
        // UEFI AllocatePages guarantees page alignment, not the PE section
        // alignment chosen by the linker. Relocation makes any 4 KiB-aligned
        // physical load address valid; requiring 64 KiB rejected legitimate
        // firmware allocations without adding a security property.
        if load_base % 4_096 != 0 || load_base.checked_add(self.image_size as u64).is_none() {
            return Err(PeError::InvalidDestination);
        }
        if load_base != self.image_base && self.relocation_size == 0 {
            return Err(PeError::RelocationsRequired);
        }
        destination.fill(0);
        let headers = self.headers_size as usize;
        destination[..headers].copy_from_slice(&self.encoded[..headers]);
        for index in 0..self.section_count {
            let section = self.section(index)?;
            let source = self.section_data(section)?;
            let start = section.virtual_address as usize;
            let end = start
                .checked_add(source.len())
                .ok_or(PeError::InvalidDestination)?;
            destination
                .get_mut(start..end)
                .ok_or(PeError::InvalidDestination)?
                .copy_from_slice(source);
        }
        if load_base != self.image_base {
            self.apply_relocations(destination, load_base)?;
        }
        load_base
            .checked_add(self.entry_rva as u64)
            .ok_or(PeError::InvalidImageSize)
    }

    fn validate_relocations(&self) -> Result<(), PeError> {
        if (self.relocation_rva == 0) != (self.relocation_size == 0) {
            return Err(PeError::InvalidRelocations);
        }
        if self.relocation_size == 0 {
            return Ok(());
        }
        let bytes = self.rva_file_data(self.relocation_rva, self.relocation_size)?;
        walk_relocations(bytes, self.image_size, |_, _| Ok(()))
    }
    fn apply_relocations(&self, destination: &mut [u8], load_base: u64) -> Result<(), PeError> {
        let bytes = self.rva_file_data(self.relocation_rva, self.relocation_size)?;
        let preferred = self.image_base;
        walk_relocations(bytes, self.image_size, |target, kind| {
            if kind == 0 {
                return Ok(());
            }
            let at = target as usize;
            let value = u64::from_le_bytes(
                destination[at..at + 8]
                    .try_into()
                    .map_err(|_| PeError::InvalidRelocations)?,
            );
            if value == 0 {
                return Ok(());
            }
            let relative = value
                .checked_sub(preferred)
                .filter(|offset| *offset < self.image_size as u64)
                .ok_or(PeError::RelocationOverflow)?;
            let relocated = load_base
                .checked_add(relative)
                .ok_or(PeError::RelocationOverflow)?;
            destination[at..at + 8].copy_from_slice(&relocated.to_le_bytes());
            Ok(())
        })
    }
    fn rva_file_data(&self, rva: u32, size: u32) -> Result<&'a [u8], PeError> {
        let end = rva.checked_add(size).ok_or(PeError::InvalidRelocations)?;
        for index in 0..self.section_count {
            let section = self.section(index)?;
            let raw_end = section
                .virtual_address
                .checked_add(section.file_size)
                .ok_or(PeError::InvalidRelocations)?;
            if rva >= section.virtual_address && end <= raw_end {
                let offset = section
                    .file_offset
                    .checked_add(rva - section.virtual_address)
                    .ok_or(PeError::InvalidRelocations)? as usize;
                return self
                    .encoded
                    .get(offset..offset + size as usize)
                    .ok_or(PeError::InvalidRelocations);
            }
        }
        Err(PeError::InvalidRelocations)
    }
    pub fn section(&self, index: usize) -> Result<PeSection, PeError> {
        if index >= self.section_count {
            return Err(PeError::InvalidSection);
        }
        let at = self
            .section_table
            .checked_add(index.checked_mul(40).ok_or(PeError::Truncated)?)
            .ok_or(PeError::Truncated)?;
        let file_size = read_u32(self.encoded, at + 16)?;
        Ok(PeSection {
            virtual_size: read_u32(self.encoded, at + 8)?.max(file_size),
            virtual_address: read_u32(self.encoded, at + 12)?,
            file_size,
            file_offset: read_u32(self.encoded, at + 20)?,
            characteristics: read_u32(self.encoded, at + 36)?,
        })
    }
    pub fn section_data(&self, section: PeSection) -> Result<&'a [u8], PeError> {
        let start = section.file_offset as usize;
        let end = start
            .checked_add(section.file_size as usize)
            .ok_or(PeError::InvalidSection)?;
        self.encoded.get(start..end).ok_or(PeError::InvalidSection)
    }
}

fn validate_section(
    section: PeSection,
    file_length: usize,
    headers_size: u32,
    image_size: u32,
    section_alignment: u32,
    file_alignment: u32,
) -> Result<(), PeError> {
    if section.virtual_size == 0
        || section.virtual_address < headers_size
        || section.virtual_address % section_alignment != 0
        || section_end(section)? > image_size
        || section.writable() && section.executable()
        || section.executable() && !section.readable()
        || section.writable() && !section.readable()
        || !(section.readable() || section.writable() || section.executable())
    {
        return Err(if section.writable() && section.executable() {
            PeError::WritableExecutableSection
        } else {
            PeError::InvalidSection
        });
    }
    if section.file_size != 0 {
        if section.file_offset < headers_size
            || section.file_offset % file_alignment != 0
            || section.file_size % file_alignment != 0
            || file_end(section)? as usize > file_length
        {
            return Err(PeError::InvalidSection);
        }
    } else if section.file_offset != 0 {
        return Err(PeError::InvalidSection);
    }
    Ok(())
}

fn section_end(section: PeSection) -> Result<u32, PeError> {
    section
        .virtual_address
        .checked_add(section.virtual_size)
        .ok_or(PeError::InvalidSection)
}
fn file_end(section: PeSection) -> Result<u32, PeError> {
    section
        .file_offset
        .checked_add(section.file_size)
        .ok_or(PeError::InvalidSection)
}
fn pages_for(bytes: u32) -> Result<u32, PeError> {
    bytes
        .checked_add(4095)
        .map(|value| value / 4096)
        .filter(|pages| *pages != 0)
        .ok_or(PeError::InvalidSection)
}
fn directory(
    bytes: &[u8],
    optional: usize,
    optional_size: usize,
    count: u32,
    index: usize,
) -> Result<(u32, u32), PeError> {
    if count as usize <= index {
        return Ok((0, 0));
    }
    let at = 112usize
        .checked_add(index.checked_mul(8).ok_or(PeError::InvalidOptionalHeader)?)
        .ok_or(PeError::InvalidOptionalHeader)?;
    if at + 8 > optional_size {
        return Err(PeError::InvalidOptionalHeader);
    }
    Ok((
        read_u32(bytes, optional + at)?,
        read_u32(bytes, optional + at + 4)?,
    ))
}
fn walk_relocations(
    mut bytes: &[u8],
    image_size: u32,
    mut visit: impl FnMut(u32, u16) -> Result<(), PeError>,
) -> Result<(), PeError> {
    let mut count = 0usize;
    while !bytes.is_empty() {
        if bytes.len() < 8 {
            return Err(PeError::InvalidRelocations);
        }
        let page = read_u32(bytes, 0)?;
        let block_size = read_u32(bytes, 4)? as usize;
        if block_size < 8 || block_size > bytes.len() || block_size % 2 != 0 || page % 4096 != 0 {
            return Err(PeError::InvalidRelocations);
        }
        for entry in bytes[8..block_size].chunks_exact(2) {
            count = count.checked_add(1).ok_or(PeError::InvalidRelocations)?;
            if count > MAX_PE_RELOCATIONS {
                return Err(PeError::InvalidRelocations);
            }
            let encoded = u16::from_le_bytes([entry[0], entry[1]]);
            let kind = encoded >> 12;
            if kind != 0 && kind != 10 {
                return Err(PeError::InvalidRelocations);
            }
            let target = page
                .checked_add((encoded & 0x0fff) as u32)
                .ok_or(PeError::InvalidRelocations)?;
            if kind == 10 && target.checked_add(8).is_none_or(|end| end > image_size) {
                return Err(PeError::InvalidRelocations);
            }
            visit(target, kind)?;
        }
        bytes = &bytes[block_size..];
    }
    Ok(())
}
const fn overlaps(a_start: u32, a_end: u32, b_start: u32, b_end: u32) -> bool {
    a_start < b_end && b_start < a_end
}
fn read_u16(bytes: &[u8], at: usize) -> Result<u16, PeError> {
    Ok(u16::from_le_bytes(
        bytes
            .get(at..at.checked_add(2).ok_or(PeError::Truncated)?)
            .ok_or(PeError::Truncated)?
            .try_into()
            .map_err(|_| PeError::Truncated)?,
    ))
}
fn read_u32(bytes: &[u8], at: usize) -> Result<u32, PeError> {
    Ok(u32::from_le_bytes(
        bytes
            .get(at..at.checked_add(4).ok_or(PeError::Truncated)?)
            .ok_or(PeError::Truncated)?
            .try_into()
            .map_err(|_| PeError::Truncated)?,
    ))
}
fn read_u64(bytes: &[u8], at: usize) -> Result<u64, PeError> {
    Ok(u64::from_le_bytes(
        bytes
            .get(at..at.checked_add(8).ok_or(PeError::Truncated)?)
            .ok_or(PeError::Truncated)?
            .try_into()
            .map_err(|_| PeError::Truncated)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryKind, MemoryMap, MemoryRegion};

    fn valid_pe() -> [u8; 1024] {
        let mut pe = [0u8; 1024];
        pe[0..2].copy_from_slice(&DOS_MAGIC.to_le_bytes());
        pe[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        pe[0x80..0x84].copy_from_slice(&PE_SIGNATURE.to_le_bytes());
        pe[0x84..0x86].copy_from_slice(&MACHINE_X86_64.to_le_bytes());
        pe[0x86..0x88].copy_from_slice(&1u16.to_le_bytes());
        pe[0x94..0x96].copy_from_slice(&240u16.to_le_bytes());
        pe[0x96..0x98].copy_from_slice(&COFF_EXECUTABLE.to_le_bytes());
        let optional = 0x98;
        pe[optional..optional + 2].copy_from_slice(&OPTIONAL_MAGIC_PE32_PLUS.to_le_bytes());
        pe[optional + 16..optional + 20].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[optional + 24..optional + 32].copy_from_slice(&0x0000_0001_4000_0000u64.to_le_bytes());
        pe[optional + 32..optional + 36].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[optional + 36..optional + 40].copy_from_slice(&0x200u32.to_le_bytes());
        pe[optional + 56..optional + 60].copy_from_slice(&0x2000u32.to_le_bytes());
        pe[optional + 60..optional + 64].copy_from_slice(&0x200u32.to_le_bytes());
        pe[optional + 70..optional + 72].copy_from_slice(&DLL_NX_COMPAT.to_le_bytes());
        let section = optional + 240;
        pe[section..section + 5].copy_from_slice(b".text");
        pe[section + 8..section + 12].copy_from_slice(&0x20u32.to_le_bytes());
        pe[section + 12..section + 16].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[section + 16..section + 20].copy_from_slice(&0x200u32.to_le_bytes());
        pe[section + 20..section + 24].copy_from_slice(&0x200u32.to_le_bytes());
        pe[section + 36..section + 40]
            .copy_from_slice(&(SECTION_READ | SECTION_EXECUTE).to_le_bytes());
        pe
    }

    fn relocatable_pe() -> [u8; 1536] {
        let mut pe = [0u8; 1536];
        pe[..1024].copy_from_slice(&valid_pe());
        pe[0x86..0x88].copy_from_slice(&2u16.to_le_bytes());
        let optional = 0x98;
        pe[optional + 56..optional + 60].copy_from_slice(&0x3000u32.to_le_bytes());
        pe[optional + 108..optional + 112].copy_from_slice(&6u32.to_le_bytes());
        let directory = optional + 112 + 5 * 8;
        pe[directory..directory + 4].copy_from_slice(&0x2000u32.to_le_bytes());
        pe[directory + 4..directory + 8].copy_from_slice(&12u32.to_le_bytes());
        let section = optional + 240 + 40;
        pe[section..section + 6].copy_from_slice(b".reloc");
        pe[section + 8..section + 12].copy_from_slice(&12u32.to_le_bytes());
        pe[section + 12..section + 16].copy_from_slice(&0x2000u32.to_le_bytes());
        pe[section + 16..section + 20].copy_from_slice(&0x200u32.to_le_bytes());
        pe[section + 20..section + 24].copy_from_slice(&0x400u32.to_le_bytes());
        pe[section + 36..section + 40].copy_from_slice(&SECTION_READ.to_le_bytes());
        pe[0x200..0x208].copy_from_slice(&0x0000_0001_4000_1234u64.to_le_bytes());
        pe[0x400..0x404].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[0x404..0x408].copy_from_slice(&12u32.to_le_bytes());
        pe[0x408..0x40a].copy_from_slice(&0xa000u16.to_le_bytes());
        pe
    }

    #[test]
    fn accepts_bounded_x86_64_pe32_plus() {
        let pe = valid_pe();
        let image = PeImage::parse(&pe).unwrap();
        assert_eq!(image.entry_rva(), 0x1000);
        assert_eq!(image.image_size(), 0x2000);
        assert_eq!(
            image.section_data(image.section(0).unwrap()).unwrap().len(),
            0x200
        );
        assert_eq!(
            PeImage::parse_with_limit(&pe, 4096).err(),
            Some(PeError::InvalidImageSize)
        );
        assert!(PeImage::parse_with_limit(&pe, 8192).is_ok());
    }

    #[test]
    fn rejects_writable_code_and_non_executable_entry() {
        let mut writable = valid_pe();
        let characteristics = 0x98 + 240 + 36;
        writable[characteristics..characteristics + 4]
            .copy_from_slice(&(SECTION_READ | SECTION_WRITE | SECTION_EXECUTE).to_le_bytes());
        assert_eq!(
            PeImage::parse(&writable).err(),
            Some(PeError::WritableExecutableSection)
        );
        let mut no_entry = valid_pe();
        no_entry[0xa8..0xac].copy_from_slice(&0x1800u32.to_le_bytes());
        assert_eq!(
            PeImage::parse(&no_entry).err(),
            Some(PeError::EntryPointNotExecutable)
        );
    }

    #[test]
    fn rejects_truncation_and_trailing_file_ranges() {
        assert_eq!(
            PeImage::parse(&valid_pe()[..100]).err(),
            Some(PeError::Truncated)
        );
        let mut outside = valid_pe();
        let raw_size = 0x98 + 240 + 16;
        outside[raw_size..raw_size + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        assert_eq!(
            PeImage::parse(&outside).err(),
            Some(PeError::InvalidSection)
        );
    }

    #[test]
    fn materializes_zero_filled_image_and_wx_plan() {
        let encoded = valid_pe();
        let image = PeImage::parse(&encoded).unwrap();
        let mut loaded = [0xa5; 0x2000];
        let entry = image.materialize(&mut loaded).unwrap();
        assert_eq!(entry, 0x0000_0001_4000_1000);
        assert_eq!(&loaded[..0x200], &encoded[..0x200]);
        assert_eq!(&loaded[0x1000..0x1200], &encoded[0x200..0x400]);
        assert!(loaded[0x1200..].iter().all(|byte| *byte == 0));
        let headers = image.load_region(0).unwrap();
        assert_eq!(
            (headers.pages(), headers.writable(), headers.executable()),
            (1, false, false)
        );
        let code = image.load_region(1).unwrap();
        assert_eq!(
            (code.pages(), code.writable(), code.executable()),
            (1, false, true)
        );
        assert_eq!(
            image.materialize(&mut loaded[..4096]),
            Err(PeError::InvalidDestination)
        );
    }

    #[test]
    fn applies_only_bounded_dir64_relocations() {
        let encoded = relocatable_pe();
        let image = PeImage::parse(&encoded).unwrap();
        let mut loaded = [0u8; 0x3000];
        assert_eq!(
            image.materialize_at(&mut loaded, 0x0000_0001_4001_1000),
            Ok(0x0000_0001_4001_2000)
        );
        assert_eq!(
            u64::from_le_bytes(loaded[0x1000..0x1008].try_into().unwrap()),
            0x0000_0001_4001_2234
        );
        let fixed_encoded = valid_pe();
        let fixed = PeImage::parse(&fixed_encoded).unwrap();
        assert_eq!(
            fixed.materialize_at(&mut loaded[..0x2000], 0x0000_0001_4001_0000),
            Err(PeError::RelocationsRequired)
        );
    }

    #[test]
    fn allocates_one_contiguous_run_per_load_region() {
        let encoded = valid_pe();
        let image = PeImage::parse(&encoded).unwrap();
        let memory = [MemoryRegion::new(PhysAddr::new(0).unwrap(), 4, MemoryKind::Free).unwrap()];
        let mut allocator = FrameAllocator::new(MemoryMap::new(&memory).unwrap());
        let mut output = [None; MAX_PE_SECTIONS + 1];
        assert_eq!(
            image.allocate_load_regions(&mut allocator, &mut output),
            Ok(2)
        );
        assert_eq!(output[0].unwrap().physical_start().get(), 0);
        assert_eq!(output[1].unwrap().physical_start().get(), 0x1000);
        assert_eq!(
            image.allocate_load_regions(&mut allocator, &mut output[..1]),
            Err(PeAllocationError::OutputTooSmall)
        );
    }
}
