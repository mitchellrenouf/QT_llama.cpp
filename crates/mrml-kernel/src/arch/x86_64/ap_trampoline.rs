use core::sync::atomic::{Ordering, compiler_fence};

use super::{
    ActivePageTables, EARLY_STACK_PAGES, Mapping, PAGE_SIZE, PagePermissions, PhysAddr, VirtAddr,
};

const PAGE_BYTES: usize = PAGE_SIZE as usize;
const LONG_MODE_OFFSET: usize = 80;
const GDTR_OFFSET: usize = 160;
const GDT_OFFSET: usize = 168;
const GDT_CODE_SELECTOR: u16 = 8;
const GDT_DATA_SELECTOR: u16 = 16;
const EARLY_STACK_BYTES: u64 = EARLY_STACK_PAGES * PAGE_SIZE;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApTrampolineError {
    InvalidPhysicalPage,
    InvalidPageTable,
    InvalidEntry,
    InvalidStack,
    InvalidCpu,
    InvalidGeneration,
    UnsafePermissions,
    WriteFailed,
    ProtectionFailed,
    RevocationFailed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrampolinePermissions {
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
}

pub trait ApTrampolinePage {
    fn permissions(&self, physical: u64) -> Option<TrampolinePermissions>;
    fn write_page(&mut self, physical: u64, bytes: &[u8; PAGE_BYTES]) -> bool;
    fn protect_read_execute(&mut self, physical: u64) -> bool;
    fn rearm_read_write_and_zero(&mut self, physical: u64) -> bool;
    fn revoke_and_zero(&mut self, physical: u64) -> bool;
}

pub struct ActiveApTrampolinePage {
    tables: ActivePageTables,
    physical: u64,
}

impl ActiveApTrampolinePage {
    /// Opens the current root for one identity-mapped low trampoline page.
    ///
    /// # Safety
    ///
    /// The safety contract of [`ActivePageTables::current`] applies, and
    /// `physical` must name an exclusively owned page whose initial active leaf
    /// is supervisor read/write and NX.
    pub unsafe fn current(physical: u64) -> Result<Self, ApTrampolineError> {
        if !(PAGE_SIZE..0x10_0000).contains(&physical) || !physical.is_multiple_of(PAGE_SIZE) {
            return Err(ApTrampolineError::InvalidPhysicalPage);
        }
        let tables = unsafe { ActivePageTables::current() }
            .map_err(|_| ApTrampolineError::UnsafePermissions)?;
        Ok(Self { tables, physical })
    }

    fn mapping(&self, permissions: PagePermissions) -> Option<Mapping> {
        Mapping::new(
            VirtAddr::new(self.physical).ok()?,
            PhysAddr::new(self.physical).ok()?,
            1,
            permissions,
        )
        .ok()
    }
}

impl ApTrampolinePage for ActiveApTrampolinePage {
    fn permissions(&self, physical: u64) -> Option<TrampolinePermissions> {
        if physical != self.physical {
            return None;
        }
        let leaf = self.tables.leaf(VirtAddr::new(physical).ok()?).ok()??;
        if leaf.physical().get() != physical || leaf.user() {
            return None;
        }
        Some(TrampolinePermissions {
            readable: true,
            writable: leaf.writable(),
            executable: leaf.executable(),
        })
    }

    fn write_page(&mut self, physical: u64, bytes: &[u8; PAGE_BYTES]) -> bool {
        if self.permissions(physical)
            != Some(TrampolinePermissions {
                readable: true,
                writable: true,
                executable: false,
            })
        {
            return false;
        }
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), physical as *mut u8, PAGE_BYTES) };
        compiler_fence(Ordering::SeqCst);
        true
    }

    fn protect_read_execute(&mut self, physical: u64) -> bool {
        if physical != self.physical {
            return false;
        }
        let Some(expected) = self.mapping(PagePermissions::KERNEL_LOW_READ_WRITE) else {
            return false;
        };
        unsafe {
            self.tables
                .protect(expected, PagePermissions::KERNEL_LOW_READ_EXECUTE)
                .is_ok()
        }
    }

    fn revoke_and_zero(&mut self, physical: u64) -> bool {
        if physical != self.physical {
            return false;
        }
        let permissions = self.permissions(physical);
        if permissions
            == Some(TrampolinePermissions {
                readable: true,
                writable: false,
                executable: true,
            })
        {
            let Some(executable) = self.mapping(PagePermissions::KERNEL_LOW_READ_EXECUTE) else {
                return false;
            };
            if unsafe {
                self.tables
                    .protect(executable, PagePermissions::KERNEL_LOW_READ_WRITE)
            }
            .is_err()
            {
                return false;
            }
        } else if permissions
            != Some(TrampolinePermissions {
                readable: true,
                writable: true,
                executable: false,
            })
        {
            return false;
        }
        unsafe { core::ptr::write_bytes(physical as *mut u8, 0, PAGE_BYTES) };
        compiler_fence(Ordering::SeqCst);
        let Some(writable) = self.mapping(PagePermissions::KERNEL_LOW_READ_WRITE) else {
            return false;
        };
        if unsafe { self.tables.unmap_exact(writable) }.is_err() {
            return false;
        }
        let Ok(address) = VirtAddr::new(physical) else {
            return false;
        };
        self.tables.leaf(address).is_ok_and(|leaf| leaf.is_none())
    }

    fn rearm_read_write_and_zero(&mut self, physical: u64) -> bool {
        if physical != self.physical
            || self.permissions(physical)
                != Some(TrampolinePermissions {
                    readable: true,
                    writable: false,
                    executable: true,
                })
        {
            return false;
        }
        let Some(executable) = self.mapping(PagePermissions::KERNEL_LOW_READ_EXECUTE) else {
            return false;
        };
        if unsafe {
            self.tables
                .protect(executable, PagePermissions::KERNEL_LOW_READ_WRITE)
        }
        .is_err()
        {
            return false;
        }
        unsafe { core::ptr::write_bytes(physical as *mut u8, 0, PAGE_BYTES) };
        compiler_fence(Ordering::SeqCst);
        self.permissions(physical)
            == Some(TrampolinePermissions {
                readable: true,
                writable: true,
                executable: false,
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstalledApTrampoline {
    physical: u64,
    startup_vector: u8,
}

impl InstalledApTrampoline {
    pub const fn physical(self) -> u64 {
        self.physical
    }

    pub const fn startup_vector(self) -> u8 {
        self.startup_vector
    }

    /// Returns a successfully used SIPI page to a zeroed RW/NX staging state
    /// so the BSP can bind the same low page to the next AP's private stack.
    /// Callers must wait for the preceding AP to leave the trampoline first.
    pub fn rearm<P: ApTrampolinePage>(self, page: &mut P) -> Result<(), ApTrampolineError> {
        if !page.rearm_read_write_and_zero(self.physical) {
            return Err(ApTrampolineError::RevocationFailed);
        }
        if page.permissions(self.physical)
            != Some(TrampolinePermissions {
                readable: true,
                writable: true,
                executable: false,
            })
        {
            return Err(ApTrampolineError::UnsafePermissions);
        }
        Ok(())
    }
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
        generation: u32,
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
        if stack_top < 16 || !canonical(stack_top) || !stack_top.wrapping_add(8).is_multiple_of(16)
        {
            return Err(ApTrampolineError::InvalidStack);
        }
        let stack_base = stack_top
            .checked_add(8)
            .and_then(|end| end.checked_sub(EARLY_STACK_BYTES))
            .filter(|base| *base != 0 && base.is_multiple_of(PAGE_SIZE))
            .ok_or(ApTrampolineError::InvalidStack)?;
        let cpu = u64::try_from(cpu_index)
            .ok()
            .filter(|value| *value < 256)
            .ok_or(ApTrampolineError::InvalidCpu)?;
        if generation == 0 {
            return Err(ApTrampolineError::InvalidGeneration);
        }

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
            &[0x66, 0x0d, 0x00, 0x09, 0x00, 0x00],
        ); // LME | NXE
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
        emit(&mut bytes, &mut cursor, &[0x48, 0xbe]);
        emit(
            &mut bytes,
            &mut cursor,
            &u64::from(generation).to_le_bytes(),
        );
        emit(&mut bytes, &mut cursor, &[0x48, 0xba]);
        emit(&mut bytes, &mut cursor, &stack_base.to_le_bytes());
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

    pub fn install<P: ApTrampolinePage>(
        &self,
        page: &mut P,
    ) -> Result<InstalledApTrampoline, ApTrampolineError> {
        let initial = page
            .permissions(self.physical)
            .ok_or(ApTrampolineError::UnsafePermissions)?;
        if !initial.readable || !initial.writable || initial.executable {
            return Err(ApTrampolineError::UnsafePermissions);
        }
        if !page.write_page(self.physical, &self.bytes) {
            return revoke_after(page, self.physical, ApTrampolineError::WriteFailed);
        }
        if !page.protect_read_execute(self.physical) {
            return revoke_after(page, self.physical, ApTrampolineError::ProtectionFailed);
        }
        if page.permissions(self.physical)
            != Some(TrampolinePermissions {
                readable: true,
                writable: false,
                executable: true,
            })
        {
            return revoke_after(page, self.physical, ApTrampolineError::UnsafePermissions);
        }
        Ok(InstalledApTrampoline {
            physical: self.physical,
            startup_vector: self.startup_vector(),
        })
    }
}

fn revoke_after<P: ApTrampolinePage>(
    page: &mut P,
    physical: u64,
    error: ApTrampolineError,
) -> Result<InstalledApTrampoline, ApTrampolineError> {
    if page.revoke_and_zero(physical) {
        Err(error)
    } else {
        Err(ApTrampolineError::RevocationFailed)
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
            0xffff_8000_0020_fff8,
            7,
            9,
        )
        .unwrap();
        assert_eq!(image.startup_vector(), 8);
        assert_eq!(&image.bytes()[33..37], &0x20_0000u32.to_le_bytes());
        assert_eq!(
            &image.bytes()[48..54],
            &[0x66, 0x0d, 0x00, 0x09, 0x00, 0x00]
        );
        assert_eq!(
            &image.bytes()[92..100],
            &0xffff_8000_0020_fff8u64.to_le_bytes()
        );
        assert_eq!(&image.bytes()[102..110], &7u64.to_le_bytes());
        assert_eq!(&image.bytes()[112..120], &9u64.to_le_bytes());
        assert_eq!(
            &image.bytes()[122..130],
            &0xffff_8000_0020_0000u64.to_le_bytes()
        );
        assert_eq!(
            &image.bytes()[132..140],
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
            ApTrampolineImage::new(0, 0x2000, 0x1000, 0x9ff8, 0, 1).err(),
            Some(ApTrampolineError::InvalidPhysicalPage)
        );
        assert_eq!(
            ApTrampolineImage::new(0x8000, 0x2001, 0x1000, 0x10ff8, 0, 1).err(),
            Some(ApTrampolineError::InvalidPageTable)
        );
        assert_eq!(
            ApTrampolineImage::new(0x8000, 0x2000, 0x0000_8000_0000_0000, 0x10ff8, 0, 1).err(),
            Some(ApTrampolineError::InvalidEntry)
        );
        assert_eq!(
            ApTrampolineImage::new(0x8000, 0x2000, 0x1000, 0x2000, 0, 1).err(),
            Some(ApTrampolineError::InvalidStack)
        );
        assert_eq!(
            ApTrampolineImage::new(0x8000, 0x2000, 0x1000, 0x10ff8, 256, 1).err(),
            Some(ApTrampolineError::InvalidCpu)
        );
        assert_eq!(
            ApTrampolineImage::new(0x8000, 0x2000, 0x1000, 0x10ff8, 1, 0).err(),
            Some(ApTrampolineError::InvalidGeneration)
        );
    }

    struct Page {
        permissions: TrampolinePermissions,
        bytes: [u8; PAGE_BYTES],
        fail_protection: bool,
        revoked: bool,
    }

    impl Page {
        fn writable() -> Self {
            Self {
                permissions: TrampolinePermissions {
                    readable: true,
                    writable: true,
                    executable: false,
                },
                bytes: [0; PAGE_BYTES],
                fail_protection: false,
                revoked: false,
            }
        }
    }

    impl ApTrampolinePage for Page {
        fn permissions(&self, _: u64) -> Option<TrampolinePermissions> {
            Some(self.permissions)
        }

        fn write_page(&mut self, _: u64, bytes: &[u8; PAGE_BYTES]) -> bool {
            self.bytes.copy_from_slice(bytes);
            true
        }

        fn protect_read_execute(&mut self, _: u64) -> bool {
            if self.fail_protection {
                return false;
            }
            self.permissions = TrampolinePermissions {
                readable: true,
                writable: false,
                executable: true,
            };
            true
        }

        fn revoke_and_zero(&mut self, _: u64) -> bool {
            self.bytes.fill(0);
            self.permissions = TrampolinePermissions {
                readable: false,
                writable: false,
                executable: false,
            };
            self.revoked = true;
            true
        }

        fn rearm_read_write_and_zero(&mut self, _: u64) -> bool {
            self.bytes.fill(0);
            self.permissions = TrampolinePermissions {
                readable: true,
                writable: true,
                executable: false,
            };
            true
        }
    }

    #[test]
    fn installation_is_write_xor_execute_and_failure_revokes() {
        let image = ApTrampolineImage::new(0x8000, 0x20_0000, 0x1000, 0x10ff8, 1, 1).unwrap();
        let mut page = Page::writable();
        let installed = image.install(&mut page).unwrap();
        assert_eq!(installed.physical(), 0x8000);
        assert_eq!(installed.startup_vector(), 8);
        assert_eq!(page.bytes, *image.bytes());
        installed.rearm(&mut page).unwrap();
        assert_eq!(page.bytes, [0; PAGE_BYTES]);
        assert!(page.permissions.writable);
        assert!(!page.permissions.executable);

        let mut failed = Page::writable();
        failed.fail_protection = true;
        assert_eq!(
            image.install(&mut failed),
            Err(ApTrampolineError::ProtectionFailed)
        );
        assert!(failed.revoked);
        assert_eq!(failed.bytes, [0; PAGE_BYTES]);
    }
}
