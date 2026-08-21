use core::arch::asm;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DescriptorError {
    MissingTable,
    InvalidTableSize,
    InvalidHandler,
    InvalidSelector,
    InvalidSelectorDescriptor,
}

#[repr(C, packed)]
struct DescriptorPointer {
    limit: u16,
    base: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterruptGate {
    low: u16,
    selector: u16,
    ist: u8,
    attributes: u8,
    middle: u16,
    high: u32,
    reserved: u32,
}

impl InterruptGate {
    pub const MISSING: Self = Self {
        low: 0,
        selector: 0,
        ist: 0,
        attributes: 0,
        middle: 0,
        high: 0,
        reserved: 0,
    };

    pub fn fail_stop(address: u64, selector: u16) -> Result<Self, DescriptorError> {
        if !canonical(address) {
            return Err(DescriptorError::InvalidHandler);
        }
        if selector == 0 || selector & 7 != 0 {
            return Err(DescriptorError::InvalidSelector);
        }
        Ok(Self {
            low: address as u16,
            selector,
            ist: 0,
            attributes: 0x8e,
            middle: (address >> 16) as u16,
            high: (address >> 32) as u32,
            reserved: 0,
        })
    }

    pub fn encode(self) -> [u8; 16] {
        let mut encoded = [0u8; 16];
        encoded[0..2].copy_from_slice(&self.low.to_le_bytes());
        encoded[2..4].copy_from_slice(&self.selector.to_le_bytes());
        encoded[4] = self.ist;
        encoded[5] = self.attributes;
        encoded[6..8].copy_from_slice(&self.middle.to_le_bytes());
        encoded[8..12].copy_from_slice(&self.high.to_le_bytes());
        encoded[12..16].copy_from_slice(&self.reserved.to_le_bytes());
        encoded
    }
}

/// Populates a complete fail-stop IDT and atomically loads the supplied GDT and
/// IDT. Callers must keep both tables alive and immovable for the CPU lifetime.
///
/// # Safety
///
/// `gdt` must reference `gdt_entries` readable `u64` values and `idt` must
/// reference `idt_entries` writable `InterruptGate` values. This operation is
/// privileged and may only run during single-CPU architectural initialization.
pub unsafe fn install_fail_stop_tables(
    gdt: *const u64,
    gdt_entries: usize,
    idt: *mut InterruptGate,
    idt_entries: usize,
    handler: u64,
    selector: u16,
) -> Result<(), DescriptorError> {
    if gdt.is_null() || idt.is_null() {
        return Err(DescriptorError::MissingTable);
    }
    let gdt_bytes = gdt_entries
        .checked_mul(core::mem::size_of::<u64>())
        .ok_or(DescriptorError::InvalidTableSize)?;
    let idt_bytes = idt_entries
        .checked_mul(core::mem::size_of::<InterruptGate>())
        .ok_or(DescriptorError::InvalidTableSize)?;
    if gdt_bytes == 0
        || idt_entries != 256
        || gdt_bytes > usize::from(u16::MAX) + 1
        || idt_bytes > usize::from(u16::MAX) + 1
        || usize::from(selector >> 3) >= gdt_entries
        || !canonical(gdt as u64)
        || !canonical(idt as u64)
        || gdt_bytes
            .checked_sub(1)
            .and_then(|bytes| (gdt as u64).checked_add(bytes as u64))
            .is_none_or(|end| !canonical(end))
        || idt_bytes
            .checked_sub(1)
            .and_then(|bytes| (idt as u64).checked_add(bytes as u64))
            .is_none_or(|end| !canonical(end))
    {
        return Err(DescriptorError::InvalidTableSize);
    }
    let gate = InterruptGate::fail_stop(handler, selector)?;
    let descriptor = unsafe { gdt.add(usize::from(selector >> 3)).read() };
    if !valid_long_mode_code_descriptor(descriptor) {
        return Err(DescriptorError::InvalidSelectorDescriptor);
    }
    for index in 0..idt_entries {
        unsafe { idt.add(index).write(gate) };
    }
    let gdtr = DescriptorPointer {
        limit: (gdt_bytes - 1) as u16,
        base: gdt as u64,
    };
    let idtr = DescriptorPointer {
        limit: (idt_bytes - 1) as u16,
        base: idt as u64,
    };
    unsafe {
        asm!("lgdt [{}]", in(reg) &gdtr, options(readonly, nostack, preserves_flags));
        asm!("lidt [{}]", in(reg) &idtr, options(readonly, nostack, preserves_flags));
    }
    Ok(())
}

const fn canonical(address: u64) -> bool {
    ((address << 16) as i64 >> 16) as u64 == address
}

const fn valid_long_mode_code_descriptor(descriptor: u64) -> bool {
    descriptor & (1 << 47) != 0
        && descriptor & (1 << 44) != 0
        && descriptor & (1 << 43) != 0
        && descriptor & (1 << 40) != 0
        && descriptor & (3 << 45) == 0
        && descriptor & (1 << 53) != 0
        && descriptor & (1 << 54) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_encoding_is_exact_and_rejects_unsafe_inputs() {
        let gate = InterruptGate::fail_stop(0xffff_8001_2345_6789, 0x38).unwrap();
        assert_eq!(
            gate.encode(),
            [
                0x89, 0x67, 0x38, 0x00, 0x00, 0x8e, 0x45, 0x23, 0x01, 0x80, 0xff, 0xff, 0x00, 0x00,
                0x00, 0x00,
            ]
        );
        assert_eq!(
            InterruptGate::fail_stop(0x0000_8000_0000_0000, 0x38),
            Err(DescriptorError::InvalidHandler)
        );
        assert_eq!(
            InterruptGate::fail_stop(0xffff_8000_0000_0000, 0x39),
            Err(DescriptorError::InvalidSelector)
        );
    }

    #[test]
    fn table_installation_rejects_invalid_storage_before_privileged_ops() {
        let mut idt = [InterruptGate::MISSING; 256];
        let gdt = [0u64; 8];
        assert_eq!(
            unsafe {
                install_fail_stop_tables(
                    core::ptr::null(),
                    8,
                    idt.as_mut_ptr(),
                    256,
                    0xffff_8000_0000_1000,
                    0x38,
                )
            },
            Err(DescriptorError::MissingTable)
        );
        assert_eq!(
            unsafe {
                install_fail_stop_tables(
                    gdt.as_ptr(),
                    gdt.len(),
                    idt.as_mut_ptr(),
                    255,
                    0xffff_8000_0000_1000,
                    0x38,
                )
            },
            Err(DescriptorError::InvalidTableSize)
        );
        assert_eq!(
            unsafe {
                install_fail_stop_tables(
                    0x0000_8000_0000_0000usize as *const u64,
                    gdt.len(),
                    idt.as_mut_ptr(),
                    256,
                    0xffff_8000_0000_1000,
                    0x38,
                )
            },
            Err(DescriptorError::InvalidTableSize)
        );
    }

    #[test]
    fn selector_descriptor_must_be_present_ring_zero_long_code_and_preaccessed() {
        assert!(valid_long_mode_code_descriptor(0x00af_9b00_0000_ffff));
        assert!(!valid_long_mode_code_descriptor(0x00af_9a00_0000_ffff));
        assert!(!valid_long_mode_code_descriptor(0x00af_9300_0000_ffff));
        assert!(!valid_long_mode_code_descriptor(0x00cf_9b00_0000_ffff));
    }
}
