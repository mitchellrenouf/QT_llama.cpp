use core::arch::asm;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DescriptorError {
    MissingTable,
    InvalidTableSize,
    InvalidHandler,
    InvalidSelector,
    InvalidSelectorDescriptor,
    InvalidIst,
    InvalidPrivilege,
    InvalidTaskState,
}

#[repr(C, packed)]
struct DescriptorPointer {
    limit: u16,
    base: u64,
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct TaskStateSegment {
    reserved0: u32,
    rsp: [u64; 3],
    reserved1: u64,
    ist: [u64; 7],
    reserved2: u64,
    reserved3: u16,
    io_map_base: u16,
}

#[repr(C, align(16))]
pub struct AlignedTaskState(pub TaskStateSegment);

impl TaskStateSegment {
    pub fn new(rsp0: u64, double_fault_stack: u64) -> Result<Self, DescriptorError> {
        if !valid_stack(rsp0) || !valid_stack(double_fault_stack) {
            return Err(DescriptorError::InvalidTaskState);
        }
        Ok(Self {
            reserved0: 0,
            rsp: [rsp0, 0, 0],
            reserved1: 0,
            ist: [double_fault_stack, 0, 0, 0, 0, 0, 0],
            reserved2: 0,
            reserved3: 0,
            io_map_base: core::mem::size_of::<Self>() as u16,
        })
    }

    pub const fn zeroed() -> Self {
        Self {
            reserved0: 0,
            rsp: [0; 3],
            reserved1: 0,
            ist: [0; 7],
            reserved2: 0,
            reserved3: 0,
            io_map_base: core::mem::size_of::<Self>() as u16,
        }
    }
}

impl AlignedTaskState {
    pub const fn zeroed() -> Self {
        Self(TaskStateSegment::zeroed())
    }
}

pub fn task_state_descriptor(task_state: &TaskStateSegment) -> Result<[u64; 2], DescriptorError> {
    let base = task_state as *const TaskStateSegment as u64;
    if !canonical(base) {
        return Err(DescriptorError::InvalidTaskState);
    }
    let limit = (core::mem::size_of::<TaskStateSegment>() - 1) as u64;
    let low = (limit & 0xffff)
        | ((base & 0x00ff_ffff) << 16)
        | (0x89 << 40)
        | (((limit >> 16) & 0xf) << 48)
        | (((base >> 24) & 0xff) << 56);
    Ok([low, base >> 32])
}

/// Writes a validated two-slot 64-bit TSS descriptor before the GDT is loaded.
///
/// # Safety
///
/// `gdt` must reference `gdt_entries` writable entries and must not be live on
/// another CPU while it is modified.
pub unsafe fn write_task_state_descriptor(
    gdt: *mut u64,
    gdt_entries: usize,
    selector: u16,
    task_state: &TaskStateSegment,
) -> Result<(), DescriptorError> {
    let slot = usize::from(selector >> 3);
    if gdt.is_null() || selector == 0 || selector & 7 != 0 || slot + 1 >= gdt_entries {
        return Err(DescriptorError::InvalidTaskState);
    }
    let descriptor = task_state_descriptor(task_state)?;
    unsafe {
        gdt.add(slot).write(descriptor[0]);
        gdt.add(slot + 1).write(descriptor[1]);
    }
    Ok(())
}

/// Loads a previously validated available 64-bit TSS descriptor.
///
/// # Safety
///
/// The current GDT must contain the live descriptor at `selector` and the
/// referenced TSS and its stacks must remain mapped and writable.
pub unsafe fn load_task_register(selector: u16) -> Result<(), DescriptorError> {
    if selector == 0 || selector & 7 != 0 {
        return Err(DescriptorError::InvalidTaskState);
    }
    unsafe { asm!("ltr {0:x}", in(reg) selector, options(nostack, preserves_flags)) };
    Ok(())
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
        Self::interrupt(address, selector, 0, 0)
    }

    pub fn interrupt(
        address: u64,
        selector: u16,
        ist: u8,
        privilege: u8,
    ) -> Result<Self, DescriptorError> {
        if !canonical(address) {
            return Err(DescriptorError::InvalidHandler);
        }
        if selector == 0 || selector & 7 != 0 {
            return Err(DescriptorError::InvalidSelector);
        }
        if ist > 7 {
            return Err(DescriptorError::InvalidIst);
        }
        if privilege > 3 {
            return Err(DescriptorError::InvalidPrivilege);
        }
        Ok(Self {
            low: address as u16,
            selector,
            ist,
            attributes: 0x8e | (privilege << 5),
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

/// Installs distinct handlers for all architectural exception vectors and a
/// fail-stop fallback for every external vector. All descriptors are validated
/// before either table is modified or loaded.
///
/// # Safety
///
/// The storage and privilege requirements are identical to
/// [`install_fail_stop_tables`]. Every handler must remain executable for the
/// lifetime of the loaded IDT.
pub unsafe fn install_exception_tables(
    gdt: *const u64,
    gdt_entries: usize,
    idt: *mut InterruptGate,
    idt_entries: usize,
    handlers: &[u64; 32],
    fallback: u64,
    selector: u16,
) -> Result<(), DescriptorError> {
    validate_table_storage(gdt, gdt_entries, idt, idt_entries, selector)?;
    let fallback = InterruptGate::fail_stop(fallback, selector)?;
    let mut exceptions = [InterruptGate::MISSING; 32];
    for (vector, handler) in handlers.iter().copied().enumerate() {
        let privilege = u8::from(vector == 3 || vector == 4) * 3;
        let ist = u8::from(vector == 8);
        exceptions[vector] = InterruptGate::interrupt(handler, selector, ist, privilege)?;
    }
    for (vector, gate) in exceptions.iter().copied().enumerate() {
        unsafe { idt.add(vector).write(gate) };
    }
    for vector in 32..idt_entries {
        unsafe { idt.add(vector).write(fallback) };
    }
    unsafe { load_tables(gdt, gdt_entries, idt, idt_entries) };
    Ok(())
}

fn validate_table_storage(
    gdt: *const u64,
    gdt_entries: usize,
    idt: *mut InterruptGate,
    idt_entries: usize,
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
        || selector == 0
        || selector & 7 != 0
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
    let descriptor = unsafe { gdt.add(usize::from(selector >> 3)).read() };
    if !valid_long_mode_code_descriptor(descriptor) {
        return Err(DescriptorError::InvalidSelectorDescriptor);
    }
    Ok(())
}

unsafe fn load_tables(
    gdt: *const u64,
    gdt_entries: usize,
    idt: *const InterruptGate,
    idt_entries: usize,
) {
    let gdtr = DescriptorPointer {
        limit: (gdt_entries * core::mem::size_of::<u64>() - 1) as u16,
        base: gdt as u64,
    };
    let idtr = DescriptorPointer {
        limit: (idt_entries * core::mem::size_of::<InterruptGate>() - 1) as u16,
        base: idt as u64,
    };
    unsafe {
        asm!("lgdt [{}]", in(reg) &gdtr, options(readonly, nostack, preserves_flags));
        asm!("lidt [{}]", in(reg) &idtr, options(readonly, nostack, preserves_flags));
    }
}

const fn canonical(address: u64) -> bool {
    ((address << 16) as i64 >> 16) as u64 == address
}

const fn valid_stack(address: u64) -> bool {
    address != 0 && address.is_multiple_of(16) && canonical(address)
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
        assert_eq!(
            InterruptGate::interrupt(0xffff_8000_0000_0000, 0x38, 8, 0),
            Err(DescriptorError::InvalidIst)
        );
        assert_eq!(
            InterruptGate::interrupt(0xffff_8000_0000_0000, 0x38, 0, 4),
            Err(DescriptorError::InvalidPrivilege)
        );
        let user_breakpoint = InterruptGate::interrupt(0xffff_8001_2345_6789, 0x38, 1, 3).unwrap();
        assert_eq!(user_breakpoint.encode()[4], 1);
        assert_eq!(user_breakpoint.encode()[5], 0xee);
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

    #[test]
    fn exception_table_validation_is_transactional() {
        let mut gdt = [0u64; 8];
        gdt[7] = 0x00af_9b00_0000_ffff;
        let mut idt = [InterruptGate::MISSING; 256];
        let mut handlers = [0xffff_8000_0000_1000; 32];
        handlers[17] = 0x0000_8000_0000_0000;
        assert_eq!(
            unsafe {
                install_exception_tables(
                    gdt.as_ptr(),
                    gdt.len(),
                    idt.as_mut_ptr(),
                    idt.len(),
                    &handlers,
                    0xffff_8000_0000_2000,
                    0x38,
                )
            },
            Err(DescriptorError::InvalidHandler)
        );
        assert!(idt.iter().all(|gate| *gate == InterruptGate::MISSING));
    }

    #[test]
    fn task_state_layout_and_descriptor_are_exact() {
        assert_eq!(core::mem::size_of::<TaskStateSegment>(), 104);
        assert_eq!(core::mem::offset_of!(TaskStateSegment, rsp), 4);
        assert_eq!(core::mem::offset_of!(TaskStateSegment, ist), 36);
        assert_eq!(core::mem::offset_of!(TaskStateSegment, io_map_base), 102);
        assert_eq!(core::mem::align_of::<AlignedTaskState>(), 16);
        assert_eq!(
            TaskStateSegment::new(0, 0x1000).err(),
            Some(DescriptorError::InvalidTaskState)
        );
        assert_eq!(
            TaskStateSegment::new(0x1008, 0x2000).err(),
            Some(DescriptorError::InvalidTaskState)
        );
        let task = TaskStateSegment::new(0x1000, 0x2000).unwrap();
        let descriptor = task_state_descriptor(&task).unwrap();
        let base = &task as *const TaskStateSegment as u64;
        assert_eq!(descriptor[0] & 0xffff, 103);
        assert_eq!((descriptor[0] >> 40) & 0xff, 0x89);
        assert_eq!(descriptor[1], base >> 32);
    }

    #[test]
    fn task_state_descriptor_write_rejects_aliasing_slots() {
        let task = TaskStateSegment::new(0x1000, 0x2000).unwrap();
        let mut gdt = [0u64; 2];
        assert_eq!(
            unsafe { write_task_state_descriptor(gdt.as_mut_ptr(), gdt.len(), 0x08, &task) },
            Err(DescriptorError::InvalidTaskState)
        );
        assert_eq!(gdt, [0; 2]);
    }
}
