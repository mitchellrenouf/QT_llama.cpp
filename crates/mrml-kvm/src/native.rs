use core::ffi::{c_char, c_int, c_ulong, c_void};
use core::ptr::NonNull;
use core::slice;
use core::time::Duration;
use mrml_kernel::arch::x86_64::{
    AddressSpace, Mapping, PagePermissions, PageTableBuildError, PageTableBuilder, PageTableStore,
    VirtAddr,
};
use mrml_kernel::{
    BootHandoff, MAX_PE_SECTIONS, PAGE_SIZE, PeImage, PhysAddr, VerifiedExecutable, VmBackend,
    VmExit,
};

use crate::{KVM_API_VERSION, KvmError, KvmMemoryRegion, decode_run_page};

mod launch;
pub use launch::{KvmLaunchLayout, KvmPageWalk, PreparedKvmGuest};

const O_RDWR: c_int = 2;
const O_CLOEXEC: c_int = 0x80000;
const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const MAP_SHARED: c_int = 1;
const MAP_PRIVATE: c_int = 2;
const MAP_ANONYMOUS: c_int = 0x20;
const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
const KVM_GET_API_VERSION: c_ulong = 0xae00;
const KVM_CREATE_VM: c_ulong = 0xae01;
const KVM_CHECK_EXTENSION: c_ulong = 0xae03;
const KVM_GET_VCPU_MMAP_SIZE: c_ulong = 0xae04;
const KVM_GET_SUPPORTED_CPUID: c_ulong = 0xc008_ae05;
const KVM_CREATE_VCPU: c_ulong = 0xae41;
const KVM_CREATE_IRQCHIP: c_ulong = 0xae60;
const KVM_SET_USER_MEMORY_REGION: c_ulong = 0x4020_ae46;
const KVM_RUN: c_ulong = 0xae80;
const KVM_GET_REGS: c_ulong = 0x8090_ae81;
const KVM_GET_SREGS: c_ulong = 0x8138_ae83;
const KVM_SET_REGS: c_ulong = 0x4090_ae82;
const KVM_SET_SREGS: c_ulong = 0x4138_ae84;
const KVM_INTERRUPT: c_ulong = 0x4004_ae86;
const KVM_SET_CPUID2: c_ulong = 0x4008_ae90;
const KVM_GET_LAPIC: c_ulong = 0x8400_ae8e;
const KVM_GET_MP_STATE: c_ulong = 0x8004_ae98;
const KVM_SET_MP_STATE: c_ulong = 0x4004_ae99;
const KVM_MP_STATE_UNINITIALIZED: u32 = 1;
const MIN_RUN_BYTES: usize = 88;
const MAX_RUN_BYTES: usize = 1024 * 1024;
const KVM_CAP_USER_MEMORY: u32 = 3;
const KVM_CAP_NR_MEMSLOTS: u32 = 10;
const REQUIRED_MEMORY_SLOTS: i32 = 5;
const MAX_CPUID_ENTRIES: usize = 128;

#[derive(Clone, Copy)]
#[repr(C)]
struct KvmCpuidEntry2 {
    function: u32,
    index: u32,
    flags: u32,
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
    padding: [u32; 3],
}

const EMPTY_CPUID_ENTRY: KvmCpuidEntry2 = KvmCpuidEntry2 {
    function: 0,
    index: 0,
    flags: 0,
    eax: 0,
    ebx: 0,
    ecx: 0,
    edx: 0,
    padding: [0; 3],
};

#[derive(Clone, Copy)]
#[repr(C)]
struct KvmCpuidSet {
    count: u32,
    padding: u32,
    entries: [KvmCpuidEntry2; MAX_CPUID_ENTRIES],
}

#[repr(C)]
struct KvmMpState {
    state: u32,
}

#[repr(C)]
struct KvmLapicState {
    registers: [u8; 1024],
}

#[link(name = "c")]
unsafe extern "C" {
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn close(file: c_int) -> c_int;
    fn ioctl(file: c_int, request: c_ulong, ...) -> c_int;
    fn mmap(
        address: *mut c_void,
        length: usize,
        protection: c_int,
        flags: c_int,
        file: c_int,
        offset: isize,
    ) -> *mut c_void;
    fn munmap(address: *mut c_void, length: usize) -> c_int;
}

struct OwnedFd(c_int);

impl Drop for OwnedFd {
    fn drop(&mut self) {
        unsafe {
            close(self.0);
        }
    }
}

pub struct KvmSystem {
    file: OwnedFd,
    run_bytes: usize,
    cpuid: KvmCpuidSet,
}

impl KvmSystem {
    fn virtual_cpuid(&self, apic_id: u32, cpus: u8) -> Result<KvmCpuidSet, KvmError> {
        let apic_id = u8::try_from(apic_id)
            .ok()
            .filter(|id| *id < cpus)
            .ok_or(KvmError::InvalidVcpu)?;
        let mut cpuid = self.cpuid;
        for entry in &mut cpuid.entries[..cpuid.count as usize] {
            match entry.function {
                1 => {
                    entry.ebx = (entry.ebx & 0x0000_ffff)
                        | (u32::from(cpus) << 16)
                        | (u32::from(apic_id) << 24);
                }
                0x0b | 0x1f => entry.edx = u32::from(apic_id),
                _ => {}
            }
        }
        Ok(cpuid)
    }

    pub fn open() -> Result<Self, KvmError> {
        let file = unsafe { open(c"/dev/kvm".as_ptr(), O_RDWR | O_CLOEXEC) };
        if file < 0 {
            return Err(KvmError::SystemCall);
        }
        let file = OwnedFd(file);
        if unsafe { ioctl(file.0, KVM_GET_API_VERSION, 0 as c_ulong) } != KVM_API_VERSION {
            return Err(KvmError::ApiVersion);
        }
        let user_memory =
            unsafe { ioctl(file.0, KVM_CHECK_EXTENSION, KVM_CAP_USER_MEMORY as c_ulong) };
        let memory_slots =
            unsafe { ioctl(file.0, KVM_CHECK_EXTENSION, KVM_CAP_NR_MEMSLOTS as c_ulong) };
        validate_capabilities(user_memory, memory_slots)?;
        let run_bytes = unsafe { ioctl(file.0, KVM_GET_VCPU_MMAP_SIZE, 0 as c_ulong) };
        if run_bytes < 0 {
            return Err(KvmError::SystemCall);
        }
        if run_bytes < MIN_RUN_BYTES as c_int || run_bytes as usize > MAX_RUN_BYTES {
            return Err(KvmError::InvalidRunSize(run_bytes));
        }
        let mut cpuid = KvmCpuidSet {
            count: MAX_CPUID_ENTRIES as u32,
            padding: 0,
            entries: [EMPTY_CPUID_ENTRY; MAX_CPUID_ENTRIES],
        };
        if unsafe { ioctl(file.0, KVM_GET_SUPPORTED_CPUID, &mut cpuid) } < 0
            || cpuid.count == 0
            || cpuid.count as usize > MAX_CPUID_ENTRIES
        {
            return Err(KvmError::SystemCall);
        }
        Ok(Self {
            file,
            run_bytes: run_bytes as usize,
            cpuid,
        })
    }

    pub(crate) fn create_vm(&self) -> Result<KvmVm, KvmError> {
        let file = unsafe { ioctl(self.file.0, KVM_CREATE_VM, 0 as c_ulong) };
        if file < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(KvmVm {
            file: OwnedFd(file),
            run_bytes: self.run_bytes,
        })
    }

    pub(crate) fn create_backend<const N: usize>(
        &self,
        vcpu_id: u32,
    ) -> Result<KvmBackend<N>, KvmError> {
        let vm = self.create_vm()?;
        let vcpu = vm.create_vcpu(vcpu_id)?;
        Ok(KvmBackend {
            vcpu,
            secondary: None,
            memory: KvmGuestMemory::new(),
            vm,
            vcpu_id,
        })
    }

    pub(crate) fn create_apic_backend<const N: usize>(
        &self,
        vcpu_id: u32,
    ) -> Result<KvmBackend<N>, KvmError> {
        let vm = self.create_vm()?;
        vm.create_irqchip()?;
        let vcpu = vm.create_vcpu(vcpu_id)?;
        vcpu.set_cpuid(&self.virtual_cpuid(vcpu_id, 1)?)?;
        Ok(KvmBackend {
            vcpu,
            secondary: None,
            memory: KvmGuestMemory::new(),
            vm,
            vcpu_id,
        })
    }

    pub(crate) fn create_smp_backend<const N: usize>(&self) -> Result<KvmBackend<N>, KvmError> {
        let vm = self.create_vm()?;
        vm.create_irqchip()?;
        let vcpu = vm.create_vcpu(0)?;
        vcpu.set_cpuid(&self.virtual_cpuid(0, 2)?)?;
        let secondary = vm.create_vcpu(1)?;
        secondary.set_cpuid(&self.virtual_cpuid(1, 2)?)?;
        secondary.set_uninitialized()?;
        let bootstrap_id = vcpu.lapic_id()?;
        let application_id = secondary.lapic_id()?;
        let application_state = secondary.mp_state()?;
        mrml_runtime::mrml_println!(
            "KVM_SMP_SETUP bsp_apic={} ap_apic={} ap_state={}",
            bootstrap_id,
            application_id,
            application_state
        );
        if bootstrap_id != 0
            || application_id != 1
            || application_state != KVM_MP_STATE_UNINITIALIZED
        {
            return Err(KvmError::InvalidVcpu);
        }
        Ok(KvmBackend {
            vcpu,
            secondary: Some(secondary),
            memory: KvmGuestMemory::new(),
            vm,
            vcpu_id: 0,
        })
    }
}

fn validate_capabilities(user_memory: c_int, memory_slots: c_int) -> Result<(), KvmError> {
    if user_memory <= 0 {
        return Err(KvmError::UnsupportedCapability(KVM_CAP_USER_MEMORY));
    }
    if memory_slots < REQUIRED_MEMORY_SLOTS {
        return Err(KvmError::InsufficientMemorySlots(memory_slots));
    }
    Ok(())
}

pub(crate) struct KvmVm {
    file: OwnedFd,
    run_bytes: usize,
}

impl KvmVm {
    pub(crate) fn create_irqchip(&self) -> Result<(), KvmError> {
        if unsafe { ioctl(self.file.0, KVM_CREATE_IRQCHIP, 0 as c_ulong) } < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(())
    }

    pub(crate) fn register_memory(&self, region: KvmMemoryRegion) -> Result<(), KvmError> {
        let encoded = region.encode();
        if unsafe { ioctl(self.file.0, KVM_SET_USER_MEMORY_REGION, encoded.as_ptr()) } < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(())
    }

    pub(crate) fn create_vcpu(&self, id: u32) -> Result<KvmVcpu, KvmError> {
        let file = unsafe { ioctl(self.file.0, KVM_CREATE_VCPU, id as c_ulong) };
        if file < 0 {
            return Err(KvmError::SystemCall);
        }
        let file = OwnedFd(file);
        let run = unsafe {
            mmap(
                core::ptr::null_mut(),
                self.run_bytes,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                file.0,
                0,
            )
        };
        if run == MAP_FAILED {
            return Err(KvmError::SystemCall);
        }
        let run = NonNull::new(run.cast()).ok_or(KvmError::SystemCall)?;
        Ok(KvmVcpu {
            file,
            run,
            run_bytes: self.run_bytes,
        })
    }
}

pub(crate) struct KvmVcpu {
    file: OwnedFd,
    run: NonNull<u8>,
    run_bytes: usize,
}

// SAFETY: the descriptor and run mapping have one owner. Moving them to a
// single runner thread creates no aliases, and VM setup is complete first.
unsafe impl Send for KvmVcpu {}

#[derive(Clone, Copy)]
struct KvmRunCancellation(NonNull<u8>);

// SAFETY: KVM defines byte one of the shared run page as `immediate_exit`, an
// asynchronously writable cancellation byte. This handle exposes only it.
unsafe impl Send for KvmRunCancellation {}
unsafe impl Sync for KvmRunCancellation {}

impl KvmRunCancellation {
    fn request(self) {
        unsafe { self.0.as_ptr().add(1).write_volatile(1) };
    }
}

impl KvmVcpu {
    fn lapic_id(&self) -> Result<u8, KvmError> {
        let mut state = KvmLapicState {
            registers: [0; 1024],
        };
        if unsafe { ioctl(self.file.0, KVM_GET_LAPIC, &mut state) } < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(state.registers[0x23])
    }

    fn mp_state(&self) -> Result<u32, KvmError> {
        let mut state = KvmMpState { state: 0 };
        if unsafe { ioctl(self.file.0, KVM_GET_MP_STATE, &mut state) } < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(state.state)
    }
    fn set_uninitialized(&self) -> Result<(), KvmError> {
        let state = KvmMpState {
            state: KVM_MP_STATE_UNINITIALIZED,
        };
        if unsafe { ioctl(self.file.0, KVM_SET_MP_STATE, &state) } < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(())
    }
    fn cancellation(&self) -> KvmRunCancellation {
        KvmRunCancellation(self.run)
    }
    fn set_cpuid(&self, cpuid: &KvmCpuidSet) -> Result<(), KvmError> {
        if cpuid.count == 0 || cpuid.count as usize > MAX_CPUID_ENTRIES {
            return Err(KvmError::InvalidRegisterState);
        }
        if unsafe { ioctl(self.file.0, KVM_SET_CPUID2, cpuid) } < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(())
    }

    pub(crate) fn run(&mut self) -> Result<VmExit, KvmError> {
        if unsafe { ioctl(self.file.0, KVM_RUN, 0 as c_ulong) } < 0 {
            return Err(KvmError::SystemCall);
        }
        let bytes = unsafe { slice::from_raw_parts(self.run.as_ptr(), self.run_bytes) };
        decode_run_page(bytes)
    }

    pub(crate) fn configure_long_mode_entry(
        &mut self,
        entry: u64,
        stack: u64,
        cr3: u64,
        argument0: u64,
        argument1: u64,
        argument2: u64,
        argument3: u64,
    ) -> Result<(), KvmError> {
        if !canonical(entry) || !canonical(stack) || !cr3.is_multiple_of(PAGE_SIZE) {
            return Err(KvmError::InvalidRegisterState);
        }
        let mut special = KvmSpecialRegisters::zeroed();
        if unsafe { ioctl(self.file.0, KVM_GET_SREGS, &mut special) } < 0 {
            return Err(KvmError::SystemCall);
        }
        special.cs = KvmSegment::code64();
        let data = KvmSegment::data64();
        special.ds = data;
        special.es = data;
        special.fs = data;
        special.gs = data;
        special.ss = data;
        special.cr0 |= (1 << 0) | (1 << 5) | (1 << 16) | (1 << 31);
        special.cr3 = cr3;
        special.cr4 |= 1 << 5;
        special.efer |= (1 << 8) | (1 << 10) | (1 << 11);
        if unsafe { ioctl(self.file.0, KVM_SET_SREGS, &special) } < 0 {
            return Err(KvmError::SystemCall);
        }
        let registers =
            KvmRegisters::initial(entry, stack, argument0, argument1, argument2, argument3);
        if unsafe { ioctl(self.file.0, KVM_SET_REGS, &registers) } < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(())
    }

    fn snapshot(&self) -> Result<KvmVcpuSnapshot, KvmError> {
        let mut registers = KvmRegisters::zeroed();
        if unsafe { ioctl(self.file.0, KVM_GET_REGS, &mut registers) } < 0 {
            return Err(KvmError::SystemCall);
        }
        let mut special = KvmSpecialRegisters::zeroed();
        if unsafe { ioctl(self.file.0, KVM_GET_SREGS, &mut special) } < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(KvmVcpuSnapshot {
            instruction_pointer: registers.rip,
            stack_pointer: registers.rsp,
            flags: registers.rflags,
            fault_address: special.cr2,
            page_table_root: special.cr3,
            code_selector: special.cs.selector,
            gdt_base: special.gdt.base,
            gdt_limit: special.gdt.limit,
            idt_base: special.idt.base,
            idt_limit: special.idt.limit,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvmVcpuSnapshot {
    instruction_pointer: u64,
    stack_pointer: u64,
    flags: u64,
    fault_address: u64,
    page_table_root: u64,
    code_selector: u16,
    gdt_base: u64,
    gdt_limit: u16,
    idt_base: u64,
    idt_limit: u16,
}

impl KvmVcpuSnapshot {
    pub const fn instruction_pointer(self) -> u64 {
        self.instruction_pointer
    }
    pub const fn stack_pointer(self) -> u64 {
        self.stack_pointer
    }
    pub const fn flags(self) -> u64 {
        self.flags
    }
    pub const fn fault_address(self) -> u64 {
        self.fault_address
    }
    pub const fn page_table_root(self) -> u64 {
        self.page_table_root
    }
    pub const fn code_selector(self) -> u16 {
        self.code_selector
    }
    pub const fn gdt_base(self) -> u64 {
        self.gdt_base
    }
    pub const fn gdt_limit(self) -> u16 {
        self.gdt_limit
    }
    pub const fn idt_base(self) -> u64 {
        self.idt_base
    }
    pub const fn idt_limit(self) -> u16 {
        self.idt_limit
    }
}

#[repr(C)]
struct KvmRegisters {
    rax: u64,
    rbx: u64,
    rcx: u64,
    rdx: u64,
    rsi: u64,
    rdi: u64,
    rsp: u64,
    rbp: u64,
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    rip: u64,
    rflags: u64,
}

impl KvmRegisters {
    const fn zeroed() -> Self {
        Self::initial(0, 0, 0, 0, 0, 0)
    }
    const fn initial(
        entry: u64,
        stack: u64,
        argument0: u64,
        argument1: u64,
        argument2: u64,
        argument3: u64,
    ) -> Self {
        Self {
            rax: 0,
            rbx: 0,
            rcx: argument0,
            rdx: argument1,
            rsi: 0,
            rdi: 0,
            rsp: stack,
            rbp: 0,
            r8: argument2,
            r9: argument3,
            r10: 0,
            r11: 0,
            r12: 0,
            r13: 0,
            r14: 0,
            r15: 0,
            rip: entry,
            rflags: 2,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct KvmSegment {
    base: u64,
    limit: u32,
    selector: u16,
    kind: u8,
    present: u8,
    dpl: u8,
    default_operand: u8,
    system: u8,
    long: u8,
    granularity: u8,
    available: u8,
    unusable: u8,
    padding: u8,
}

impl KvmSegment {
    const fn code64() -> Self {
        Self {
            base: 0,
            limit: u32::MAX,
            selector: 8,
            kind: 11,
            present: 1,
            dpl: 0,
            default_operand: 0,
            system: 1,
            long: 1,
            granularity: 1,
            available: 0,
            unusable: 0,
            padding: 0,
        }
    }
    const fn data64() -> Self {
        Self {
            base: 0,
            limit: u32::MAX,
            selector: 16,
            kind: 3,
            present: 1,
            dpl: 0,
            default_operand: 1,
            system: 1,
            long: 0,
            granularity: 1,
            available: 0,
            unusable: 0,
            padding: 0,
        }
    }
    const fn zeroed() -> Self {
        Self {
            base: 0,
            limit: 0,
            selector: 0,
            kind: 0,
            present: 0,
            dpl: 0,
            default_operand: 0,
            system: 0,
            long: 0,
            granularity: 0,
            available: 0,
            unusable: 0,
            padding: 0,
        }
    }
}

#[repr(C)]
struct KvmDescriptorTable {
    base: u64,
    limit: u16,
    padding: [u16; 3],
}

impl KvmDescriptorTable {
    const fn zeroed() -> Self {
        Self {
            base: 0,
            limit: 0,
            padding: [0; 3],
        }
    }
}

#[repr(C)]
struct KvmSpecialRegisters {
    cs: KvmSegment,
    ds: KvmSegment,
    es: KvmSegment,
    fs: KvmSegment,
    gs: KvmSegment,
    ss: KvmSegment,
    tr: KvmSegment,
    ldt: KvmSegment,
    gdt: KvmDescriptorTable,
    idt: KvmDescriptorTable,
    cr0: u64,
    cr2: u64,
    cr3: u64,
    cr4: u64,
    cr8: u64,
    efer: u64,
    apic_base: u64,
    interrupt_bitmap: [u64; 4],
}

impl KvmSpecialRegisters {
    const fn zeroed() -> Self {
        Self {
            cs: KvmSegment::zeroed(),
            ds: KvmSegment::zeroed(),
            es: KvmSegment::zeroed(),
            fs: KvmSegment::zeroed(),
            gs: KvmSegment::zeroed(),
            ss: KvmSegment::zeroed(),
            tr: KvmSegment::zeroed(),
            ldt: KvmSegment::zeroed(),
            gdt: KvmDescriptorTable::zeroed(),
            idt: KvmDescriptorTable::zeroed(),
            cr0: 0,
            cr2: 0,
            cr3: 0,
            cr4: 0,
            cr8: 0,
            efer: 0,
            apic_base: 0,
            interrupt_bitmap: [0; 4],
        }
    }
}

const fn canonical(address: u64) -> bool {
    ((address << 16) as i64 >> 16) as u64 == address
}

impl Drop for KvmVcpu {
    fn drop(&mut self) {
        unsafe {
            munmap(self.run.as_ptr().cast(), self.run_bytes);
        }
    }
}

struct HostRegion {
    guest_address: u64,
    bytes: usize,
    host: NonNull<u8>,
    readonly: bool,
}

impl HostRegion {
    fn allocate(guest_address: u64, bytes: usize, readonly: bool) -> Result<Self, KvmError> {
        let bytes_u64 = u64::try_from(bytes).map_err(|_| KvmError::MemoryOverflow)?;
        KvmMemoryRegion::new(0, guest_address, bytes_u64, PAGE_SIZE, readonly)?;
        let host = unsafe {
            mmap(
                core::ptr::null_mut(),
                bytes,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if host == MAP_FAILED {
            return Err(KvmError::SystemCall);
        }
        Ok(Self {
            guest_address,
            bytes,
            host: NonNull::new(host.cast()).ok_or(KvmError::SystemCall)?,
            readonly,
        })
    }

    fn end(&self) -> u64 {
        self.guest_address + self.bytes as u64
    }
}

impl Drop for HostRegion {
    fn drop(&mut self) {
        unsafe {
            munmap(self.host.as_ptr().cast(), self.bytes);
        }
    }
}

pub struct KvmGuestMemory<const N: usize> {
    regions: [Option<HostRegion>; N],
    count: usize,
}

impl<const N: usize> KvmGuestMemory<N> {
    pub const fn new() -> Self {
        Self {
            regions: [const { None }; N],
            count: 0,
        }
    }

    fn allocate(
        &mut self,
        guest_address: u64,
        bytes: usize,
        readonly: bool,
    ) -> Result<(u32, u64), KvmError> {
        if self.count == N || self.count >= crate::MAX_KVM_MEMORY_SLOTS as usize {
            return Err(KvmError::MemoryTableFull);
        }
        let region = HostRegion::allocate(guest_address, bytes, readonly)?;
        if self.regions[..self.count]
            .iter()
            .flatten()
            .any(|existing| guest_address < existing.end() && existing.guest_address < region.end())
        {
            return Err(KvmError::MemoryOverlap);
        }
        let slot = self.count as u32;
        let host = region.host.as_ptr() as u64;
        self.regions[self.count] = Some(region);
        self.count += 1;
        Ok((slot, host))
    }

    fn rollback_last(&mut self) {
        if self.count != 0 {
            self.count -= 1;
            self.regions[self.count] = None;
        }
    }

    pub fn read(&self, guest_address: u64, output: &mut [u8]) -> Result<(), KvmError> {
        let (region, offset) = self.locate(guest_address, output.len())?;
        unsafe {
            core::ptr::copy_nonoverlapping(
                region.host.as_ptr().add(offset),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        Ok(())
    }

    pub fn write(&mut self, guest_address: u64, input: &[u8]) -> Result<(), KvmError> {
        let (region, offset) = self.locate(guest_address, input.len())?;
        if region.readonly {
            return Err(KvmError::ReadOnlyMemory);
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                input.as_ptr(),
                region.host.as_ptr().add(offset),
                input.len(),
            )
        };
        Ok(())
    }

    /// Writes through the host-owned backing even when KVM denies guest writes
    /// to the slot. Callers must hold the isolated service authority for the
    /// target shared-memory protocol.
    pub(crate) fn write_service(
        &mut self,
        guest_address: u64,
        input: &[u8],
    ) -> Result<(), KvmError> {
        let (region, offset) = self.locate(guest_address, input.len())?;
        unsafe {
            core::ptr::copy_nonoverlapping(
                input.as_ptr(),
                region.host.as_ptr().add(offset),
                input.len(),
            )
        };
        Ok(())
    }

    pub fn load_verified_executable(
        &mut self,
        executable: &VerifiedExecutable<'_>,
        guest_physical_base: u64,
        virtual_base: u64,
    ) -> Result<KvmLoadedImage, KvmError> {
        self.load_pe(executable.image(), guest_physical_base, virtual_base)
    }

    pub fn load_boot_handoff(
        &mut self,
        encoded: &[u8],
        guest_physical_address: u64,
    ) -> Result<KvmLoadedHandoff, KvmError> {
        if !guest_physical_address.is_multiple_of(PAGE_SIZE) {
            return Err(KvmError::UnalignedMemory);
        }
        BootHandoff::decode(encoded, |_| {}).map_err(KvmError::Handoff)?;
        let pages = (encoded.len() as u64)
            .checked_add(PAGE_SIZE - 1)
            .ok_or(KvmError::MemoryOverflow)?
            / PAGE_SIZE;
        let allocation_bytes = pages
            .checked_mul(PAGE_SIZE)
            .ok_or(KvmError::MemoryOverflow)?;
        let allocation_bytes =
            usize::try_from(allocation_bytes).map_err(|_| KvmError::MemoryOverflow)?;
        let (host, offset, readonly) = {
            let (region, offset) = self.locate(guest_physical_address, allocation_bytes)?;
            (region.host, offset, region.readonly)
        };
        if readonly {
            return Err(KvmError::ReadOnlyMemory);
        }
        let destination =
            unsafe { slice::from_raw_parts_mut(host.as_ptr().add(offset), allocation_bytes) };
        destination.fill(0);
        destination[..encoded.len()].copy_from_slice(encoded);
        Ok(KvmLoadedHandoff {
            physical_address: guest_physical_address,
            bytes: encoded.len() as u32,
            pages: pages as u32,
        })
    }

    fn load_pe(
        &mut self,
        image: &PeImage<'_>,
        guest_physical_base: u64,
        virtual_base: u64,
    ) -> Result<KvmLoadedImage, KvmError> {
        let image_bytes = image.image_size() as usize;
        let (host, offset, readonly) = {
            let (region, offset) = self.locate(guest_physical_base, image_bytes)?;
            (region.host, offset, region.readonly)
        };
        if readonly {
            return Err(KvmError::ReadOnlyMemory);
        }
        let destination =
            unsafe { slice::from_raw_parts_mut(host.as_ptr().add(offset), image_bytes) };
        let entry = image
            .materialize_at(destination, virtual_base)
            .map_err(KvmError::Pe)?;
        Ok(KvmLoadedImage {
            entry,
            virtual_base,
            physical_base: guest_physical_base,
            image_bytes: image.image_size(),
        })
    }

    fn locate(&self, guest_address: u64, bytes: usize) -> Result<(&HostRegion, usize), KvmError> {
        if bytes == 0 {
            return Err(KvmError::UnmappedMemory);
        }
        let end = guest_address
            .checked_add(bytes as u64)
            .ok_or(KvmError::MemoryOverflow)?;
        let region = self.regions[..self.count]
            .iter()
            .flatten()
            .find(|region| guest_address >= region.guest_address && end <= region.end())
            .ok_or(KvmError::UnmappedMemory)?;
        Ok((region, (guest_address - region.guest_address) as usize))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvmLoadedImage {
    entry: u64,
    virtual_base: u64,
    physical_base: u64,
    image_bytes: u32,
}

impl KvmLoadedImage {
    pub const fn entry(self) -> u64 {
        self.entry
    }
    pub const fn virtual_base(self) -> u64 {
        self.virtual_base
    }
    pub const fn physical_base(self) -> u64 {
        self.physical_base
    }
    pub const fn image_bytes(self) -> u32 {
        self.image_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvmLoadedHandoff {
    physical_address: u64,
    bytes: u32,
    pages: u32,
}

impl KvmLoadedHandoff {
    pub const fn physical_address(self) -> u64 {
        self.physical_address
    }
    pub const fn bytes(self) -> u32 {
        self.bytes
    }
    pub const fn pages(self) -> u32 {
        self.pages
    }
}

pub fn map_loaded_handoff<S: PageTableStore>(
    tables: &mut PageTableBuilder<S>,
    handoff: KvmLoadedHandoff,
    virtual_address: u64,
    user: bool,
) -> Result<(), KvmError> {
    let permissions = if user {
        PagePermissions::USER_READ
    } else {
        PagePermissions::KERNEL_READ
    };
    let mapping = Mapping::new(
        VirtAddr::new(virtual_address).map_err(|_| KvmError::InvalidMapping)?,
        PhysAddr::new(handoff.physical_address).map_err(|_| KvmError::InvalidMapping)?,
        handoff.pages as u64,
        permissions,
    )
    .map_err(|_| KvmError::InvalidMapping)?;
    tables.map(mapping).map_err(|_| KvmError::PageTable)
}

pub fn map_loaded_pe<S: PageTableStore>(
    tables: &mut PageTableBuilder<S>,
    executable: &VerifiedExecutable<'_>,
    loaded: KvmLoadedImage,
    user: bool,
) -> Result<(), KvmError> {
    map_pe(tables, executable.image(), loaded, user)
}

fn map_pe<S: PageTableStore>(
    tables: &mut PageTableBuilder<S>,
    image: &PeImage<'_>,
    loaded: KvmLoadedImage,
    user: bool,
) -> Result<(), KvmError> {
    if loaded.image_bytes != image.image_size() {
        return Err(KvmError::InvalidMapping);
    }
    let mut validated = AddressSpace::<{ MAX_PE_SECTIONS + 1 }>::new();
    for index in 0..image.load_region_count() {
        validated
            .map(pe_mapping(image, loaded, index, user)?)
            .map_err(|_| KvmError::InvalidMapping)?;
    }
    for index in 0..image.load_region_count() {
        tables
            .map(pe_mapping(image, loaded, index, user)?)
            .map_err(|_| KvmError::PageTable)?;
    }
    Ok(())
}

fn pe_mapping(
    image: &PeImage<'_>,
    loaded: KvmLoadedImage,
    index: usize,
    user: bool,
) -> Result<Mapping, KvmError> {
    let region = image.load_region(index).map_err(KvmError::Pe)?;
    let offset = region.virtual_address() as u64;
    let virtual_address = loaded
        .virtual_base
        .checked_add(offset)
        .ok_or(KvmError::InvalidMapping)?;
    let physical_address = loaded
        .physical_base
        .checked_add(offset)
        .ok_or(KvmError::InvalidMapping)?;
    let permissions = match (user, region.writable(), region.executable()) {
        (true, true, false) => PagePermissions::USER_READ_WRITE,
        (true, false, true) => PagePermissions::USER_READ_EXECUTE,
        (true, false, false) => PagePermissions::USER_READ,
        (false, true, false) => PagePermissions::KERNEL_READ_WRITE,
        (false, false, true) => PagePermissions::KERNEL_READ_EXECUTE,
        (false, false, false) => PagePermissions::KERNEL_READ,
        (_, true, true) => return Err(KvmError::InvalidMapping),
    };
    Mapping::new(
        VirtAddr::new(virtual_address).map_err(|_| KvmError::InvalidMapping)?,
        PhysAddr::new(physical_address).map_err(|_| KvmError::InvalidMapping)?,
        region.pages() as u64,
        permissions,
    )
    .map_err(|_| KvmError::InvalidMapping)
}

impl<const N: usize> Default for KvmGuestMemory<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Page-table frame arena inside an existing writable guest-RAM region.
/// Only frames returned by `allocate_zeroed` may subsequently be accessed.
pub struct KvmPageTableStore<'a, const N: usize> {
    memory: &'a mut KvmGuestMemory<N>,
    start: u64,
    next: u64,
    end: u64,
}

impl<'a, const N: usize> KvmPageTableStore<'a, N> {
    pub fn new(
        memory: &'a mut KvmGuestMemory<N>,
        guest_start: u64,
        pages: u64,
    ) -> Result<Self, KvmError> {
        if pages == 0 || !guest_start.is_multiple_of(PAGE_SIZE) {
            return Err(KvmError::UnalignedMemory);
        }
        let bytes = pages
            .checked_mul(PAGE_SIZE)
            .ok_or(KvmError::MemoryOverflow)?;
        let bytes_usize = usize::try_from(bytes).map_err(|_| KvmError::MemoryOverflow)?;
        let (region, _) = memory.locate(guest_start, bytes_usize)?;
        if region.readonly {
            return Err(KvmError::ReadOnlyMemory);
        }
        let end = guest_start
            .checked_add(bytes)
            .ok_or(KvmError::MemoryOverflow)?;
        Ok(Self {
            memory,
            start: guest_start,
            next: guest_start,
            end,
        })
    }

    fn allocated_entry(&self, table: PhysAddr, index: usize) -> Result<u64, PageTableBuildError> {
        if index >= 512
            || table.get() < self.start
            || table.get() >= self.next
            || !table.get().is_multiple_of(PAGE_SIZE)
        {
            return Err(PageTableBuildError::Storage);
        }
        table
            .get()
            .checked_add((index as u64) * 8)
            .ok_or(PageTableBuildError::AddressOverflow)
    }

    pub(crate) const fn allocated_pages(&self) -> u64 {
        (self.next - self.start) / PAGE_SIZE
    }

    pub(crate) fn allocated_frame(&self, index: u64) -> Option<PhysAddr> {
        (index < self.allocated_pages())
            .then(|| self.start + index * PAGE_SIZE)
            .and_then(|address| PhysAddr::new(address).ok())
    }
}

impl<const N: usize> PageTableStore for KvmPageTableStore<'_, N> {
    fn allocate_zeroed(&mut self) -> Result<PhysAddr, PageTableBuildError> {
        let next = self
            .next
            .checked_add(PAGE_SIZE)
            .ok_or(PageTableBuildError::AddressOverflow)?;
        if next > self.end {
            return Err(PageTableBuildError::Storage);
        }
        let frame = PhysAddr::new(self.next).map_err(|_| PageTableBuildError::Storage)?;
        let zero = [0u8; PAGE_SIZE as usize];
        self.memory
            .write(self.next, &zero)
            .map_err(|_| PageTableBuildError::Storage)?;
        self.next = next;
        Ok(frame)
    }

    fn read(&self, table: PhysAddr, index: usize) -> Result<u64, PageTableBuildError> {
        let address = self.allocated_entry(table, index)?;
        let mut encoded = [0u8; 8];
        self.memory
            .read(address, &mut encoded)
            .map_err(|_| PageTableBuildError::Storage)?;
        Ok(u64::from_le_bytes(encoded))
    }

    fn write(
        &mut self,
        table: PhysAddr,
        index: usize,
        value: u64,
    ) -> Result<(), PageTableBuildError> {
        let address = self.allocated_entry(table, index)?;
        self.memory
            .write(address, &value.to_le_bytes())
            .map_err(|_| PageTableBuildError::Storage)
    }
}

pub(crate) struct KvmBackend<const N: usize> {
    vcpu: KvmVcpu,
    secondary: Option<KvmVcpu>,
    memory: KvmGuestMemory<N>,
    vm: KvmVm,
    vcpu_id: u32,
}

impl<const N: usize> KvmBackend<N> {
    pub(crate) fn run_smp(&mut self) -> Result<(VmExit, VmExit), KvmError> {
        let mut application = self.secondary.take().ok_or(KvmError::InvalidVcpu)?;
        let cancellation = application.cancellation();
        let result = mrml_runtime::Shared::new(mrml_runtime::SpinMutex::new(None));
        let worker_result = result.clone();
        mrml_runtime::spawn_detached(move || {
            let deadline = mrml_runtime::Instant::now();
            let exit = loop {
                match application.mp_state() {
                    Ok(KVM_MP_STATE_UNINITIALIZED)
                        if deadline.elapsed() < Duration::from_secs(2) =>
                    {
                        mrml_runtime::yield_now();
                    }
                    Ok(KVM_MP_STATE_UNINITIALIZED) => break Err(KvmError::InvalidVcpu),
                    Ok(_) => break application.run(),
                    Err(error) => break Err(error),
                }
            };
            *worker_result.lock() = Some(exit);
        })
        .map_err(|_| KvmError::SystemCall)?;

        let bootstrap = loop {
            let exit = self.vcpu.run()?;
            if let VmExit::Io {
                port: 0x00e9,
                size: 1,
                write: true,
                value,
            } = exit
            {
                mrml_runtime::mrml_println!("KVM_SMP_TRACE stage={:#04x}", value);
                continue;
            }
            break Ok(exit);
        };
        let deadline = mrml_runtime::Instant::now();
        while result.lock().is_none() && deadline.elapsed() < Duration::from_secs(2) {
            mrml_runtime::yield_now();
        }
        if result.lock().is_none() {
            cancellation.request();
        }
        let secondary = loop {
            if let Some(exit) = result.lock().take() {
                break exit;
            }
            mrml_runtime::yield_now();
        };
        Ok((bootstrap?, secondary?))
    }
    pub fn map_memory(
        &mut self,
        guest_address: u64,
        bytes: usize,
        readonly: bool,
    ) -> Result<(), KvmError> {
        let (slot, host) = self.memory.allocate(guest_address, bytes, readonly)?;
        let region = KvmMemoryRegion::new(slot, guest_address, bytes as u64, host, readonly)?;
        if let Err(error) = self.vm.register_memory(region) {
            self.memory.rollback_last();
            return Err(error);
        }
        Ok(())
    }

    pub fn page_tables(
        &mut self,
        guest_start: u64,
        pages: u64,
    ) -> Result<PageTableBuilder<KvmPageTableStore<'_, N>>, KvmError> {
        let store = KvmPageTableStore::new(&mut self.memory, guest_start, pages)?;
        PageTableBuilder::new(store).map_err(|_| KvmError::InvalidRegisterState)
    }

    pub(crate) fn snapshot(&self) -> Result<KvmVcpuSnapshot, KvmError> {
        self.vcpu.snapshot()
    }
}

impl<const N: usize> VmBackend for KvmBackend<N> {
    type Error = KvmError;

    fn run(&mut self, vcpu: u32) -> Result<VmExit, Self::Error> {
        if vcpu != self.vcpu_id {
            return Err(KvmError::InvalidVcpu);
        }
        self.vcpu.run()
    }

    fn read_guest(&self, guest_address: u64, output: &mut [u8]) -> Result<(), Self::Error> {
        self.memory.read(guest_address, output)
    }

    fn write_guest(&mut self, guest_address: u64, input: &[u8]) -> Result<(), Self::Error> {
        self.memory.write(guest_address, input)
    }

    fn inject_interrupt(&mut self, vcpu: u32, vector: u8) -> Result<(), Self::Error> {
        if vcpu != self.vcpu_id {
            return Err(KvmError::InvalidVcpu);
        }
        let interrupt = u32::from(vector);
        if unsafe { ioctl(self.vcpu.file.0, KVM_INTERRUPT, &interrupt) } < 0 {
            return Err(KvmError::SystemCall);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_pe() -> [u8; 1024] {
        let mut pe = [0u8; 1024];
        pe[0..2].copy_from_slice(&0x5a4du16.to_le_bytes());
        pe[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        pe[0x80..0x84].copy_from_slice(&0x0000_4550u32.to_le_bytes());
        pe[0x84..0x86].copy_from_slice(&0x8664u16.to_le_bytes());
        pe[0x86..0x88].copy_from_slice(&1u16.to_le_bytes());
        pe[0x94..0x96].copy_from_slice(&240u16.to_le_bytes());
        pe[0x96..0x98].copy_from_slice(&2u16.to_le_bytes());
        let optional = 0x98;
        pe[optional..optional + 2].copy_from_slice(&0x20bu16.to_le_bytes());
        pe[optional + 16..optional + 20].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[optional + 24..optional + 32].copy_from_slice(&0x0000_0001_4000_0000u64.to_le_bytes());
        pe[optional + 32..optional + 36].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[optional + 36..optional + 40].copy_from_slice(&0x200u32.to_le_bytes());
        pe[optional + 56..optional + 60].copy_from_slice(&0x2000u32.to_le_bytes());
        pe[optional + 60..optional + 64].copy_from_slice(&0x200u32.to_le_bytes());
        pe[optional + 70..optional + 72].copy_from_slice(&0x100u16.to_le_bytes());
        let section = optional + 240;
        pe[section..section + 5].copy_from_slice(b".text");
        pe[section + 8..section + 12].copy_from_slice(&0x20u32.to_le_bytes());
        pe[section + 12..section + 16].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[section + 16..section + 20].copy_from_slice(&0x200u32.to_le_bytes());
        pe[section + 20..section + 24].copy_from_slice(&0x200u32.to_le_bytes());
        pe[section + 36..section + 40].copy_from_slice(&0x6000_0000u32.to_le_bytes());
        pe[0x200] = 0xcc;
        pe
    }

    fn valid_handoff() -> [u8; 240] {
        let mut encoded = [0u8; 240];
        encoded[..16].copy_from_slice(b"MRML-HANDOFF-v1\0");
        encoded[16..20].copy_from_slice(&240u32.to_le_bytes());
        encoded[20..22].copy_from_slice(&3u16.to_le_bytes());
        encoded[22..24].copy_from_slice(&7u16.to_le_bytes());
        encoded[24..32].copy_from_slice(&7u64.to_le_bytes());
        encoded[32..64].fill(1);
        encoded[64..128].fill(2);
        encoded[128..136].copy_from_slice(&0x9000u64.to_le_bytes());
        encoded[136..144].copy_from_slice(&0xa0000u64.to_le_bytes());
        encoded[144..152].copy_from_slice(&0x1000u64.to_le_bytes());
        encoded[152..156].copy_from_slice(&16u32.to_le_bytes());
        encoded[156..160].copy_from_slice(&16u32.to_le_bytes());
        encoded[160..164].copy_from_slice(&16u32.to_le_bytes());
        encoded[164] = 1;
        encoded[168..176].copy_from_slice(&0x1000u64.to_le_bytes());
        encoded[176..184].copy_from_slice(&2u64.to_le_bytes());
        encoded[184] = 0;
        encoded[192..200].copy_from_slice(&0x3000u64.to_le_bytes());
        encoded[200..208].copy_from_slice(&1u64.to_le_bytes());
        encoded[208] = 1;
        encoded[216..224].copy_from_slice(&0xa0000u64.to_le_bytes());
        encoded[224..232].copy_from_slice(&1u64.to_le_bytes());
        encoded[232] = 3;
        encoded
    }

    #[test]
    fn ioctl_numbers_match_x86_64_kvm_uapi() {
        assert_eq!(KVM_GET_API_VERSION, 0xae00);
        assert_eq!(KVM_CREATE_VM, 0xae01);
        assert_eq!(KVM_CHECK_EXTENSION, 0xae03);
        assert_eq!(KVM_GET_SUPPORTED_CPUID, 0xc008_ae05);
        assert_eq!(KVM_SET_USER_MEMORY_REGION, 0x4020_ae46);
        assert_eq!(KVM_CREATE_VCPU, 0xae41);
        assert_eq!(KVM_SET_MP_STATE, 0x4004_ae99);
        assert_eq!(KVM_CREATE_IRQCHIP, 0xae60);
        assert_eq!(KVM_RUN, 0xae80);
        assert_eq!(KVM_GET_REGS, 0x8090_ae81);
        assert_eq!(KVM_GET_SREGS, 0x8138_ae83);
        assert_eq!(KVM_SET_REGS, 0x4090_ae82);
        assert_eq!(KVM_SET_SREGS, 0x4138_ae84);
        assert_eq!(KVM_SET_CPUID2, 0x4008_ae90);
        assert_eq!(core::mem::size_of::<KvmCpuidEntry2>(), 40);
        assert_eq!(core::mem::size_of::<KvmCpuidSet>(), 5_128);
        assert_eq!(core::mem::size_of::<KvmRegisters>(), 144);
        assert_eq!(core::mem::size_of::<KvmSegment>(), 24);
        assert_eq!(core::mem::size_of::<KvmSpecialRegisters>(), 312);
    }

    #[test]
    fn required_capabilities_fail_closed_before_vm_creation() {
        assert_eq!(
            validate_capabilities(0, 32),
            Err(KvmError::UnsupportedCapability(KVM_CAP_USER_MEMORY))
        );
        assert_eq!(
            validate_capabilities(1, 4),
            Err(KvmError::InsufficientMemorySlots(4))
        );
        assert_eq!(validate_capabilities(1, 5), Ok(()));
    }

    #[test]
    fn live_api_version_when_kvm_is_available() {
        match KvmSystem::open() {
            Ok(system) => assert!((MIN_RUN_BYTES..=MAX_RUN_BYTES).contains(&system.run_bytes)),
            Err(KvmError::SystemCall) => {}
            Err(error) => panic!("available KVM failed validation: {error:?}"),
        }
    }

    #[test]
    fn owned_guest_memory_is_bounded_and_readonly() {
        let mut memory = KvmGuestMemory::<2>::new();
        memory.allocate(0x1000, 0x2000, false).unwrap();
        memory.write(0x1fff, &[1, 2]).unwrap();
        let mut output = [0u8; 2];
        memory.read(0x1fff, &mut output).unwrap();
        assert_eq!(output, [1, 2]);
        assert_eq!(
            memory.read(0x2fff, &mut output),
            Err(KvmError::UnmappedMemory)
        );
        assert_eq!(
            memory.allocate(0x2000, 0x1000, false),
            Err(KvmError::MemoryOverlap)
        );
        memory.allocate(0x4000, 0x1000, true).unwrap();
        assert_eq!(memory.write(0x4000, &[1]), Err(KvmError::ReadOnlyMemory));
    }

    #[test]
    fn initial_register_state_enables_hardened_long_mode() {
        let registers =
            KvmRegisters::initial(0x20_0000, 0x40_0000, 0x50_0000, 240, 0x60_0000, 0x70_0000);
        assert_eq!(registers.rip, 0x20_0000);
        assert_eq!(registers.rsp, 0x40_0000);
        assert_eq!(registers.rflags, 2);
        assert_eq!((registers.rcx, registers.rdx), (0x50_0000, 240));
        assert_eq!((registers.r8, registers.r9), (0x60_0000, 0x70_0000));
        assert!(canonical(0xffff_8000_0000_0000));
        assert!(!canonical(0x0001_0000_0000_0000));
        let code = KvmSegment::code64();
        assert_eq!(
            (code.present, code.long, code.system, code.default_operand),
            (1, 1, 1, 0)
        );
    }

    #[test]
    fn guest_page_table_store_bounds_frames_and_builds_wx_entries() {
        use mrml_kernel::arch::x86_64::{PagePermissions, VirtAddr};

        let mut memory = KvmGuestMemory::<1>::new();
        memory.allocate(0x10_0000, 0x8000, false).unwrap();
        let store = KvmPageTableStore::new(&mut memory, 0x10_0000, 8).unwrap();
        let mut tables = PageTableBuilder::new(store).unwrap();
        tables
            .map_page(
                VirtAddr::new(0x20_0000).unwrap(),
                PhysAddr::new(0x30_0000).unwrap(),
                PagePermissions::KERNEL_READ_EXECUTE,
            )
            .unwrap();
        assert_eq!(tables.root().get(), 0x10_0000);
        assert_eq!(
            tables.store().read(PhysAddr::new(0x10_7000).unwrap(), 0),
            Err(PageTableBuildError::Storage)
        );
    }

    #[test]
    fn pe_is_materialized_then_mapped_with_validated_permissions() {
        let encoded = valid_pe();
        let image = PeImage::parse(&encoded).unwrap();
        let mut memory = KvmGuestMemory::<2>::new();
        memory.allocate(0x10_0000, 0x8000, false).unwrap();
        memory.allocate(0x20_0000, 0x2000, false).unwrap();
        let loaded = memory
            .load_pe(&image, 0x20_0000, image.image_base())
            .unwrap();
        assert_eq!(loaded.entry(), image.image_base() + 0x1000);
        let mut opcode = [0u8; 1];
        memory.read(0x20_1000, &mut opcode).unwrap();
        assert_eq!(opcode, [0xcc]);
        let store = KvmPageTableStore::new(&mut memory, 0x10_0000, 8).unwrap();
        let mut tables = PageTableBuilder::new(store).unwrap();
        map_pe(&mut tables, &image, loaded, true).unwrap();
    }

    #[test]
    fn handoff_is_validated_before_copy_and_mapped_read_only() {
        let mut memory = KvmGuestMemory::<2>::new();
        memory.allocate(0x10_0000, 0x8000, false).unwrap();
        memory.allocate(0x20_0000, 0x1000, false).unwrap();
        memory.write(0x20_0000, &[0xaa]).unwrap();
        let mut malformed = valid_handoff();
        malformed[0] = 0;
        assert!(matches!(
            memory.load_boot_handoff(&malformed, 0x20_0000),
            Err(KvmError::Handoff(_))
        ));
        let mut marker = [0u8; 1];
        memory.read(0x20_0000, &mut marker).unwrap();
        assert_eq!(marker, [0xaa]);

        let encoded = valid_handoff();
        let loaded = memory.load_boot_handoff(&encoded, 0x20_0000).unwrap();
        assert_eq!((loaded.bytes(), loaded.pages()), (240, 1));
        let mut tail = [1u8; 1];
        memory.read(0x20_0fff, &mut tail).unwrap();
        assert_eq!(tail, [0]);
        let store = KvmPageTableStore::new(&mut memory, 0x10_0000, 8).unwrap();
        let mut tables = PageTableBuilder::new(store).unwrap();
        map_loaded_handoff(&mut tables, loaded, 0x40_0000, true).unwrap();
    }
}
