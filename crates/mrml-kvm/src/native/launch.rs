use mrml_kernel::arch::x86_64::{Mapping, PagePermissions, VirtAddr};
use mrml_kernel::{
    BootHandoff, GpuSharedQueueLayout, GpuVmmMemory, PAGE_SIZE, PhysAddr, VerifiedExecutable,
    VmBackend, VmExit,
};

use super::{KvmBackend, KvmError, KvmSystem, KvmVcpuSnapshot, map_loaded_handoff, map_loaded_pe};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvmLaunchLayout {
    table_physical: u64,
    table_pages: u64,
    image_physical: u64,
    image_virtual: u64,
    handoff_physical: u64,
    handoff_virtual: u64,
    stack_physical: u64,
    stack_virtual: u64,
    stack_pages: u64,
    user: bool,
}

impl KvmLaunchLayout {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        table_physical: u64,
        table_pages: u64,
        image_physical: u64,
        image_virtual: u64,
        handoff_physical: u64,
        handoff_virtual: u64,
        stack_physical: u64,
        stack_virtual: u64,
        stack_pages: u64,
        user: bool,
    ) -> Result<Self, KvmError> {
        if table_pages < 4
            || stack_pages == 0
            || [
                table_physical,
                image_physical,
                handoff_physical,
                stack_physical,
            ]
            .iter()
            .any(|address| *address == 0 || address % PAGE_SIZE != 0)
            || [image_virtual, handoff_virtual, stack_virtual]
                .iter()
                .any(|address| *address == 0 || address % PAGE_SIZE != 0)
        {
            return Err(KvmError::InvalidMapping);
        }
        Ok(Self {
            table_physical,
            table_pages,
            image_physical,
            image_virtual,
            handoff_physical,
            handoff_virtual,
            stack_physical,
            stack_virtual,
            stack_pages,
            user,
        })
    }
}

pub struct PreparedKvmGuest<const N: usize> {
    backend: KvmBackend<N>,
    entry: u64,
    root: PhysAddr,
}

impl<const N: usize> PreparedKvmGuest<N> {
    pub const fn entry(&self) -> u64 {
        self.entry
    }
    pub const fn page_table_root(&self) -> PhysAddr {
        self.root
    }
    pub fn snapshot(&self) -> Result<KvmVcpuSnapshot, KvmError> {
        self.backend.snapshot()
    }
    pub fn page_walk(&self, virtual_address: u64) -> Result<KvmPageWalk, KvmError> {
        if ((virtual_address << 16) as i64 >> 16) as u64 != virtual_address {
            return Err(KvmError::InvalidMapping);
        }
        let indexes = [
            ((virtual_address >> 39) & 0x1ff) as usize,
            ((virtual_address >> 30) & 0x1ff) as usize,
            ((virtual_address >> 21) & 0x1ff) as usize,
            ((virtual_address >> 12) & 0x1ff) as usize,
        ];
        let mut entries = [0u64; 4];
        let mut table = self.root.get();
        let mut levels = 0u8;
        for (level, index) in indexes.into_iter().enumerate() {
            let address = table
                .checked_add((index as u64) * 8)
                .ok_or(KvmError::MemoryOverflow)?;
            let mut encoded = [0u8; 8];
            self.backend.read_guest(address, &mut encoded)?;
            let entry = u64::from_le_bytes(encoded);
            entries[level] = entry;
            levels += 1;
            if entry & 1 == 0 || (level < 3 && entry & (1 << 7) != 0) {
                break;
            }
            table = entry & 0x000f_ffff_ffff_f000;
        }
        Ok(KvmPageWalk { entries, levels })
    }

    /// Registers the two validated mediated-GPU rings as dedicated KVM memory
    /// slots. The command ring is guest-writable; the completion ring is KVM
    /// read-only to the guest and remains writable by the isolated host GPU
    /// service. Consuming `self` makes failure transactional from the caller's
    /// perspective: a partially attached VM is dropped and cannot be run.
    pub fn attach_gpu_queue_memory(
        mut self,
        layout: GpuSharedQueueLayout,
    ) -> Result<Self, KvmError> {
        let bytes = layout
            .pages_per_ring()
            .checked_mul(PAGE_SIZE)
            .ok_or(KvmError::MemoryOverflow)?;
        let bytes = usize::try_from(bytes).map_err(|_| KvmError::MemoryOverflow)?;
        self.backend
            .map_memory(layout.command_base(), bytes, false)?;
        self.backend
            .map_memory(layout.completion_base(), bytes, true)?;
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvmPageWalk {
    entries: [u64; 4],
    levels: u8,
}

impl KvmPageWalk {
    pub const fn entries(self) -> [u64; 4] {
        self.entries
    }
    pub const fn levels(self) -> u8 {
        self.levels
    }
    pub fn physical_address(self, virtual_address: u64) -> Option<u64> {
        if self.levels != 4 || self.entries[3] & 1 == 0 {
            return None;
        }
        (self.entries[3] & 0x000f_ffff_ffff_f000).checked_add(virtual_address & 0xfff)
    }
}

impl<const N: usize> VmBackend for PreparedKvmGuest<N> {
    type Error = KvmError;
    fn run(&mut self, vcpu: u32) -> Result<VmExit, Self::Error> {
        self.backend.run(vcpu)
    }
    fn read_guest(&self, address: u64, output: &mut [u8]) -> Result<(), Self::Error> {
        self.backend.read_guest(address, output)
    }
    fn write_guest(&mut self, address: u64, input: &[u8]) -> Result<(), Self::Error> {
        self.backend.write_guest(address, input)
    }
    fn inject_interrupt(&mut self, vcpu: u32, vector: u8) -> Result<(), Self::Error> {
        self.backend.inject_interrupt(vcpu, vector)
    }
}

impl<const N: usize> GpuVmmMemory for PreparedKvmGuest<N> {
    fn write_gpu_service(&mut self, address: u64, input: &[u8]) -> Result<(), Self::Error> {
        self.backend.memory.write_service(address, input)
    }
}

impl KvmSystem {
    pub fn prepare_guest<const N: usize>(
        &self,
        vcpu_id: u32,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: KvmLaunchLayout,
    ) -> Result<PreparedKvmGuest<N>, KvmError> {
        self.prepare_guest_inner(vcpu_id, executable, handoff, layout, false, None)
    }

    /// Prepares a kernel guest and maps the authenticated handoff framebuffer
    /// as writable, non-executable memory at its identity address.
    pub fn prepare_kernel_guest<const N: usize>(
        &self,
        vcpu_id: u32,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: KvmLaunchLayout,
    ) -> Result<PreparedKvmGuest<N>, KvmError> {
        self.prepare_guest_inner(vcpu_id, executable, handoff, layout, true, None)
    }

    /// Prepares a kernel guest with its authenticated framebuffer and mediated
    /// GPU rings present in the initial address space. Both rings are identity
    /// mapped NX; only the command ring is guest-writable.
    pub fn prepare_kernel_gpu_guest<const N: usize>(
        &self,
        vcpu_id: u32,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: KvmLaunchLayout,
        queue: GpuSharedQueueLayout,
    ) -> Result<PreparedKvmGuest<N>, KvmError> {
        self.prepare_guest_inner(vcpu_id, executable, handoff, layout, true, Some(queue))
    }

    fn prepare_guest_inner<const N: usize>(
        &self,
        vcpu_id: u32,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: KvmLaunchLayout,
        map_framebuffer: bool,
        queue: Option<GpuSharedQueueLayout>,
    ) -> Result<PreparedKvmGuest<N>, KvmError> {
        if N < 4 + usize::from(map_framebuffer) + usize::from(queue.is_some()) * 2 {
            return Err(KvmError::MemoryTableFull);
        }
        let decoded = BootHandoff::decode(handoff, |_| {}).map_err(KvmError::Handoff)?;
        let image_bytes = page_bytes(executable.image().image_size() as u64)?;
        let handoff_bytes = page_bytes(handoff.len() as u64)?;
        let table_bytes = layout
            .table_pages
            .checked_mul(PAGE_SIZE)
            .ok_or(KvmError::MemoryOverflow)?;
        let stack_bytes = layout
            .stack_pages
            .checked_mul(PAGE_SIZE)
            .ok_or(KvmError::MemoryOverflow)?;
        let framebuffer = decoded.framebuffer();
        let framebuffer_bytes = page_bytes(framebuffer.byte_length())?;
        let queue_bytes = match queue {
            Some(value) => value
                .pages_per_ring()
                .checked_mul(PAGE_SIZE)
                .ok_or(KvmError::MemoryOverflow)?,
            None => PAGE_SIZE,
        };
        let (command_base, completion_base) = queue
            .map(|value| (value.command_base(), value.completion_base()))
            .unwrap_or((0, 0));
        let physical_ranges = [
            (layout.table_physical, table_bytes),
            (layout.image_physical, image_bytes),
            (layout.handoff_physical, handoff_bytes),
            (layout.stack_physical, stack_bytes),
            (framebuffer.base().get(), framebuffer_bytes),
            (command_base, queue_bytes),
            (completion_base, queue_bytes),
        ];
        let physical_count = 4 + usize::from(map_framebuffer) + usize::from(queue.is_some()) * 2;
        validate_ranges(&physical_ranges[..physical_count])?;
        let virtual_ranges = [
            (layout.image_virtual, image_bytes),
            (layout.handoff_virtual, handoff_bytes),
            (layout.stack_virtual, stack_bytes),
            (command_base, queue_bytes),
            (completion_base, queue_bytes),
        ];
        validate_ranges(&virtual_ranges[..3 + usize::from(queue.is_some()) * 2])?;

        let mut backend = self.create_backend::<N>(vcpu_id)?;
        backend.map_memory(layout.table_physical, table_bytes as usize, false)?;
        backend.map_memory(layout.image_physical, image_bytes as usize, false)?;
        backend.map_memory(layout.handoff_physical, handoff_bytes as usize, false)?;
        backend.map_memory(layout.stack_physical, stack_bytes as usize, false)?;
        if map_framebuffer {
            backend.map_memory(framebuffer.base().get(), framebuffer_bytes as usize, false)?;
        }
        if let Some(queue) = queue {
            let bytes = usize::try_from(queue_bytes).map_err(|_| KvmError::MemoryOverflow)?;
            backend.map_memory(queue.command_base(), bytes, false)?;
            backend.map_memory(queue.completion_base(), bytes, true)?;
        }
        let loaded_image = backend.memory.load_verified_executable(
            executable,
            layout.image_physical,
            layout.image_virtual,
        )?;
        let loaded_handoff = backend
            .memory
            .load_boot_handoff(handoff, layout.handoff_physical)?;
        let stack_physical_end = layout
            .stack_physical
            .checked_add(stack_bytes)
            .ok_or(KvmError::MemoryOverflow)?;
        backend
            .memory
            .write(stack_physical_end - 8, &0u64.to_le_bytes())?;
        let stack = layout
            .stack_virtual
            .checked_add(stack_bytes)
            .and_then(|end| end.checked_sub(8))
            .ok_or(KvmError::MemoryOverflow)?;
        let mut tables = backend.page_tables(layout.table_physical, layout.table_pages)?;
        map_loaded_pe(&mut tables, executable, loaded_image, layout.user)?;
        map_loaded_handoff(
            &mut tables,
            loaded_handoff,
            layout.handoff_virtual,
            layout.user,
        )?;
        let stack_permissions = if layout.user {
            PagePermissions::USER_READ_WRITE
        } else {
            PagePermissions::KERNEL_READ_WRITE
        };
        tables
            .map(
                Mapping::new(
                    VirtAddr::new(layout.stack_virtual).map_err(|_| KvmError::InvalidMapping)?,
                    PhysAddr::new(layout.stack_physical).map_err(|_| KvmError::InvalidMapping)?,
                    layout.stack_pages,
                    stack_permissions,
                )
                .map_err(|_| KvmError::InvalidMapping)?,
            )
            .map_err(|_| KvmError::PageTable)?;
        if map_framebuffer {
            tables
                .map(
                    Mapping::new(
                        VirtAddr::new(framebuffer.base().get())
                            .map_err(|_| KvmError::InvalidMapping)?,
                        PhysAddr::new(framebuffer.base().get())
                            .map_err(|_| KvmError::InvalidMapping)?,
                        framebuffer_bytes / PAGE_SIZE,
                        PagePermissions::KERNEL_MMIO_READ_WRITE,
                    )
                    .map_err(|_| KvmError::InvalidMapping)?,
                )
                .map_err(|_| KvmError::PageTable)?;
        }
        if let Some(queue) = queue {
            tables
                .map(
                    Mapping::new(
                        VirtAddr::new(queue.command_base())
                            .map_err(|_| KvmError::InvalidMapping)?,
                        PhysAddr::new(queue.command_base())
                            .map_err(|_| KvmError::InvalidMapping)?,
                        queue.pages_per_ring(),
                        PagePermissions::KERNEL_SHARED_READ_WRITE,
                    )
                    .map_err(|_| KvmError::InvalidMapping)?,
                )
                .map_err(|_| KvmError::PageTable)?;
            tables
                .map(
                    Mapping::new(
                        VirtAddr::new(queue.completion_base())
                            .map_err(|_| KvmError::InvalidMapping)?,
                        PhysAddr::new(queue.completion_base())
                            .map_err(|_| KvmError::InvalidMapping)?,
                        queue.pages_per_ring(),
                        PagePermissions::KERNEL_SHARED_READ,
                    )
                    .map_err(|_| KvmError::InvalidMapping)?,
                )
                .map_err(|_| KvmError::PageTable)?;
        }
        let root = tables.root();
        let _ = tables.into_store();
        backend.vcpu.configure_long_mode_entry(
            loaded_image.entry(),
            stack,
            root.get(),
            layout.handoff_virtual,
            loaded_handoff.bytes() as u64,
        )?;
        Ok(PreparedKvmGuest {
            backend,
            entry: loaded_image.entry(),
            root,
        })
    }
}

fn page_bytes(bytes: u64) -> Result<u64, KvmError> {
    if bytes == 0 {
        return Err(KvmError::EmptyMemory);
    }
    bytes
        .checked_add(PAGE_SIZE - 1)
        .ok_or(KvmError::MemoryOverflow)
        .map(|value| value / PAGE_SIZE * PAGE_SIZE)
}

fn validate_ranges(ranges: &[(u64, u64)]) -> Result<(), KvmError> {
    for (index, &(start, bytes)) in ranges.iter().enumerate() {
        let end = start.checked_add(bytes).ok_or(KvmError::MemoryOverflow)?;
        for &(other_start, other_bytes) in &ranges[..index] {
            let other_end = other_start
                .checked_add(other_bytes)
                .ok_or(KvmError::MemoryOverflow)?;
            if start < other_end && other_start < end {
                return Err(KvmError::MemoryOverlap);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrml_crypto::{
        LAMPORT_PRIVATE_KEY_BYTES, LAMPORT_PUBLIC_KEY_BYTES, LAMPORT_SIGNATURE_BYTES, Sha3_512,
        lamport_public_key, lamport_sign,
    };
    use mrml_kernel::{
        ArtifactKind, SIGNED_ARTIFACT_HEADER_BYTES, SIGNED_ARTIFACT_OVERHEAD_BYTES, SignedArtifact,
        TrustRoot, artifact_statement,
    };

    fn framebuffer_pe() -> [u8; 1024] {
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
        pe[optional + 24..optional + 32].copy_from_slice(&0xffff_8001_4000_0000u64.to_le_bytes());
        pe[optional + 32..optional + 36].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[optional + 36..optional + 40].copy_from_slice(&0x200u32.to_le_bytes());
        pe[optional + 56..optional + 60].copy_from_slice(&0x2000u32.to_le_bytes());
        pe[optional + 60..optional + 64].copy_from_slice(&0x200u32.to_le_bytes());
        pe[optional + 70..optional + 72].copy_from_slice(&0x100u16.to_le_bytes());
        let section = optional + 240;
        pe[section..section + 5].copy_from_slice(b".text");
        pe[section + 8..section + 12].copy_from_slice(&17u32.to_le_bytes());
        pe[section + 12..section + 16].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[section + 16..section + 20].copy_from_slice(&0x200u32.to_le_bytes());
        pe[section + 20..section + 24].copy_from_slice(&0x200u32.to_le_bytes());
        pe[section + 36..section + 40].copy_from_slice(&0x6000_0000u32.to_le_bytes());
        // mov rax, 0xa0000; mov dword ptr [rax], 0x00ffc857; hlt
        pe[0x200..0x211].copy_from_slice(&[
            0x48, 0xb8, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc7, 0x00, 0x57, 0xc8,
            0xff, 0x00, 0xf4,
        ]);
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
        encoded[192..200].copy_from_slice(&0x3000u64.to_le_bytes());
        encoded[200..208].copy_from_slice(&1u64.to_le_bytes());
        encoded[208] = 1;
        encoded[216..224].copy_from_slice(&0xa0000u64.to_le_bytes());
        encoded[224..232].copy_from_slice(&1u64.to_le_bytes());
        encoded[232] = 3;
        encoded
    }

    fn signed_pe() -> [u8; SIGNED_ARTIFACT_OVERHEAD_BYTES + 1024] {
        let payload = framebuffer_pe();
        let mut private = [0u8; LAMPORT_PRIVATE_KEY_BYTES];
        for (index, byte) in private.iter_mut().enumerate() {
            *byte = (index as u64).wrapping_mul(61).wrapping_add(17) as u8;
        }
        let mut public = [0u8; LAMPORT_PUBLIC_KEY_BYTES];
        lamport_public_key(&private, &mut public).unwrap();
        let digest = Sha3_512::digest(&payload);
        let statement = artifact_statement(ArtifactKind::VmImage, 1, payload.len() as u64, digest);
        let mut signature = [0u8; LAMPORT_SIGNATURE_BYTES];
        lamport_sign(&private, &statement, &mut signature).unwrap();
        let mut encoded = [0u8; SIGNED_ARTIFACT_OVERHEAD_BYTES + 1024];
        encoded[..16].copy_from_slice(b"MRML-SIGNED-v1\0\0");
        encoded[16] = ArtifactKind::VmImage as u8;
        encoded[24..32].copy_from_slice(&1u64.to_le_bytes());
        encoded[32..40].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        encoded[40..104].copy_from_slice(&digest);
        let signature_at = SIGNED_ARTIFACT_HEADER_BYTES + LAMPORT_PUBLIC_KEY_BYTES;
        let payload_at = signature_at + LAMPORT_SIGNATURE_BYTES;
        encoded[SIGNED_ARTIFACT_HEADER_BYTES..signature_at].copy_from_slice(&public);
        encoded[signature_at..payload_at].copy_from_slice(&signature);
        encoded[payload_at..].copy_from_slice(&payload);
        encoded
    }

    #[test]
    fn launch_layout_rejects_small_table_arenas_and_overlaps() {
        assert_eq!(
            KvmLaunchLayout::new(
                0x1000,
                3,
                0x10000,
                0x140000000,
                0x20000,
                0x200000,
                0x30000,
                0x300000,
                2,
                true
            ),
            Err(KvmError::InvalidMapping)
        );
        assert_eq!(
            validate_ranges(&[(0x1000, 0x2000), (0x2000, 0x1000)]),
            Err(KvmError::MemoryOverlap)
        );
        assert!(
            KvmLaunchLayout::new(
                0x1000,
                8,
                0x10000,
                0x140000000,
                0x20000,
                0x200000,
                0x30000,
                0x300000,
                2,
                true
            )
            .is_ok()
        );
    }

    #[test]
    fn live_signed_high_half_pe_writes_authenticated_framebuffer() {
        let system = match KvmSystem::open() {
            Ok(system) => system,
            Err(KvmError::SystemCall) => return,
            Err(error) => panic!("available KVM failed validation: {error:?}"),
        };
        let encoded = signed_pe();
        let public_at = SIGNED_ARTIFACT_HEADER_BYTES;
        let public_end = public_at + LAMPORT_PUBLIC_KEY_BYTES;
        let root = TrustRoot::new(
            ArtifactKind::VmImage,
            Sha3_512::digest(&encoded[public_at..public_end]),
            1,
        );
        let signed = SignedArtifact::decode(&encoded).unwrap();
        let executable = signed
            .verify_executable(&root, ArtifactKind::VmImage)
            .unwrap();
        let layout = KvmLaunchLayout::new(
            0x10_0000,
            16,
            0x20_0000,
            0xffff_8001_4000_0000,
            0x30_0000,
            0xffff_8001_5000_0000,
            0x40_0000,
            0xffff_8001_6000_0000,
            2,
            false,
        )
        .unwrap();
        let queue_layout = GpuSharedQueueLayout::new(0x50_0000, 0x50_2000, 64).unwrap();
        let mut guest = system
            .prepare_kernel_gpu_guest::<7>(0, &executable, &valid_handoff(), layout, queue_layout)
            .unwrap();
        assert_eq!(guest.entry(), 0xffff_8001_4000_1000);
        let walk = guest.page_walk(guest.entry()).unwrap();
        assert_eq!(walk.levels(), 4);
        assert_eq!(walk.physical_address(guest.entry()), Some(0x20_1000));
        assert_eq!(walk.entries()[3] & (1 << 63), 0);
        assert_eq!(
            guest.page_walk(0x0000_8000_0000_0000),
            Err(KvmError::InvalidMapping)
        );
        let mut opcode = [0u8; 1];
        VmBackend::read_guest(&guest, 0x20_1000, &mut opcode).unwrap();
        assert_eq!(opcode, [0x48]);
        let command_walk = guest.page_walk(queue_layout.command_base()).unwrap();
        assert_eq!(
            command_walk.physical_address(queue_layout.command_base()),
            Some(queue_layout.command_base())
        );
        assert_ne!(command_walk.entries()[3] & (1 << 1), 0);
        assert_ne!(command_walk.entries()[3] & (1 << 63), 0);
        let completion_walk = guest.page_walk(queue_layout.completion_base()).unwrap();
        assert_eq!(
            completion_walk.physical_address(queue_layout.completion_base()),
            Some(queue_layout.completion_base())
        );
        assert_eq!(completion_walk.entries()[3] & (1 << 1), 0);
        assert_ne!(completion_walk.entries()[3] & (1 << 63), 0);
        VmBackend::write_guest(&mut guest, queue_layout.command_base(), &[1]).unwrap();
        assert_eq!(
            VmBackend::write_guest(&mut guest, queue_layout.completion_base(), &[1]),
            Err(KvmError::ReadOnlyMemory)
        );
        assert_eq!(VmBackend::run(&mut guest, 0), Ok(VmExit::Halted));
        let mut pixel = [0u8; 4];
        VmBackend::read_guest(&guest, 0xa0000, &mut pixel).unwrap();
        assert_eq!(pixel, [0x57, 0xc8, 0xff, 0x00]);
    }
}
