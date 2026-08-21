use mrml_kernel::arch::x86_64::{
    AddressSpace, Mapping, PagePermissions, PageTableBuildError, PageTableBuilder, PageTableStore,
    VirtAddr,
};
use mrml_kernel::{
    BootHandoff, MAX_PE_SECTIONS, PAGE_SIZE, PeImage, PhysAddr, VerifiedExecutable, VmBackend,
    VmExit,
};

use crate::{GuestRange, MapPermissions, WhpError};

use super::{PreparedWhpPartition, WhpSystem};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WhpLaunchLayout {
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

impl WhpLaunchLayout {
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
    ) -> Result<Self, WhpError> {
        if table_pages < 5
            || stack_pages == 0
            || [
                table_physical,
                image_physical,
                handoff_physical,
                stack_physical,
            ]
            .iter()
            .any(|value| *value == 0 || !value.is_multiple_of(PAGE_SIZE))
            || [image_virtual, handoff_virtual, stack_virtual]
                .iter()
                .any(|value| *value == 0 || !value.is_multiple_of(PAGE_SIZE))
        {
            return Err(WhpError::InvalidMapping);
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

pub struct PreparedWhpGuest<'system> {
    partition: PreparedWhpPartition<'system>,
    entry: u64,
    root: PhysAddr,
}

impl PreparedWhpGuest<'_> {
    pub const fn entry(&self) -> u64 {
        self.entry
    }
    pub const fn page_table_root(&self) -> PhysAddr {
        self.root
    }
    pub fn run(&mut self) -> Result<VmExit, WhpError> {
        self.partition.run()
    }
    pub fn read_guest(&self, address: u64, output: &mut [u8]) -> Result<(), WhpError> {
        self.partition.read_guest(address, output)
    }
    pub fn inject_interrupt(&mut self, vector: u8) -> Result<(), WhpError> {
        self.partition.inject_interrupt(vector)
    }
}

impl VmBackend for PreparedWhpGuest<'_> {
    type Error = WhpError;

    fn run(&mut self, vcpu: u32) -> Result<VmExit, Self::Error> {
        if vcpu != 0 {
            return Err(WhpError::InvalidVcpu);
        }
        self.partition.run()
    }

    fn read_guest(&self, address: u64, output: &mut [u8]) -> Result<(), Self::Error> {
        self.partition.read_guest(address, output)
    }

    fn write_guest(&mut self, address: u64, input: &[u8]) -> Result<(), Self::Error> {
        self.partition.write_guest(address, input)
    }

    fn inject_interrupt(&mut self, vcpu: u32, vector: u8) -> Result<(), Self::Error> {
        if vcpu != 0 {
            return Err(WhpError::InvalidVcpu);
        }
        self.partition.inject_interrupt(vector)
    }
}

impl WhpSystem {
    pub fn prepare_guest<'system>(
        &'system self,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: WhpLaunchLayout,
    ) -> Result<PreparedWhpGuest<'system>, WhpError> {
        BootHandoff::decode(handoff, |_| {}).map_err(WhpError::Handoff)?;
        let image_bytes = page_bytes(executable.image().image_size() as u64)?;
        let handoff_bytes = page_bytes(handoff.len() as u64)?;
        let table_bytes = layout
            .table_pages
            .checked_mul(PAGE_SIZE)
            .ok_or(WhpError::MemoryOverflow)?;
        let stack_bytes = layout
            .stack_pages
            .checked_mul(PAGE_SIZE)
            .ok_or(WhpError::MemoryOverflow)?;
        validate_ranges(&[
            (layout.table_physical, table_bytes),
            (layout.image_physical, image_bytes),
            (layout.handoff_physical, handoff_bytes),
            (layout.stack_physical, stack_bytes),
        ])?;
        validate_ranges(&[
            (layout.table_physical, PAGE_SIZE),
            (layout.image_virtual, image_bytes),
            (layout.handoff_virtual, handoff_bytes),
            (layout.stack_virtual, stack_bytes),
        ])?;

        let mut partition = self.prepare_partition()?;
        partition.map_zeroed(range(
            layout.table_physical,
            table_bytes,
            MapPermissions::read_write(),
        )?)?;
        let image_mapping = partition.map_zeroed(range(
            layout.image_physical,
            image_bytes,
            MapPermissions::read_write(),
        )?)?;
        partition.map_initialized(
            range(
                layout.handoff_physical,
                handoff_bytes,
                MapPermissions::read_only(),
            )?,
            handoff,
        )?;
        partition.map_zeroed(range(
            layout.stack_physical,
            stack_bytes,
            MapPermissions::read_write(),
        )?)?;

        let image = executable.image();
        let destination =
            partition.mutable_guest(layout.image_physical, image.image_size() as usize)?;
        let entry = image
            .materialize_at(destination, layout.image_virtual)
            .map_err(WhpError::Pe)?;
        partition.seal_pe(image_mapping, image)?;

        let stack = layout
            .stack_virtual
            .checked_add(stack_bytes)
            .and_then(|end| end.checked_sub(8))
            .ok_or(WhpError::MemoryOverflow)?;
        partition.write_guest(layout.stack_physical + stack_bytes - 8, &0u64.to_le_bytes())?;

        let root = build_page_tables(&mut partition, image, layout, handoff_bytes)?;
        partition.configure_long_mode(
            entry,
            stack,
            root.get(),
            layout.table_physical,
            layout.handoff_virtual,
            handoff.len() as u64,
        )?;
        Ok(PreparedWhpGuest {
            partition,
            entry,
            root,
        })
    }
}

fn build_page_tables(
    partition: &mut PreparedWhpPartition<'_>,
    image: &PeImage<'_>,
    layout: WhpLaunchLayout,
    handoff_bytes: u64,
) -> Result<PhysAddr, WhpError> {
    partition.write_guest(layout.table_physical, &0u64.to_le_bytes())?;
    partition.write_guest(
        layout.table_physical + 8,
        &0x00af_9b00_0000_ffffu64.to_le_bytes(),
    )?;
    let table_start = layout
        .table_physical
        .checked_add(PAGE_SIZE)
        .ok_or(WhpError::MemoryOverflow)?;
    let store = WhpPageTableStore::new(partition, table_start, layout.table_pages - 1)?;
    let mut tables = PageTableBuilder::new(store).map_err(|_| WhpError::InvalidRegisterState)?;
    map_pe(
        &mut tables,
        image,
        layout.image_physical,
        layout.image_virtual,
        layout.user,
    )?;
    let handoff_permissions = if layout.user {
        PagePermissions::USER_READ
    } else {
        PagePermissions::KERNEL_READ
    };
    tables
        .map(
            Mapping::new(
                VirtAddr::new(layout.handoff_virtual).map_err(|_| WhpError::InvalidMapping)?,
                PhysAddr::new(layout.handoff_physical).map_err(|_| WhpError::InvalidMapping)?,
                handoff_bytes / PAGE_SIZE,
                handoff_permissions,
            )
            .map_err(|_| WhpError::InvalidMapping)?,
        )
        .map_err(|_| WhpError::PageTable)?;
    tables
        .map_page(
            VirtAddr::new(layout.table_physical).map_err(|_| WhpError::InvalidMapping)?,
            PhysAddr::new(layout.table_physical).map_err(|_| WhpError::InvalidMapping)?,
            if layout.user {
                PagePermissions::USER_READ
            } else {
                PagePermissions::KERNEL_READ
            },
        )
        .map_err(|_| WhpError::PageTable)?;
    let stack_permissions = if layout.user {
        PagePermissions::USER_READ_WRITE
    } else {
        PagePermissions::KERNEL_READ_WRITE
    };
    tables
        .map(
            Mapping::new(
                VirtAddr::new(layout.stack_virtual).map_err(|_| WhpError::InvalidMapping)?,
                PhysAddr::new(layout.stack_physical).map_err(|_| WhpError::InvalidMapping)?,
                layout.stack_pages,
                stack_permissions,
            )
            .map_err(|_| WhpError::InvalidMapping)?,
        )
        .map_err(|_| WhpError::PageTable)?;
    Ok(tables.root())
}

pub(super) struct WhpPageTableStore<'a, 'system> {
    partition: &'a mut PreparedWhpPartition<'system>,
    start: u64,
    next: u64,
    end: u64,
}

impl<'a, 'system> WhpPageTableStore<'a, 'system> {
    pub(super) fn new(
        partition: &'a mut PreparedWhpPartition<'system>,
        start: u64,
        pages: u64,
    ) -> Result<Self, WhpError> {
        let end = start
            .checked_add(
                pages
                    .checked_mul(PAGE_SIZE)
                    .ok_or(WhpError::MemoryOverflow)?,
            )
            .ok_or(WhpError::MemoryOverflow)?;
        Ok(Self {
            partition,
            start,
            next: start,
            end,
        })
    }

    fn entry_address(&self, table: PhysAddr, index: usize) -> Result<u64, PageTableBuildError> {
        if index >= 512
            || table.get() < self.start
            || table.get() >= self.next
            || !table.get().is_multiple_of(PAGE_SIZE)
        {
            return Err(PageTableBuildError::Storage);
        }
        table
            .get()
            .checked_add(index as u64 * 8)
            .ok_or(PageTableBuildError::AddressOverflow)
    }
}

impl PageTableStore for WhpPageTableStore<'_, '_> {
    fn allocate_zeroed(&mut self) -> Result<PhysAddr, PageTableBuildError> {
        let following = self
            .next
            .checked_add(PAGE_SIZE)
            .ok_or(PageTableBuildError::AddressOverflow)?;
        if following > self.end {
            return Err(PageTableBuildError::Storage);
        }
        let address = PhysAddr::new(self.next).map_err(|_| PageTableBuildError::Storage)?;
        self.partition
            .write_guest(self.next, &[0u8; PAGE_SIZE as usize])
            .map_err(|_| PageTableBuildError::Storage)?;
        self.next = following;
        Ok(address)
    }

    fn read(&self, table: PhysAddr, index: usize) -> Result<u64, PageTableBuildError> {
        let mut bytes = [0u8; 8];
        self.partition
            .read_guest(self.entry_address(table, index)?, &mut bytes)
            .map_err(|_| PageTableBuildError::Storage)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn write(
        &mut self,
        table: PhysAddr,
        index: usize,
        value: u64,
    ) -> Result<(), PageTableBuildError> {
        let address = self.entry_address(table, index)?;
        self.partition
            .write_guest(address, &value.to_le_bytes())
            .map_err(|_| PageTableBuildError::Storage)
    }
}

fn map_pe<S: PageTableStore>(
    tables: &mut PageTableBuilder<S>,
    image: &PeImage<'_>,
    physical: u64,
    virtual_base: u64,
    user: bool,
) -> Result<(), WhpError> {
    let mut validated = AddressSpace::<{ MAX_PE_SECTIONS + 1 }>::new();
    for index in 0..image.load_region_count() {
        validated
            .map(pe_mapping(image, physical, virtual_base, index, user)?)
            .map_err(|_| WhpError::InvalidMapping)?;
    }
    for index in 0..image.load_region_count() {
        tables
            .map(pe_mapping(image, physical, virtual_base, index, user)?)
            .map_err(|_| WhpError::PageTable)?;
    }
    Ok(())
}

fn pe_mapping(
    image: &PeImage<'_>,
    physical: u64,
    virtual_base: u64,
    index: usize,
    user: bool,
) -> Result<Mapping, WhpError> {
    let region = image.load_region(index).map_err(WhpError::Pe)?;
    let permissions = match (user, region.writable(), region.executable()) {
        (true, true, false) => PagePermissions::USER_READ_WRITE,
        (true, false, true) => PagePermissions::USER_READ_EXECUTE,
        (true, false, false) => PagePermissions::USER_READ,
        (false, true, false) => PagePermissions::KERNEL_READ_WRITE,
        (false, false, true) => PagePermissions::KERNEL_READ_EXECUTE,
        (false, false, false) => PagePermissions::KERNEL_READ,
        (_, true, true) => return Err(WhpError::InvalidMapping),
    };
    let offset = region.virtual_address() as u64;
    Mapping::new(
        VirtAddr::new(
            virtual_base
                .checked_add(offset)
                .ok_or(WhpError::InvalidMapping)?,
        )
        .map_err(|_| WhpError::InvalidMapping)?,
        PhysAddr::new(
            physical
                .checked_add(offset)
                .ok_or(WhpError::InvalidMapping)?,
        )
        .map_err(|_| WhpError::InvalidMapping)?,
        region.pages() as u64,
        permissions,
    )
    .map_err(|_| WhpError::InvalidMapping)
}

fn range(start: u64, bytes: u64, permissions: MapPermissions) -> Result<GuestRange, WhpError> {
    GuestRange::new(start, bytes, permissions)
}

fn page_bytes(bytes: u64) -> Result<u64, WhpError> {
    if bytes == 0 {
        return Err(WhpError::EmptyMemory);
    }
    bytes
        .checked_add(PAGE_SIZE - 1)
        .ok_or(WhpError::MemoryOverflow)
        .map(|value| value / PAGE_SIZE * PAGE_SIZE)
}

fn validate_ranges(ranges: &[(u64, u64)]) -> Result<(), WhpError> {
    for (index, &(start, bytes)) in ranges.iter().enumerate() {
        let end = start.checked_add(bytes).ok_or(WhpError::MemoryOverflow)?;
        for &(other, other_bytes) in &ranges[..index] {
            let other_end = other
                .checked_add(other_bytes)
                .ok_or(WhpError::MemoryOverflow)?;
            if start < other_end && other < end {
                return Err(WhpError::MemoryOverlap);
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
        pe[optional + 24..optional + 32].copy_from_slice(&0x20_0000u64.to_le_bytes());
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
        encoded[192..200].copy_from_slice(&0x3000u64.to_le_bytes());
        encoded[200..208].copy_from_slice(&1u64.to_le_bytes());
        encoded[208] = 1;
        encoded[216..224].copy_from_slice(&0xa0000u64.to_le_bytes());
        encoded[224..232].copy_from_slice(&1u64.to_le_bytes());
        encoded[232] = 3;
        encoded
    }

    fn signed_pe() -> [u8; SIGNED_ARTIFACT_OVERHEAD_BYTES + 1024] {
        let payload = valid_pe();
        let mut private = [0u8; LAMPORT_PRIVATE_KEY_BYTES];
        for (index, byte) in private.iter_mut().enumerate() {
            *byte = (index as u64).wrapping_mul(73).wrapping_add(29) as u8;
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
    fn launch_layout_rejects_aliases_and_weak_arenas() {
        assert_eq!(
            WhpLaunchLayout::new(
                0x1000,
                3,
                0x10000,
                0x140000000,
                0x20000,
                0x200000,
                0x30000,
                0x300000,
                2,
                false
            ),
            Err(WhpError::InvalidMapping)
        );
        assert_eq!(
            validate_ranges(&[(0x1000, 0x2000), (0x2000, 0x1000)]),
            Err(WhpError::MemoryOverlap)
        );
        assert!(
            WhpLaunchLayout::new(
                0x1000,
                8,
                0x10000,
                0x140000000,
                0x20000,
                0x200000,
                0x30000,
                0x300000,
                2,
                false
            )
            .is_ok()
        );
    }

    #[test]
    fn backend_rejects_nonexistent_virtual_processors() {
        let system = WhpSystem::open().unwrap();
        if !system.hypervisor_present().unwrap() {
            return;
        }
        let partition = system.prepare_partition().unwrap();
        let mut guest = PreparedWhpGuest {
            partition,
            entry: 0x20_0000,
            root: PhysAddr::new(0x10_0000).unwrap(),
        };
        assert_eq!(VmBackend::run(&mut guest, 1), Err(WhpError::InvalidVcpu));
        assert_eq!(
            VmBackend::inject_interrupt(&mut guest, 1, 48),
            Err(WhpError::InvalidVcpu)
        );
    }

    #[test]
    fn live_long_mode_guest_reaches_breakpoint_exit() {
        let system = WhpSystem::open().unwrap();
        if !system.hypervisor_present().unwrap() {
            return;
        }
        let mut partition = system.prepare_partition().unwrap();
        partition
            .map_zeroed(GuestRange::new(0x10_0000, 0x8000, MapPermissions::read_write()).unwrap())
            .unwrap();
        partition
            .map_initialized(
                GuestRange::new(0x20_0000, 0x1000, MapPermissions::read_execute()).unwrap(),
                &[0xcc],
            )
            .unwrap();
        partition
            .map_zeroed(GuestRange::new(0x30_0000, 0x1000, MapPermissions::read_write()).unwrap())
            .unwrap();
        let root = {
            partition
                .write_guest(0x10_0008, &0x00af_9b00_0000_ffffu64.to_le_bytes())
                .unwrap();
            let store = WhpPageTableStore::new(&mut partition, 0x10_1000, 7).unwrap();
            let mut tables = PageTableBuilder::new(store).unwrap();
            for (address, permissions) in [
                (0x10_0000, PagePermissions::KERNEL_READ),
                (0x20_0000, PagePermissions::KERNEL_READ_EXECUTE),
                (0x30_0000, PagePermissions::KERNEL_READ_WRITE),
            ] {
                tables
                    .map_page(
                        VirtAddr::new(address).unwrap(),
                        PhysAddr::new(address).unwrap(),
                        permissions,
                    )
                    .unwrap();
            }
            tables.root()
        };
        partition
            .configure_long_mode(0x20_0000, 0x30_0ff8, root.get(), 0x10_0000, 0x30_0000, 8)
            .unwrap();
        assert_eq!(
            partition.run(),
            Ok(VmExit::Unknown {
                reason: (0x1002u64 << 32) | 3,
            })
        );
    }

    #[test]
    fn live_verified_pe_reaches_signed_entry_point() {
        let system = WhpSystem::open().unwrap();
        if !system.hypervisor_present().unwrap() {
            return;
        }
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
        let layout = WhpLaunchLayout::new(
            0x10_0000, 16, 0x20_0000, 0x20_0000, 0x30_0000, 0x30_0000, 0x40_0000, 0x40_0000, 2,
            true,
        )
        .unwrap();
        let mut guest = system
            .prepare_guest(&executable, &valid_handoff(), layout)
            .unwrap();
        assert_eq!(guest.entry(), 0x20_1000);
        assert_eq!(
            VmBackend::write_guest(&mut guest, 0x20_1000, &[0xf4]),
            Err(WhpError::ReadOnlyMemory)
        );
        assert_eq!(
            guest.run(),
            Ok(VmExit::Unknown {
                reason: (0x1002u64 << 32) | 3,
            })
        );
    }
}
