use mrml_kernel::arch::x86_64::{
    AddressSpace, Mapping, PagePermissions, PageTableBuildError, PageTableBuilder, PageTableStore,
    PerCpuPrivilegeStacks, PrivilegeStackLayout, VirtAddr,
};
use mrml_kernel::{
    ArtifactKind, BootHandoff, GpuSharedQueueLayout, GpuVmmMemory, MAX_PE_SECTIONS, PAGE_SIZE,
    PeImage, PhysAddr, VerifiedExecutable, VmBackend, VmExit,
};

use crate::{GuestRange, MapPermissions, WhpError};

use super::{PreparedWhpPartition, WhpSystem};

const XAPIC_BASE: u64 = 0xfee0_0000;

#[derive(Clone, Copy)]
struct KernelDevices {
    framebuffer: bool,
    local_apic: bool,
    gpu_queue: Option<GpuSharedQueueLayout>,
    intercept_breakpoint: bool,
}

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
        PerCpuPrivilegeStacks::<1>::new(stack_physical, stack_virtual, stack_pages)
            .and_then(|set| set.cpu(0))
            .map_err(|_| WhpError::InvalidMapping)?;
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
    service_entry: [Option<u64>; 2],
    service_root: [Option<PhysAddr>; 2],
    service_instance: [Option<ServiceInstance>; 2],
}

#[derive(Clone, Copy)]
struct ServiceInstance {
    digest: [u8; 64],
    image_physical: u64,
    image_virtual: u64,
    image_bytes: u64,
    stack_physical: u64,
    stack_bytes: u64,
}

impl PreparedWhpGuest<'_> {
    pub const fn entry(&self) -> u64 {
        self.entry
    }
    pub const fn page_table_root(&self) -> PhysAddr {
        self.root
    }
    pub const fn service_entry(&self) -> Option<u64> {
        self.service_entry[0]
    }
    pub const fn service_page_table_root(&self) -> Option<PhysAddr> {
        self.service_root[0]
    }
    pub const fn service_entry_at(&self, slot: usize) -> Option<u64> {
        if slot < 2 {
            self.service_entry[slot]
        } else {
            None
        }
    }
    pub const fn service_page_table_root_at(&self, slot: usize) -> Option<PhysAddr> {
        if slot < 2 {
            self.service_root[slot]
        } else {
            None
        }
    }
    /// Recreates one stopped service's writable state from its exact verified
    /// artifact. The caller must have observed kernel retirement while the
    /// virtual processor is stopped. Publication is cleared before mutation.
    pub fn reprovision_isolated_service_at(
        &mut self,
        slot: usize,
        service: &VerifiedExecutable<'_>,
    ) -> Result<(u64, PhysAddr), WhpError> {
        let instance = self
            .service_instance
            .get(slot)
            .and_then(|instance| *instance)
            .ok_or(WhpError::InvalidMapping)?;
        if service.artifact().kind() != ArtifactKind::ServiceImage
            || service.artifact().digest() != &instance.digest
            || u64::from(service.image().image_size()) != instance.image_bytes
        {
            return Err(WhpError::InvalidMapping);
        }
        let root = self.service_root[slot].ok_or(WhpError::InvalidMapping)?;
        self.service_entry[slot] = None;
        self.service_root[slot] = None;
        let image_bytes =
            usize::try_from(instance.image_bytes).map_err(|_| WhpError::MemoryOverflow)?;
        let destination = self
            .partition
            .mutable_service(instance.image_physical, image_bytes)?;
        destination.fill(0);
        let entry = service
            .image()
            .materialize_at(destination, instance.image_virtual)
            .map_err(WhpError::Pe)?;
        let stack_bytes =
            usize::try_from(instance.stack_bytes).map_err(|_| WhpError::MemoryOverflow)?;
        self.partition
            .mutable_service(instance.stack_physical, stack_bytes)?
            .fill(0);
        self.service_entry[slot] = Some(entry);
        self.service_root[slot] = Some(root);
        Ok((entry, root))
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

    pub fn page_walk(&self, virtual_address: u64) -> Result<WhpPageWalk, WhpError> {
        if ((virtual_address << 16) as i64 >> 16) as u64 != virtual_address {
            return Err(WhpError::InvalidMapping);
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
                .ok_or(WhpError::MemoryOverflow)?;
            let mut encoded = [0u8; 8];
            self.partition.read_guest(address, &mut encoded)?;
            let entry = u64::from_le_bytes(encoded);
            entries[level] = entry;
            levels += 1;
            if entry & 1 == 0 || (level < 3 && entry & (1 << 7) != 0) {
                break;
            }
            table = entry & 0x000f_ffff_ffff_f000;
        }
        Ok(WhpPageWalk { entries, levels })
    }

    /// Attaches the common mediated-GPU queue layout using separate WHP GPA
    /// ranges. Command memory is guest-writable and completion memory is
    /// guest-read-only. Consuming `self` prevents use of a partially attached
    /// partition if the second platform mapping fails.
    pub fn attach_gpu_queue_memory(
        mut self,
        layout: GpuSharedQueueLayout,
    ) -> Result<Self, WhpError> {
        let bytes = layout
            .pages_per_ring()
            .checked_mul(PAGE_SIZE)
            .ok_or(WhpError::MemoryOverflow)?;
        self.partition.map_zeroed(range(
            layout.command_base(),
            bytes,
            MapPermissions::read_write(),
        )?)?;
        self.partition.map_zeroed_service_readonly(range(
            layout.completion_base(),
            bytes,
            MapPermissions::read_only(),
        )?)?;
        Ok(self)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attach_isolated_service(
        self,
        kernel: &VerifiedExecutable<'_>,
        kernel_layout: WhpLaunchLayout,
        service: &VerifiedExecutable<'_>,
        service_physical: u64,
        service_virtual: u64,
        stack_physical: u64,
        stack_virtual: u64,
        stack_pages: u64,
        table_physical: u64,
        table_pages: u64,
    ) -> Result<Self, WhpError> {
        self.attach_isolated_service_at(
            0,
            kernel,
            kernel_layout,
            service,
            service_physical,
            service_virtual,
            stack_physical,
            stack_virtual,
            stack_pages,
            table_physical,
            table_pages,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attach_isolated_service_at(
        mut self,
        slot: usize,
        kernel: &VerifiedExecutable<'_>,
        kernel_layout: WhpLaunchLayout,
        service: &VerifiedExecutable<'_>,
        service_physical: u64,
        service_virtual: u64,
        stack_physical: u64,
        stack_virtual: u64,
        stack_pages: u64,
        table_physical: u64,
        table_pages: u64,
        local_apic: bool,
    ) -> Result<Self, WhpError> {
        if slot >= 2
            || self.service_root[slot].is_some()
            || kernel.artifact().kind() != ArtifactKind::Kernel
            || service.artifact().kind() != ArtifactKind::ServiceImage
            || stack_pages == 0
            || table_pages < 4
            || service_virtual == 0
            || service_virtual >= 1 << 47
            || !(PAGE_SIZE..1 << 47).contains(&stack_virtual)
            || [
                service_physical,
                service_virtual,
                stack_physical,
                stack_virtual,
                table_physical,
            ]
            .iter()
            .any(|address| !address.is_multiple_of(PAGE_SIZE))
        {
            return Err(WhpError::InvalidMapping);
        }
        let image_bytes = page_bytes(service.image().image_size() as u64)?;
        let stack_bytes = stack_pages
            .checked_mul(PAGE_SIZE)
            .ok_or(WhpError::MemoryOverflow)?;
        let table_bytes = table_pages
            .checked_mul(PAGE_SIZE)
            .ok_or(WhpError::MemoryOverflow)?;
        validate_ranges(&[
            (service_physical, image_bytes),
            (stack_physical, stack_bytes),
            (table_physical, table_bytes),
        ])?;
        validate_ranges(&[
            (service_virtual, image_bytes),
            (
                stack_virtual - PAGE_SIZE,
                stack_bytes
                    .checked_add(PAGE_SIZE)
                    .ok_or(WhpError::MemoryOverflow)?,
            ),
        ])?;
        let service_mapping = self.partition.map_zeroed(range(
            service_physical,
            image_bytes,
            MapPermissions::read_write(),
        )?)?;
        self.partition.map_zeroed(range(
            stack_physical,
            stack_bytes,
            MapPermissions::read_write(),
        )?)?;
        self.partition.map_zeroed(range(
            table_physical,
            table_bytes,
            MapPermissions::read_write(),
        )?)?;
        let image = service.image();
        let destination = self
            .partition
            .mutable_guest(service_physical, image.image_size() as usize)?;
        let service_entry = image
            .materialize_at(destination, service_virtual)
            .map_err(WhpError::Pe)?;
        self.partition.seal_pe(service_mapping, image)?;

        let store = WhpPageTableStore::new(&mut self.partition, table_physical, table_pages)?;
        let mut tables =
            PageTableBuilder::new(store).map_err(|_| WhpError::InvalidRegisterState)?;
        map_pe(
            &mut tables,
            kernel.image(),
            kernel_layout.image_physical,
            kernel_layout.image_virtual,
            false,
        )?;
        map_pe(&mut tables, image, service_physical, service_virtual, true)?;
        let kernel_stack =
            PrivilegeStackLayout::new(kernel_layout.stack_virtual, kernel_layout.stack_pages)
                .map_err(|_| WhpError::InvalidMapping)?;
        let kernel_stack_physical =
            PrivilegeStackLayout::new(kernel_layout.stack_physical, kernel_layout.stack_pages)
                .map_err(|_| WhpError::InvalidMapping)?;
        for (virtual_base, physical_base, pages) in [
            (
                kernel_stack.entry_base(),
                kernel_stack_physical.entry_base(),
                kernel_stack.entry_pages(),
            ),
            (
                kernel_stack.double_fault_base(),
                kernel_stack_physical.double_fault_base(),
                kernel_stack.double_fault_pages(),
            ),
        ] {
            tables
                .map(
                    Mapping::new(
                        VirtAddr::new(virtual_base.map_err(|_| WhpError::InvalidMapping)?)
                            .map_err(|_| WhpError::InvalidMapping)?,
                        PhysAddr::new(physical_base.map_err(|_| WhpError::InvalidMapping)?)
                            .map_err(|_| WhpError::InvalidMapping)?,
                        pages,
                        PagePermissions::KERNEL_READ_WRITE,
                    )
                    .map_err(|_| WhpError::InvalidMapping)?,
                )
                .map_err(|_| WhpError::InvalidRegisterState)?;
        }
        if local_apic {
            tables
                .map_page(
                    VirtAddr::new(XAPIC_BASE).map_err(|_| WhpError::InvalidMapping)?,
                    PhysAddr::new(XAPIC_BASE).map_err(|_| WhpError::InvalidMapping)?,
                    PagePermissions::KERNEL_MMIO_READ_WRITE,
                )
                .map_err(|_| WhpError::InvalidRegisterState)?;
        }
        tables
            .map(
                Mapping::new(
                    VirtAddr::new(stack_virtual).map_err(|_| WhpError::InvalidMapping)?,
                    PhysAddr::new(stack_physical).map_err(|_| WhpError::InvalidMapping)?,
                    stack_pages,
                    PagePermissions::USER_READ_WRITE,
                )
                .map_err(|_| WhpError::InvalidMapping)?,
            )
            .map_err(|_| WhpError::InvalidRegisterState)?;
        let service_root = tables.root();
        let _ = tables.into_store();
        self.service_entry[slot] = Some(service_entry);
        self.service_root[slot] = Some(service_root);
        self.service_instance[slot] = Some(ServiceInstance {
            digest: *service.artifact().digest(),
            image_physical: service_physical,
            image_virtual: service_virtual,
            image_bytes,
            stack_physical,
            stack_bytes,
        });
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WhpPageWalk {
    entries: [u64; 4],
    levels: u8,
}

impl WhpPageWalk {
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

impl GpuVmmMemory for PreparedWhpGuest<'_> {
    fn write_gpu_service(&mut self, address: u64, input: &[u8]) -> Result<(), Self::Error> {
        self.partition.write_service(address, input)
    }
}

impl WhpSystem {
    pub fn prepare_guest<'system>(
        &'system self,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: WhpLaunchLayout,
    ) -> Result<PreparedWhpGuest<'system>, WhpError> {
        self.prepare_guest_inner(
            executable,
            handoff,
            layout,
            KernelDevices {
                framebuffer: false,
                local_apic: false,
                gpu_queue: None,
                intercept_breakpoint: true,
            },
        )
    }

    /// Prepares a standalone kernel whose architectural exceptions are
    /// delivered through its guest IDT instead of intercepted by WHP.
    pub fn prepare_isolated_service_kernel<'system>(
        &'system self,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: WhpLaunchLayout,
    ) -> Result<PreparedWhpGuest<'system>, WhpError> {
        self.prepare_guest_inner(
            executable,
            handoff,
            layout,
            KernelDevices {
                framebuffer: false,
                local_apic: false,
                gpu_queue: None,
                intercept_breakpoint: false,
            },
        )
    }

    /// Prepares a kernel with its framebuffer and supervisor-only local APIC.
    pub fn prepare_timer_kernel<'system>(
        &'system self,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: WhpLaunchLayout,
    ) -> Result<PreparedWhpGuest<'system>, WhpError> {
        self.prepare_guest_inner(
            executable,
            handoff,
            layout,
            KernelDevices {
                framebuffer: true,
                local_apic: true,
                gpu_queue: None,
                intercept_breakpoint: true,
            },
        )
    }

    /// Prepares the timer kernel with guest-IDT delivery for CPL3 proof traps.
    pub fn prepare_preemption_kernel<'system>(
        &'system self,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: WhpLaunchLayout,
    ) -> Result<PreparedWhpGuest<'system>, WhpError> {
        self.prepare_guest_inner(
            executable,
            handoff,
            layout,
            KernelDevices {
                framebuffer: true,
                local_apic: true,
                gpu_queue: None,
                intercept_breakpoint: false,
            },
        )
    }

    /// Prepares a signed guest with mediated GPU rings in its initial address
    /// space. Command memory is writable/NX and completion memory is
    /// read-only/NX to the guest.
    pub fn prepare_gpu_guest<'system>(
        &'system self,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: WhpLaunchLayout,
        queue: GpuSharedQueueLayout,
    ) -> Result<PreparedWhpGuest<'system>, WhpError> {
        self.prepare_guest_inner(
            executable,
            handoff,
            layout,
            KernelDevices {
                framebuffer: false,
                local_apic: false,
                gpu_queue: Some(queue),
                intercept_breakpoint: true,
            },
        )
    }

    /// Prepares the standalone kernel with its authenticated framebuffer and
    /// mediated GPU queues present before the vCPU can execute.
    pub fn prepare_kernel_gpu_guest<'system>(
        &'system self,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: WhpLaunchLayout,
        queue: GpuSharedQueueLayout,
    ) -> Result<PreparedWhpGuest<'system>, WhpError> {
        self.prepare_guest_inner(
            executable,
            handoff,
            layout,
            KernelDevices {
                framebuffer: true,
                local_apic: false,
                gpu_queue: Some(queue),
                intercept_breakpoint: true,
            },
        )
    }

    fn prepare_guest_inner<'system>(
        &'system self,
        executable: &VerifiedExecutable<'_>,
        handoff: &[u8],
        layout: WhpLaunchLayout,
        devices: KernelDevices,
    ) -> Result<PreparedWhpGuest<'system>, WhpError> {
        let map_framebuffer = devices.framebuffer;
        let queue = devices.gpu_queue;
        let decoded = BootHandoff::decode(handoff, |_| {}).map_err(WhpError::Handoff)?;
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
        let stack_layout = PrivilegeStackLayout::new(layout.stack_virtual, layout.stack_pages)
            .map_err(|_| WhpError::InvalidMapping)?;
        let physical_stack = PrivilegeStackLayout::new(layout.stack_physical, layout.stack_pages)
            .map_err(|_| WhpError::InvalidMapping)?;
        let framebuffer = decoded.framebuffer();
        let framebuffer_bytes = page_bytes(framebuffer.byte_length())?;
        let queue_bytes = match queue {
            Some(value) => value
                .pages_per_ring()
                .checked_mul(PAGE_SIZE)
                .ok_or(WhpError::MemoryOverflow)?,
            None => PAGE_SIZE,
        };
        let (command_base, completion_base) = queue
            .map(|value| (value.command_base(), value.completion_base()))
            .unwrap_or((0, 0));
        let mut physical_ranges = [(0, 0); 7];
        physical_ranges[..4].copy_from_slice(&[
            (layout.table_physical, table_bytes),
            (layout.image_physical, image_bytes),
            (layout.handoff_physical, handoff_bytes),
            (layout.stack_physical, stack_bytes),
        ]);
        let mut physical_count = 4;
        if map_framebuffer {
            physical_ranges[physical_count] = (framebuffer.base().get(), framebuffer_bytes);
            physical_count += 1;
        }
        if queue.is_some() {
            physical_ranges[physical_count] = (command_base, queue_bytes);
            physical_ranges[physical_count + 1] = (completion_base, queue_bytes);
            physical_count += 2;
        }
        validate_ranges(&physical_ranges[..physical_count])?;
        let mut virtual_ranges = [(0, 0); 7];
        virtual_ranges[..4].copy_from_slice(&[
            (layout.table_physical, PAGE_SIZE),
            (layout.image_virtual, image_bytes),
            (layout.handoff_virtual, handoff_bytes),
            (layout.stack_virtual, stack_bytes),
        ]);
        let mut virtual_count = 4;
        if map_framebuffer {
            virtual_ranges[virtual_count] = (framebuffer.base().get(), framebuffer_bytes);
            virtual_count += 1;
        }
        if queue.is_some() {
            virtual_ranges[virtual_count] = (command_base, queue_bytes);
            virtual_ranges[virtual_count + 1] = (completion_base, queue_bytes);
            virtual_count += 2;
        }
        validate_ranges(&virtual_ranges[..virtual_count])?;

        let mut partition =
            self.prepare_partition_with_breakpoint_exit(devices.intercept_breakpoint)?;
        partition.map_zeroed(range(
            layout.table_physical,
            table_bytes,
            MapPermissions::read_write(),
        )?)?;
        if map_framebuffer {
            partition.map_zeroed(range(
                framebuffer.base().get(),
                framebuffer_bytes,
                MapPermissions::read_write(),
            )?)?;
        }
        if let Some(queue) = queue {
            partition.map_zeroed(range(
                queue.command_base(),
                queue_bytes,
                MapPermissions::read_write(),
            )?)?;
            partition.map_zeroed_service_readonly(range(
                queue.completion_base(),
                queue_bytes,
                MapPermissions::read_only(),
            )?)?;
        }
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

        let stack = stack_layout
            .early_top()
            .map_err(|_| WhpError::MemoryOverflow)?;
        partition.write_guest(
            physical_stack
                .early_top()
                .map_err(|_| WhpError::MemoryOverflow)?,
            &0u64.to_le_bytes(),
        )?;

        let root = build_page_tables(
            &mut partition,
            image,
            layout,
            handoff_bytes,
            map_framebuffer.then_some((framebuffer.base().get(), framebuffer_bytes)),
            devices.local_apic,
            queue,
        )?;
        partition.configure_long_mode(
            entry,
            stack,
            root.get(),
            layout.table_physical,
            layout.handoff_virtual,
            handoff.len() as u64,
            stack_layout
                .entry_top()
                .map_err(|_| WhpError::MemoryOverflow)?,
            stack_layout
                .double_fault_top()
                .map_err(|_| WhpError::MemoryOverflow)?,
        )?;
        Ok(PreparedWhpGuest {
            partition,
            entry,
            root,
            service_entry: [None; 2],
            service_root: [None; 2],
            service_instance: [None; 2],
        })
    }
}

fn build_page_tables(
    partition: &mut PreparedWhpPartition<'_>,
    image: &PeImage<'_>,
    layout: WhpLaunchLayout,
    handoff_bytes: u64,
    framebuffer: Option<(u64, u64)>,
    local_apic: bool,
    queue: Option<GpuSharedQueueLayout>,
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
    if let Some((base, bytes)) = framebuffer {
        tables
            .map(
                Mapping::new(
                    VirtAddr::new(base).map_err(|_| WhpError::InvalidMapping)?,
                    PhysAddr::new(base).map_err(|_| WhpError::InvalidMapping)?,
                    bytes / PAGE_SIZE,
                    PagePermissions::KERNEL_MMIO_READ_WRITE,
                )
                .map_err(|_| WhpError::InvalidMapping)?,
            )
            .map_err(|_| WhpError::PageTable)?;
    }
    if local_apic {
        tables
            .map_page(
                VirtAddr::new(XAPIC_BASE).map_err(|_| WhpError::InvalidMapping)?,
                PhysAddr::new(XAPIC_BASE).map_err(|_| WhpError::InvalidMapping)?,
                PagePermissions::KERNEL_MMIO_READ_WRITE,
            )
            .map_err(|_| WhpError::PageTable)?;
    }
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
    let stack_permissions = PagePermissions::KERNEL_READ_WRITE;
    let stack_layout = PrivilegeStackLayout::new(layout.stack_virtual, layout.stack_pages)
        .map_err(|_| WhpError::InvalidMapping)?;
    let physical_stack = PrivilegeStackLayout::new(layout.stack_physical, layout.stack_pages)
        .map_err(|_| WhpError::InvalidMapping)?;
    for (virtual_base, physical_base, pages, permissions) in [
        (
            stack_layout.early_base(),
            physical_stack.early_base(),
            stack_layout.early_pages(),
            stack_permissions,
        ),
        (
            stack_layout
                .entry_base()
                .map_err(|_| WhpError::MemoryOverflow)?,
            physical_stack
                .entry_base()
                .map_err(|_| WhpError::MemoryOverflow)?,
            stack_layout.entry_pages(),
            PagePermissions::KERNEL_READ_WRITE,
        ),
        (
            stack_layout
                .double_fault_base()
                .map_err(|_| WhpError::MemoryOverflow)?,
            physical_stack
                .double_fault_base()
                .map_err(|_| WhpError::MemoryOverflow)?,
            stack_layout.double_fault_pages(),
            PagePermissions::KERNEL_READ_WRITE,
        ),
    ] {
        tables
            .map(
                Mapping::new(
                    VirtAddr::new(virtual_base).map_err(|_| WhpError::InvalidMapping)?,
                    PhysAddr::new(physical_base).map_err(|_| WhpError::InvalidMapping)?,
                    pages,
                    permissions,
                )
                .map_err(|_| WhpError::InvalidMapping)?,
            )
            .map_err(|_| WhpError::PageTable)?;
    }
    if let Some(queue) = queue {
        tables
            .map(
                Mapping::new(
                    VirtAddr::new(queue.command_base()).map_err(|_| WhpError::InvalidMapping)?,
                    PhysAddr::new(queue.command_base()).map_err(|_| WhpError::InvalidMapping)?,
                    queue.pages_per_ring(),
                    PagePermissions::KERNEL_SHARED_READ_WRITE,
                )
                .map_err(|_| WhpError::InvalidMapping)?,
            )
            .map_err(|_| WhpError::PageTable)?;
        tables
            .map(
                Mapping::new(
                    VirtAddr::new(queue.completion_base()).map_err(|_| WhpError::InvalidMapping)?,
                    PhysAddr::new(queue.completion_base()).map_err(|_| WhpError::InvalidMapping)?,
                    queue.pages_per_ring(),
                    PagePermissions::KERNEL_SHARED_READ,
                )
                .map_err(|_| WhpError::InvalidMapping)?,
            )
            .map_err(|_| WhpError::PageTable)?;
    }
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
                0xffff_8000_0030_0000,
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
                32,
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
            service_entry: [None; 2],
            service_root: [None; 2],
            service_instance: [None; 2],
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
            .configure_long_mode(
                0x20_0000,
                0x30_0ff8,
                root.get(),
                0x10_0000,
                0x30_0000,
                8,
                0x50_0000,
                0x60_0000,
            )
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
            0x10_0000,
            16,
            0x20_0000,
            0x20_0000,
            0x30_0000,
            0x30_0000,
            0x40_0000,
            0xffff_8000_0040_0000,
            32,
            true,
        )
        .unwrap();
        let queue_layout = GpuSharedQueueLayout::new(0x50_0000, 0x50_2000, 64).unwrap();
        let mut guest = system
            .prepare_gpu_guest(&executable, &valid_handoff(), layout, queue_layout)
            .unwrap();
        assert_eq!(guest.entry(), 0x20_1000);
        let stack_layout = PrivilegeStackLayout::new(0xffff_8000_0040_0000, 32).unwrap();
        assert_eq!(
            guest
                .page_walk(stack_layout.early_base())
                .unwrap()
                .physical_address(stack_layout.early_base()),
            Some(0x40_0000)
        );
        for guard in [
            stack_layout.entry_guard().unwrap(),
            stack_layout.double_fault_guard().unwrap(),
        ] {
            assert_eq!(
                guest.page_walk(guard).unwrap().physical_address(guard),
                None
            );
        }
        let double_fault_base = stack_layout.double_fault_base().unwrap();
        assert_eq!(
            guest
                .page_walk(double_fault_base)
                .unwrap()
                .physical_address(double_fault_base),
            Some(0x41_8000)
        );
        let command_walk = guest.page_walk(queue_layout.command_base()).unwrap();
        assert_eq!(command_walk.levels(), 4);
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
        assert_eq!(
            VmBackend::write_guest(&mut guest, 0x20_1000, &[0xf4]),
            Err(WhpError::ReadOnlyMemory)
        );
        GpuVmmMemory::write_gpu_service(&mut guest, queue_layout.completion_base(), &[0x5a])
            .unwrap();
        let mut service_byte = [0; 1];
        VmBackend::read_guest(&guest, queue_layout.completion_base(), &mut service_byte).unwrap();
        assert_eq!(service_byte, [0x5a]);
        VmBackend::write_guest(&mut guest, queue_layout.command_base(), &[1]).unwrap();
        assert_eq!(
            VmBackend::write_guest(&mut guest, queue_layout.completion_base(), &[1]),
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
