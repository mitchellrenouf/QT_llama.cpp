use core::array;

use crate::arch::x86_64::{TrapDisposition, UserContext};
use crate::{
    Capability, CapabilityError, CapabilitySpace, FaultRetirement, ObjectId, Priority,
    ScheduleOutcome, TaskId, TaskRuntime, TaskRuntimeError, TaskTermination,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServiceId {
    slot: u32,
    generation: u32,
}

impl ServiceId {
    pub const fn token(self) -> u64 {
        ((self.generation as u64) << 32) | self.slot as u64
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceState {
    Running,
    Exited,
    Faulted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceError {
    Full,
    InvalidService,
    InvalidObject,
    TaskMismatch,
    StillRunning,
    GenerationExhausted,
    Unauthorized(CapabilityError),
    Runtime(TaskRuntimeError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServiceRetirement {
    pub service: ServiceId,
    pub task: TaskId,
    pub next: ScheduleOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServiceFault {
    pub service: ServiceId,
    pub retirement: FaultRetirement,
}

#[derive(Clone, Copy)]
struct ServiceSlot {
    generation: u32,
    object: ObjectId,
    task: Option<TaskId>,
    state: ServiceState,
    occupied: bool,
}

impl ServiceSlot {
    const EMPTY: Self = Self {
        generation: 1,
        object: ObjectId(0),
        task: None,
        state: ServiceState::Exited,
        occupied: false,
    };
}

/// Fixed-capacity owner of service-to-task identity. A stopped or faulted
/// instance can restart only through CONTROL authority for its exact object and
/// a freshly supplied context. Restart advances the service generation, making
/// every handle to the retired instance permanently stale.
pub struct ServiceSupervisor<const SERVICES: usize> {
    slots: [ServiceSlot; SERVICES],
}

impl<const SERVICES: usize> ServiceSupervisor<SERVICES> {
    pub fn new() -> Self {
        Self {
            slots: array::from_fn(|_| ServiceSlot::EMPTY),
        }
    }

    pub fn register(&mut self, object: ObjectId, task: TaskId) -> Result<ServiceId, ServiceError> {
        if object.0 == 0 {
            return Err(ServiceError::InvalidObject);
        }
        let (slot, entry) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, entry)| !entry.occupied)
            .ok_or(ServiceError::Full)?;
        entry.object = object;
        entry.task = Some(task);
        entry.state = ServiceState::Running;
        entry.occupied = true;
        Ok(ServiceId {
            slot: slot as u32,
            generation: entry.generation,
        })
    }

    pub fn state(&self, service: ServiceId) -> Result<ServiceState, ServiceError> {
        Ok(self.slot(service)?.state)
    }

    pub fn task(&self, service: ServiceId) -> Result<Option<TaskId>, ServiceError> {
        Ok(self.slot(service)?.task)
    }

    pub fn exit_current<const TASKS: usize, const CAPS: usize>(
        &mut self,
        runtime: &mut TaskRuntime<TASKS, CAPS>,
    ) -> Result<ServiceRetirement, ServiceError> {
        let current = runtime
            .current()
            .ok_or(ServiceError::Runtime(TaskRuntimeError::NoCurrentTask))?;
        let slot = self.running_task_slot(current)?;
        let service = self.id(slot);
        let TaskTermination { task, next } =
            runtime.terminate_current().map_err(ServiceError::Runtime)?;
        if task != current {
            return Err(ServiceError::TaskMismatch);
        }
        self.slots[slot].task = None;
        self.slots[slot].state = ServiceState::Exited;
        Ok(ServiceRetirement {
            service,
            task,
            next,
        })
    }

    pub fn fault_current<const TASKS: usize, const CAPS: usize>(
        &mut self,
        runtime: &mut TaskRuntime<TASKS, CAPS>,
        disposition: TrapDisposition,
    ) -> Result<ServiceFault, ServiceError> {
        let current = runtime
            .current()
            .ok_or(ServiceError::Runtime(TaskRuntimeError::NoCurrentTask))?;
        let slot = self.running_task_slot(current)?;
        let service = self.id(slot);
        let retirement = runtime
            .terminate_current_fault(disposition)
            .map_err(ServiceError::Runtime)?;
        if retirement.task != current {
            return Err(ServiceError::TaskMismatch);
        }
        self.slots[slot].task = None;
        self.slots[slot].state = ServiceState::Faulted;
        Ok(ServiceFault {
            service,
            retirement,
        })
    }

    pub fn restart<const TASKS: usize, const CAPS: usize, const MANAGEMENT_CAPS: usize>(
        &mut self,
        service: ServiceId,
        management: &CapabilitySpace<MANAGEMENT_CAPS>,
        control: Capability,
        runtime: &mut TaskRuntime<TASKS, CAPS>,
        priority: Priority,
        fresh_context: UserContext,
    ) -> Result<ServiceId, ServiceError> {
        let slot = service.slot as usize;
        let entry = self.slot(service)?;
        if entry.state == ServiceState::Running || entry.task.is_some() {
            return Err(ServiceError::StillRunning);
        }
        let authorized = management
            .authorize(control, crate::Rights::CONTROL)
            .map_err(ServiceError::Unauthorized)?;
        if authorized != entry.object {
            return Err(ServiceError::Unauthorized(
                CapabilityError::PermissionDenied,
            ));
        }
        let generation = entry
            .generation
            .checked_add(1)
            .filter(|generation| *generation != 0)
            .ok_or(ServiceError::GenerationExhausted)?;
        let task = runtime
            .create(priority, fresh_context)
            .map_err(ServiceError::Runtime)?;
        self.slots[slot].generation = generation;
        self.slots[slot].task = Some(task);
        self.slots[slot].state = ServiceState::Running;
        Ok(ServiceId {
            slot: slot as u32,
            generation,
        })
    }

    fn slot(&self, service: ServiceId) -> Result<&ServiceSlot, ServiceError> {
        self.slots
            .get(service.slot as usize)
            .filter(|entry| entry.occupied && entry.generation == service.generation)
            .ok_or(ServiceError::InvalidService)
    }

    fn running_task_slot(&self, task: TaskId) -> Result<usize, ServiceError> {
        self.slots
            .iter()
            .position(|entry| {
                entry.occupied && entry.state == ServiceState::Running && entry.task == Some(task)
            })
            .ok_or(ServiceError::TaskMismatch)
    }

    fn id(&self, slot: usize) -> ServiceId {
        ServiceId {
            slot: slot as u32,
            generation: self.slots[slot].generation,
        }
    }
}

impl<const SERVICES: usize> Default for ServiceSupervisor<SERVICES> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PhysAddr, Rights};

    fn context(root: u64, entry: u64) -> UserContext {
        UserContext::new(PhysAddr::new(root).unwrap(), entry, 0x7000_0000).unwrap()
    }

    #[test]
    fn clean_exit_revokes_task_and_restart_requires_exact_control() {
        let object = ObjectId(0x51);
        let mut runtime = TaskRuntime::<2, 1>::new(1_000, 1).unwrap();
        let task = runtime
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        runtime.start();
        let mut supervisor = ServiceSupervisor::<1>::new();
        let old = supervisor.register(object, task).unwrap();
        let retired = supervisor.exit_current(&mut runtime).unwrap();
        assert_eq!(retired.service, old);
        assert_eq!(supervisor.state(old), Ok(ServiceState::Exited));
        assert_eq!(runtime.context(task), Err(TaskRuntimeError::MissingTask));

        let mut management = CapabilitySpace::<2>::new();
        let wrong = management.insert(ObjectId(0x52), Rights::CONTROL).unwrap();
        assert_eq!(
            supervisor.restart(
                old,
                &management,
                wrong,
                &mut runtime,
                Priority::NORMAL,
                context(0x30_0000, 0x50_0000),
            ),
            Err(ServiceError::Unauthorized(
                CapabilityError::PermissionDenied
            ))
        );
        let control = management.insert(object, Rights::CONTROL).unwrap();
        let replacement = supervisor
            .restart(
                old,
                &management,
                control,
                &mut runtime,
                Priority::RESPONSIVE,
                context(0x30_0000, 0x50_0000),
            )
            .unwrap();
        assert_ne!(replacement, old);
        assert_eq!(supervisor.state(old), Err(ServiceError::InvalidService));
        assert_eq!(supervisor.state(replacement), Ok(ServiceState::Running));
        assert!(supervisor.task(replacement).unwrap().is_some());
    }

    #[test]
    fn faulted_service_needs_fresh_generation_to_restart() {
        let mut runtime = TaskRuntime::<1, 0>::new(1_000, 1).unwrap();
        let task = runtime
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        runtime.start();
        let mut supervisor = ServiceSupervisor::<1>::new();
        let service = supervisor.register(ObjectId(7), task).unwrap();
        let fault = supervisor
            .fault_current(
                &mut runtime,
                TrapDisposition::TerminateUser {
                    vector: 6,
                    address: None,
                },
            )
            .unwrap();
        assert_eq!(fault.service, service);
        assert_eq!(supervisor.state(service), Ok(ServiceState::Faulted));
    }
}
