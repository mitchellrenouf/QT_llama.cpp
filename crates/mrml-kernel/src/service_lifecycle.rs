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
    InvalidRestartPolicy,
    RestartLimit,
    RestartBackoff { eligible_at: u64 },
    TimeExhausted,
    Unauthorized(CapabilityError),
    Runtime(TaskRuntimeError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RestartPolicy {
    maximum_restarts: u32,
    base_backoff_ticks: u64,
    maximum_backoff_ticks: u64,
}

impl RestartPolicy {
    pub const ONCE_IMMEDIATE: Self = Self {
        maximum_restarts: 1,
        base_backoff_ticks: 0,
        maximum_backoff_ticks: 0,
    };

    pub const fn new(
        maximum_restarts: u32,
        base_backoff_ticks: u64,
        maximum_backoff_ticks: u64,
    ) -> Result<Self, ServiceError> {
        if maximum_restarts == 0 || maximum_backoff_ticks < base_backoff_ticks {
            return Err(ServiceError::InvalidRestartPolicy);
        }
        Ok(Self {
            maximum_restarts,
            base_backoff_ticks,
            maximum_backoff_ticks,
        })
    }

    fn delay(self, completed_restarts: u32) -> u64 {
        let shift = completed_restarts.min(63);
        self.base_backoff_ticks
            .saturating_mul(1u64 << shift)
            .min(self.maximum_backoff_ticks)
    }
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
    restart_policy: RestartPolicy,
    completed_restarts: u32,
    next_restart_tick: u64,
}

impl ServiceSlot {
    const EMPTY: Self = Self {
        generation: 1,
        object: ObjectId(0),
        task: None,
        state: ServiceState::Exited,
        occupied: false,
        restart_policy: RestartPolicy::ONCE_IMMEDIATE,
        completed_restarts: 0,
        next_restart_tick: 0,
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
        self.register_with_policy(object, task, RestartPolicy::ONCE_IMMEDIATE)
    }

    pub fn register_with_policy(
        &mut self,
        object: ObjectId,
        task: TaskId,
        restart_policy: RestartPolicy,
    ) -> Result<ServiceId, ServiceError> {
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
        entry.restart_policy = restart_policy;
        entry.completed_restarts = 0;
        entry.next_restart_tick = 0;
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
        self.exit_current_at(runtime, 0)
    }

    pub fn exit_current_at<const TASKS: usize, const CAPS: usize>(
        &mut self,
        runtime: &mut TaskRuntime<TASKS, CAPS>,
        now: u64,
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
        self.arm_restart(slot, now)?;
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
        self.fault_current_at(runtime, disposition, 0)
    }

    pub fn fault_current_at<const TASKS: usize, const CAPS: usize>(
        &mut self,
        runtime: &mut TaskRuntime<TASKS, CAPS>,
        disposition: TrapDisposition,
        now: u64,
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
        self.arm_restart(slot, now)?;
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
        self.restart_at(
            service,
            management,
            control,
            runtime,
            priority,
            fresh_context,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn restart_at<const TASKS: usize, const CAPS: usize, const MANAGEMENT_CAPS: usize>(
        &mut self,
        service: ServiceId,
        management: &CapabilitySpace<MANAGEMENT_CAPS>,
        control: Capability,
        runtime: &mut TaskRuntime<TASKS, CAPS>,
        priority: Priority,
        fresh_context: UserContext,
        now: u64,
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
        if entry.completed_restarts >= entry.restart_policy.maximum_restarts {
            return Err(ServiceError::RestartLimit);
        }
        if now < entry.next_restart_tick {
            return Err(ServiceError::RestartBackoff {
                eligible_at: entry.next_restart_tick,
            });
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
        self.slots[slot].completed_restarts += 1;
        Ok(ServiceId {
            slot: slot as u32,
            generation,
        })
    }

    fn arm_restart(&mut self, slot: usize, now: u64) -> Result<(), ServiceError> {
        if self.slots[slot].completed_restarts >= self.slots[slot].restart_policy.maximum_restarts {
            self.slots[slot].next_restart_tick = u64::MAX;
            return Ok(());
        }
        let delay = self.slots[slot]
            .restart_policy
            .delay(self.slots[slot].completed_restarts);
        self.slots[slot].next_restart_tick =
            now.checked_add(delay).ok_or(ServiceError::TimeExhausted)?;
        Ok(())
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

    #[test]
    fn restart_policy_enforces_authority_backoff_and_budget() {
        assert_eq!(
            RestartPolicy::new(0, 1, 1),
            Err(ServiceError::InvalidRestartPolicy)
        );
        assert_eq!(
            RestartPolicy::new(1, 2, 1),
            Err(ServiceError::InvalidRestartPolicy)
        );
        assert_eq!(
            RestartPolicy::new(2, u64::MAX - 1, u64::MAX)
                .unwrap()
                .delay(1),
            u64::MAX
        );
        let object = ObjectId(0x61);
        let policy = RestartPolicy::new(2, 10, 15).unwrap();
        let mut runtime = TaskRuntime::<1, 0>::new(1_000, 1).unwrap();
        let task = runtime
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        runtime.start();
        let mut supervisor = ServiceSupervisor::<1>::new();
        let first = supervisor
            .register_with_policy(object, task, policy)
            .unwrap();
        supervisor.exit_current_at(&mut runtime, 100).unwrap();

        let mut management = CapabilitySpace::<2>::new();
        let wrong = management.insert(ObjectId(0x62), Rights::CONTROL).unwrap();
        let control = management.insert(object, Rights::CONTROL).unwrap();
        assert_eq!(
            supervisor.restart_at(
                first,
                &management,
                wrong,
                &mut runtime,
                Priority::NORMAL,
                context(0x30_0000, 0x50_0000),
                101,
            ),
            Err(ServiceError::Unauthorized(
                CapabilityError::PermissionDenied
            ))
        );
        assert_eq!(
            supervisor.restart_at(
                first,
                &management,
                control,
                &mut runtime,
                Priority::NORMAL,
                context(0x30_0000, 0x50_0000),
                109,
            ),
            Err(ServiceError::RestartBackoff { eligible_at: 110 })
        );
        let second = supervisor
            .restart_at(
                first,
                &management,
                control,
                &mut runtime,
                Priority::NORMAL,
                context(0x30_0000, 0x50_0000),
                110,
            )
            .unwrap();
        assert!(matches!(runtime.start(), ScheduleOutcome::Switch { .. }));
        supervisor
            .fault_current_at(
                &mut runtime,
                TrapDisposition::TerminateUser {
                    vector: 6,
                    address: None,
                },
                200,
            )
            .unwrap();
        assert_eq!(
            supervisor.restart_at(
                second,
                &management,
                control,
                &mut runtime,
                Priority::NORMAL,
                context(0x40_0000, 0x60_0000),
                214,
            ),
            Err(ServiceError::RestartBackoff { eligible_at: 215 })
        );
        let third = supervisor
            .restart_at(
                second,
                &management,
                control,
                &mut runtime,
                Priority::NORMAL,
                context(0x40_0000, 0x60_0000),
                215,
            )
            .unwrap();
        assert!(matches!(runtime.start(), ScheduleOutcome::Switch { .. }));
        supervisor.exit_current_at(&mut runtime, 300).unwrap();
        assert_eq!(
            supervisor.restart_at(
                third,
                &management,
                control,
                &mut runtime,
                Priority::NORMAL,
                context(0x50_0000, 0x70_0000),
                u64::MAX,
            ),
            Err(ServiceError::RestartLimit)
        );
    }

    #[test]
    fn retirement_time_overflow_fails_after_revocation() {
        let mut runtime = TaskRuntime::<1, 0>::new(1_000, 1).unwrap();
        let task = runtime
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        runtime.start();
        let mut supervisor = ServiceSupervisor::<1>::new();
        supervisor
            .register_with_policy(ObjectId(7), task, RestartPolicy::new(1, 10, 10).unwrap())
            .unwrap();
        assert_eq!(
            supervisor.exit_current_at(&mut runtime, u64::MAX - 5),
            Err(ServiceError::TimeExhausted)
        );
        assert_eq!(runtime.context(task), Err(TaskRuntimeError::MissingTask));
        assert_eq!(runtime.current(), None);
    }
}
