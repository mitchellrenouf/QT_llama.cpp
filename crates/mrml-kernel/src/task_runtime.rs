use core::array;

use crate::arch::x86_64::{TrapDisposition, UserContext};
use crate::{
    BalancePolicyError, BalanceTarget, Capability, CapabilitySpace, DetachedTask, Endpoint,
    IpcError, KernelScheduleError, KernelScheduler, Message, OwnershipMailbox, PeriodicBalancer,
    Priority, Rights, ScheduleOutcome, SchedulerLoad, TaskId, TaskMigration,
};

pub const TASK_INBOX_MESSAGES: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskRuntimeError {
    Full,
    MissingTask,
    NoCurrentTask,
    AddressSpaceMismatch,
    NonRecoverableFault,
    Scheduler(KernelScheduleError),
    IntegrityFailure,
    SameTaskIpc,
    Ipc(IpcError),
    InboxFull,
    InboxEmpty,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FaultRetirement {
    pub task: TaskId,
    pub context: UserContext,
    pub vector: u8,
    pub address: Option<u64>,
    pub next: ScheduleOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TaskTermination {
    pub task: TaskId,
    pub next: ScheduleOutcome,
}

struct TaskDomain<const CAPS: usize> {
    task: TaskId,
    context: UserContext,
    capabilities: CapabilitySpace<CAPS>,
    inbox: [Option<Message>; TASK_INBOX_MESSAGES],
    inbox_head: u8,
    inbox_tail: u8,
    inbox_count: u8,
}

pub struct DetachedTaskDomain<const CAPS: usize> {
    scheduling: DetachedTask,
    domain: TaskDomain<CAPS>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DomainBalanceError {
    Policy(BalancePolicyError),
    Runtime(TaskRuntimeError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DomainBalanceOutcome {
    Idle,
    Published(BalanceTarget),
    RetryPending(BalanceTarget),
}

/// Couples periodic peer selection to linear complete-domain ownership. A full
/// destination mailbox retains the detached domain inside this controller and
/// retries it before considering another tick, so cadence cannot duplicate,
/// discard, or reorder migration tickets.
pub struct PeriodicDomainBalancer<const CPUS: usize, const CAPS: usize> {
    policy: PeriodicBalancer<CPUS>,
    pending: Option<(BalanceTarget, DetachedTaskDomain<CAPS>)>,
}

impl<const CPUS: usize, const CAPS: usize> PeriodicDomainBalancer<CPUS, CAPS> {
    pub const fn new(start_tick: u64, interval_ticks: u64) -> Result<Self, DomainBalanceError> {
        let policy = match PeriodicBalancer::new(start_tick, interval_ticks) {
            Ok(policy) => policy,
            Err(error) => return Err(DomainBalanceError::Policy(error)),
        };
        Ok(Self {
            policy,
            pending: None,
        })
    }

    pub fn pending_target(&self) -> Option<BalanceTarget> {
        self.pending.as_ref().map(|pending| pending.0)
    }

    pub fn poll_and_publish<const TASKS: usize>(
        &mut self,
        tick: u64,
        local_cpu: usize,
        loads: &[SchedulerLoad; CPUS],
        runtime: &mut TaskRuntime<TASKS, CAPS>,
        mailboxes: &[&OwnershipMailbox<DetachedTaskDomain<CAPS>>; CPUS],
    ) -> Result<DomainBalanceOutcome, DomainBalanceError> {
        if let Some((target, ticket)) = self.pending.take() {
            return match mailboxes[target.cpu()].publish(ticket) {
                Ok(()) => Ok(DomainBalanceOutcome::Published(target)),
                Err((_, ticket)) => {
                    self.pending = Some((target, ticket));
                    Ok(DomainBalanceOutcome::RetryPending(target))
                }
            };
        }

        let Some(target) = self
            .policy
            .poll(tick, local_cpu, loads)
            .map_err(DomainBalanceError::Policy)?
        else {
            return Ok(DomainBalanceOutcome::Idle);
        };
        let Some(ticket) = runtime
            .detach_domain_for_rebalance(target.load())
            .map_err(DomainBalanceError::Runtime)?
        else {
            return Ok(DomainBalanceOutcome::Idle);
        };
        match mailboxes[target.cpu()].publish(ticket) {
            Ok(()) => Ok(DomainBalanceOutcome::Published(target)),
            Err((_, ticket)) => {
                self.pending = Some((target, ticket));
                Ok(DomainBalanceOutcome::RetryPending(target))
            }
        }
    }
}

/// Owns the scheduler-visible identity, saved context, and capability space as
/// one revocation domain. Fault retirement removes the domain before it asks
/// the scheduler for a replacement, so a stale task is never resumable with a
/// partially live authority set.
pub struct TaskRuntime<const TASKS: usize, const CAPS: usize> {
    scheduler: KernelScheduler<TASKS>,
    domains: [Option<TaskDomain<CAPS>>; TASKS],
}

impl<const TASKS: usize, const CAPS: usize> TaskRuntime<TASKS, CAPS> {
    pub fn new(ticks_per_second: u32, quantum_ticks: u32) -> Result<Self, TaskRuntimeError> {
        let scheduler = KernelScheduler::new(ticks_per_second, quantum_ticks)
            .map_err(TaskRuntimeError::Scheduler)?;
        Ok(Self {
            scheduler,
            domains: array::from_fn(|_| None),
        })
    }

    pub fn create(
        &mut self,
        priority: Priority,
        context: UserContext,
    ) -> Result<TaskId, TaskRuntimeError> {
        let slot = self
            .domains
            .iter()
            .position(Option::is_none)
            .ok_or(TaskRuntimeError::Full)?;
        let task = self
            .scheduler
            .create(priority)
            .map_err(TaskRuntimeError::Scheduler)?;
        self.domains[slot] = Some(TaskDomain {
            task,
            context,
            capabilities: CapabilitySpace::new(),
            inbox: array::from_fn(|_| None),
            inbox_head: 0,
            inbox_tail: 0,
            inbox_count: 0,
        });
        Ok(task)
    }

    pub fn start(&mut self) -> ScheduleOutcome {
        self.scheduler.start()
    }

    pub const fn ticks(&self) -> u64 {
        self.scheduler.ticks()
    }

    pub fn load(&self) -> SchedulerLoad {
        self.scheduler.load()
    }

    pub fn timer_tick(&mut self) -> Result<ScheduleOutcome, TaskRuntimeError> {
        self.scheduler
            .timer_tick()
            .map_err(TaskRuntimeError::Scheduler)
    }

    /// Accounts one timer tick and atomically installs the interrupted CPL3
    /// context into the domain that was current at interrupt entry.
    pub fn preempt_current(
        &mut self,
        interrupted: UserContext,
    ) -> Result<ScheduleOutcome, TaskRuntimeError> {
        let current = self
            .scheduler
            .current()
            .ok_or(TaskRuntimeError::NoCurrentTask)?;
        let domain = self
            .domain(current)
            .ok_or(TaskRuntimeError::IntegrityFailure)?;
        if domain.context.page_table() != interrupted.page_table() {
            return Err(TaskRuntimeError::AddressSpaceMismatch);
        }
        let outcome = self
            .scheduler
            .timer_tick()
            .map_err(TaskRuntimeError::Scheduler)?;
        *self.context_mut(current)? = interrupted;
        Ok(outcome)
    }

    pub fn yield_current(&mut self) -> Result<ScheduleOutcome, TaskRuntimeError> {
        self.scheduler
            .yield_current()
            .map_err(TaskRuntimeError::Scheduler)
    }

    pub fn context(&self, task: TaskId) -> Result<&UserContext, TaskRuntimeError> {
        self.domain(task)
            .map(|domain| &domain.context)
            .ok_or(TaskRuntimeError::MissingTask)
    }

    pub fn context_mut(&mut self, task: TaskId) -> Result<&mut UserContext, TaskRuntimeError> {
        self.domains
            .iter_mut()
            .flatten()
            .find(|domain| domain.task == task)
            .map(|domain| &mut domain.context)
            .ok_or(TaskRuntimeError::MissingTask)
    }

    pub const fn current(&self) -> Option<TaskId> {
        self.scheduler.current()
    }

    /// Retires a non-current scheduler identity and removes its complete user
    /// revocation domain as one linear object. On scheduler rejection the
    /// untouched domain is restored to its exact slot.
    pub fn detach_domain(
        &mut self,
        task: TaskId,
    ) -> Result<DetachedTaskDomain<CAPS>, TaskRuntimeError> {
        let slot = self
            .domains
            .iter()
            .position(|domain| domain.as_ref().is_some_and(|domain| domain.task == task))
            .ok_or(TaskRuntimeError::MissingTask)?;
        let domain = self.domains[slot]
            .take()
            .ok_or(TaskRuntimeError::IntegrityFailure)?;
        match self.scheduler.detach(task) {
            Ok(scheduling) => Ok(DetachedTaskDomain { scheduling, domain }),
            Err(error) => {
                self.domains[slot] = Some(domain);
                Err(TaskRuntimeError::Scheduler(error))
            }
        }
    }

    pub fn detach_domain_for_rebalance(
        &mut self,
        destination: SchedulerLoad,
    ) -> Result<Option<DetachedTaskDomain<CAPS>>, TaskRuntimeError> {
        let Some(candidate) = self
            .scheduler
            .rebalance_candidate(destination)
            .map_err(TaskRuntimeError::Scheduler)?
        else {
            return Ok(None);
        };
        if self.domain(candidate).is_none() {
            return Err(TaskRuntimeError::IntegrityFailure);
        }
        self.detach_domain(candidate).map(Some)
    }

    /// Admits a complete migrated user domain under a fresh local scheduler
    /// identity. Capacity failure returns context, capabilities, inbox, and
    /// scheduling ticket together so authority cannot be orphaned.
    pub fn attach_domain(
        &mut self,
        ticket: &mut Option<DetachedTaskDomain<CAPS>>,
    ) -> Result<TaskMigration, TaskRuntimeError> {
        let Some(slot) = self.domains.iter().position(Option::is_none) else {
            return Err(TaskRuntimeError::Full);
        };
        let task = ticket.take().ok_or(TaskRuntimeError::MissingTask)?;
        let DetachedTaskDomain {
            scheduling,
            mut domain,
        } = task;
        let migration = match self.scheduler.attach(scheduling) {
            Ok(migration) => migration,
            Err(error) => {
                let schedule_error = error.error();
                *ticket = Some(DetachedTaskDomain {
                    scheduling: error.into_task(),
                    domain,
                });
                return Err(TaskRuntimeError::Scheduler(schedule_error));
            }
        };
        domain.task = migration.destination();
        self.domains[slot] = Some(domain);
        Ok(migration)
    }

    pub fn capabilities_mut(
        &mut self,
        task: TaskId,
    ) -> Result<&mut CapabilitySpace<CAPS>, TaskRuntimeError> {
        self.domains
            .iter_mut()
            .flatten()
            .find(|domain| domain.task == task)
            .map(|domain| &mut domain.capabilities)
            .ok_or(TaskRuntimeError::MissingTask)
    }

    pub fn terminate_current_fault(
        &mut self,
        disposition: TrapDisposition,
    ) -> Result<FaultRetirement, TaskRuntimeError> {
        let (vector, address) = match disposition {
            TrapDisposition::TerminateUser { vector, address } => (vector, address),
            TrapDisposition::HaltKernel { .. } => {
                return Err(TaskRuntimeError::NonRecoverableFault);
            }
        };
        let current = self
            .scheduler
            .current()
            .ok_or(TaskRuntimeError::NoCurrentTask)?;
        let slot = self
            .domains
            .iter()
            .position(|domain| domain.as_ref().is_some_and(|domain| domain.task == current))
            .ok_or(TaskRuntimeError::IntegrityFailure)?;

        // Taking the complete domain first revokes the saved context and every
        // task-local capability before scheduler selection can expose `next`.
        let domain = self.domains[slot]
            .take()
            .ok_or(TaskRuntimeError::IntegrityFailure)?;
        let next = self
            .scheduler
            .terminate_current()
            .map_err(TaskRuntimeError::Scheduler)?;
        Ok(FaultRetirement {
            task: current,
            context: domain.context,
            vector,
            address,
            next,
        })
    }

    /// Removes the current task's complete revocation domain for a voluntary
    /// exit before selecting a replacement. No context or authority from the
    /// removed domain is returned to its caller.
    pub fn terminate_current(&mut self) -> Result<TaskTermination, TaskRuntimeError> {
        let current = self
            .scheduler
            .current()
            .ok_or(TaskRuntimeError::NoCurrentTask)?;
        let slot = self
            .domains
            .iter()
            .position(|domain| domain.as_ref().is_some_and(|domain| domain.task == current))
            .ok_or(TaskRuntimeError::IntegrityFailure)?;
        let domain = self.domains[slot]
            .take()
            .ok_or(TaskRuntimeError::IntegrityFailure)?;
        let next = self
            .scheduler
            .terminate_current()
            .map_err(TaskRuntimeError::Scheduler)?;
        let _ = domain;
        Ok(TaskTermination {
            task: current,
            next,
        })
    }

    /// Transfers a message and attenuated capabilities between two distinct
    /// task domains. Tentative receiver capabilities are revoked if endpoint
    /// authorization fails, so an unsuccessful send cannot grant authority.
    pub fn send_ipc(
        &mut self,
        sender: TaskId,
        receiver: TaskId,
        endpoint: &mut Endpoint,
        endpoint_capability: Capability,
        payload: &[u8],
        requested: &[(Capability, Rights)],
    ) -> Result<(u64, Message), TaskRuntimeError> {
        if sender == receiver {
            return Err(TaskRuntimeError::SameTaskIpc);
        }
        let sender_slot = self
            .domains
            .iter()
            .position(|domain| domain.as_ref().is_some_and(|domain| domain.task == sender))
            .ok_or(TaskRuntimeError::MissingTask)?;
        let receiver_slot = self
            .domains
            .iter()
            .position(|domain| {
                domain
                    .as_ref()
                    .is_some_and(|domain| domain.task == receiver)
            })
            .ok_or(TaskRuntimeError::MissingTask)?;
        let (source, destination) = two_domains_mut(&mut self.domains, sender_slot, receiver_slot)
            .ok_or(TaskRuntimeError::IntegrityFailure)?;
        let message = Message::transfer(
            payload,
            requested,
            &source.capabilities,
            &mut destination.capabilities,
        )
        .map_err(TaskRuntimeError::Ipc)?;
        match endpoint.authorize_send(&source.capabilities, endpoint_capability, &message) {
            Ok(sequence) => Ok((sequence, message)),
            Err(error) => {
                for capability in message.capabilities() {
                    if destination.capabilities.revoke(capability).is_err() {
                        return Err(TaskRuntimeError::IntegrityFailure);
                    }
                }
                Err(TaskRuntimeError::Ipc(error))
            }
        }
    }

    /// Delivers into the receiver's bounded kernel-owned inbox and makes a
    /// blocked receiver runnable. Capacity is checked before endpoint sequence
    /// or capability state changes.
    pub fn deliver_ipc(
        &mut self,
        sender: TaskId,
        receiver: TaskId,
        endpoint: &mut Endpoint,
        endpoint_capability: Capability,
        payload: &[u8],
        requested: &[(Capability, Rights)],
    ) -> Result<u64, TaskRuntimeError> {
        let receiver_slot = self
            .domains
            .iter()
            .position(|domain| {
                domain
                    .as_ref()
                    .is_some_and(|domain| domain.task == receiver)
            })
            .ok_or(TaskRuntimeError::MissingTask)?;
        if self.domains[receiver_slot]
            .as_ref()
            .is_none_or(|domain| usize::from(domain.inbox_count) == TASK_INBOX_MESSAGES)
        {
            return Err(TaskRuntimeError::InboxFull);
        }
        let (sequence, message) = self.send_ipc(
            sender,
            receiver,
            endpoint,
            endpoint_capability,
            payload,
            requested,
        )?;
        let domain = self.domains[receiver_slot]
            .as_mut()
            .ok_or(TaskRuntimeError::IntegrityFailure)?;
        let tail = usize::from(domain.inbox_tail);
        if domain.inbox[tail].is_some() {
            return Err(TaskRuntimeError::IntegrityFailure);
        }
        domain.inbox[tail] = Some(message);
        domain.inbox_tail = ((tail + 1) % TASK_INBOX_MESSAGES) as u8;
        domain.inbox_count += 1;
        self.scheduler
            .wake(receiver)
            .map_err(TaskRuntimeError::Scheduler)?;
        Ok(sequence)
    }

    pub fn receive_ipc(&mut self, task: TaskId) -> Result<Message, TaskRuntimeError> {
        let domain = self
            .domains
            .iter_mut()
            .flatten()
            .find(|domain| domain.task == task)
            .ok_or(TaskRuntimeError::MissingTask)?;
        if domain.inbox_count == 0 {
            return Err(TaskRuntimeError::InboxEmpty);
        }
        let head = usize::from(domain.inbox_head);
        let message = domain.inbox[head]
            .take()
            .ok_or(TaskRuntimeError::IntegrityFailure)?;
        domain.inbox_head = ((head + 1) % TASK_INBOX_MESSAGES) as u8;
        domain.inbox_count -= 1;
        Ok(message)
    }

    pub fn receive_or_block_current(
        &mut self,
    ) -> Result<Result<Message, ScheduleOutcome>, TaskRuntimeError> {
        let current = self
            .scheduler
            .current()
            .ok_or(TaskRuntimeError::NoCurrentTask)?;
        match self.receive_ipc(current) {
            Ok(message) => Ok(Ok(message)),
            Err(TaskRuntimeError::InboxEmpty) => self
                .scheduler
                .block_current()
                .map(Err)
                .map_err(TaskRuntimeError::Scheduler),
            Err(error) => Err(error),
        }
    }

    fn domain(&self, task: TaskId) -> Option<&TaskDomain<CAPS>> {
        self.domains
            .iter()
            .flatten()
            .find(|domain| domain.task == task)
    }
}

fn two_domains_mut<const TASKS: usize, const CAPS: usize>(
    domains: &mut [Option<TaskDomain<CAPS>>; TASKS],
    first: usize,
    second: usize,
) -> Option<(&mut TaskDomain<CAPS>, &mut TaskDomain<CAPS>)> {
    if first == second || first >= TASKS || second >= TASKS {
        return None;
    }
    if first < second {
        let (left, right) = domains.split_at_mut(second);
        Some((left[first].as_mut()?, right[0].as_mut()?))
    } else {
        let (left, right) = domains.split_at_mut(first);
        Some((right[0].as_mut()?, left[second].as_mut()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObjectId, OwnershipMailbox, PhysAddr, Rights};

    fn context(root: u64, entry: u64) -> UserContext {
        UserContext::new(PhysAddr::new(root).unwrap(), entry, 0x0000_7000_0000_0000).unwrap()
    }

    #[test]
    fn user_fault_revokes_domain_before_selecting_replacement() {
        let mut runtime = TaskRuntime::<2, 2>::new(1_000, 1).unwrap();
        let faulted = runtime
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        let replacement = runtime
            .create(Priority::NORMAL, context(0x30_0000, 0x50_0000))
            .unwrap();
        let stale_capability = runtime
            .capabilities_mut(faulted)
            .unwrap()
            .insert(ObjectId(7), Rights::READ)
            .unwrap();
        assert_eq!(
            runtime.start(),
            ScheduleOutcome::Switch {
                from: None,
                to: faulted
            }
        );

        let retired = runtime
            .terminate_current_fault(TrapDisposition::TerminateUser {
                vector: 6,
                address: None,
            })
            .unwrap();
        assert_eq!(retired.task, faulted);
        assert_eq!(retired.vector, 6);
        assert_eq!(
            retired.next,
            ScheduleOutcome::Switch {
                from: Some(faulted),
                to: replacement
            }
        );
        assert_eq!(runtime.context(faulted), Err(TaskRuntimeError::MissingTask));
        assert_eq!(
            runtime.capabilities_mut(faulted).map(|_| ()),
            Err(TaskRuntimeError::MissingTask)
        );
        let _ = stale_capability;
        assert_eq!(
            runtime.context(replacement).unwrap().instruction_pointer(),
            0x50_0000
        );
    }

    #[test]
    fn kernel_fault_cannot_retire_a_user_domain() {
        let mut runtime = TaskRuntime::<1, 1>::new(1_000, 1).unwrap();
        let task = runtime
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        runtime.start();
        assert_eq!(
            runtime.terminate_current_fault(TrapDisposition::HaltKernel { vector: 13 }),
            Err(TaskRuntimeError::NonRecoverableFault)
        );
        assert!(runtime.context(task).is_ok());
    }

    #[test]
    fn voluntary_exit_revokes_domain_before_switching() {
        let mut runtime = TaskRuntime::<2, 1>::new(1_000, 1).unwrap();
        let exiting = runtime
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        let replacement = runtime
            .create(Priority::NORMAL, context(0x30_0000, 0x50_0000))
            .unwrap();
        let capability = runtime
            .capabilities_mut(exiting)
            .unwrap()
            .insert(ObjectId(17), Rights::READ)
            .unwrap();
        assert!(matches!(
            runtime.start(),
            ScheduleOutcome::Switch { to, .. } if to == exiting
        ));
        let terminated = runtime.terminate_current().unwrap();
        assert_eq!(terminated.task, exiting);
        assert_eq!(
            terminated.next,
            ScheduleOutcome::Switch {
                from: Some(exiting),
                to: replacement,
            }
        );
        assert_eq!(runtime.context(exiting), Err(TaskRuntimeError::MissingTask));
        assert_eq!(
            runtime.capabilities_mut(exiting).map(|_| ()),
            Err(TaskRuntimeError::MissingTask)
        );
        let _ = capability;
    }

    #[test]
    fn timer_preemption_saves_only_the_current_address_space() {
        let mut runtime = TaskRuntime::<2, 1>::new(1_000, 1).unwrap();
        let first = runtime
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        let second = runtime
            .create(Priority::NORMAL, context(0x30_0000, 0x50_0000))
            .unwrap();
        runtime.start();
        assert_eq!(
            runtime.preempt_current(context(0x30_0000, 0x60_0000)),
            Err(TaskRuntimeError::AddressSpaceMismatch)
        );
        assert_eq!(runtime.ticks(), 0);
        let saved = context(0x20_0000, 0x60_0000);
        assert_eq!(
            runtime.preempt_current(saved),
            Ok(ScheduleOutcome::Switch {
                from: Some(first),
                to: second,
            })
        );
        assert_eq!(
            runtime.context(first).unwrap().instruction_pointer(),
            0x60_0000
        );
    }

    #[test]
    fn ipc_transfer_is_attenuated_and_authorization_failure_rolls_back() {
        let mut runtime = TaskRuntime::<2, 3>::new(1_000, 1).unwrap();
        let sender = runtime
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        let receiver = runtime
            .create(Priority::NORMAL, context(0x30_0000, 0x50_0000))
            .unwrap();
        let endpoint_object = ObjectId(91);
        let endpoint_capability = runtime
            .capabilities_mut(sender)
            .unwrap()
            .insert(endpoint_object, Rights::SIGNAL)
            .unwrap();
        let delegated = runtime
            .capabilities_mut(sender)
            .unwrap()
            .insert(ObjectId(7), Rights::READ.union(Rights::DELEGATE))
            .unwrap();
        let mut endpoint = Endpoint::new(endpoint_object);
        let (sequence, message) = runtime
            .send_ipc(
                sender,
                receiver,
                &mut endpoint,
                endpoint_capability,
                b"read",
                &[(delegated, Rights::READ)],
            )
            .unwrap();
        assert_eq!(sequence, 1);
        assert_eq!(message.payload(), b"read");
        let received = message.capabilities().next().unwrap();
        assert_eq!(
            runtime
                .capabilities_mut(receiver)
                .unwrap()
                .authorize(received, Rights::READ),
            Ok(ObjectId(7))
        );

        let wrong_endpoint = runtime
            .capabilities_mut(sender)
            .unwrap()
            .insert(ObjectId(92), Rights::SIGNAL)
            .unwrap();
        assert_eq!(
            runtime
                .send_ipc(
                    sender,
                    receiver,
                    &mut endpoint,
                    wrong_endpoint,
                    b"denied",
                    &[(delegated, Rights::READ)],
                )
                .err(),
            Some(TaskRuntimeError::Ipc(IpcError::Unauthorized))
        );
        // One original received capability plus two free slots proves the
        // tentative denied transfer did not consume receiver authority space.
        let receiver_space = runtime.capabilities_mut(receiver).unwrap();
        assert!(receiver_space.insert(ObjectId(8), Rights::READ).is_ok());
        assert!(receiver_space.insert(ObjectId(9), Rights::READ).is_ok());
    }

    #[test]
    fn empty_receive_blocks_and_delivery_wakes_without_overwriting() {
        let mut runtime = TaskRuntime::<2, 2>::new(1_000, 1).unwrap();
        let receiver = runtime
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        let sender = runtime
            .create(Priority::NORMAL, context(0x30_0000, 0x50_0000))
            .unwrap();
        let object = ObjectId(44);
        let endpoint_capability = runtime
            .capabilities_mut(sender)
            .unwrap()
            .insert(object, Rights::SIGNAL)
            .unwrap();
        let mut endpoint = Endpoint::new(object);
        assert_eq!(
            runtime.start(),
            ScheduleOutcome::Switch {
                from: None,
                to: receiver
            }
        );
        assert!(matches!(
            runtime.receive_or_block_current(),
            Ok(Err(ScheduleOutcome::Switch { to, .. })) if to == sender
        ));
        assert_eq!(
            runtime
                .deliver_ipc(
                    sender,
                    receiver,
                    &mut endpoint,
                    endpoint_capability,
                    b"one",
                    &[],
                )
                .unwrap(),
            1
        );
        runtime
            .deliver_ipc(
                sender,
                receiver,
                &mut endpoint,
                endpoint_capability,
                b"two",
                &[],
            )
            .unwrap();
        assert_eq!(
            runtime
                .deliver_ipc(
                    sender,
                    receiver,
                    &mut endpoint,
                    endpoint_capability,
                    b"three",
                    &[],
                )
                .err(),
            Some(TaskRuntimeError::InboxFull)
        );
        assert_eq!(runtime.receive_ipc(receiver).unwrap().payload(), b"one");
        assert_eq!(runtime.receive_ipc(receiver).unwrap().payload(), b"two");
        assert_eq!(
            runtime.receive_ipc(receiver).err(),
            Some(TaskRuntimeError::InboxEmpty)
        );
    }

    #[test]
    fn migration_moves_context_authority_and_inbox_as_one_domain() {
        let mut source = TaskRuntime::<2, 2>::new(1_000, 1).unwrap();
        let mut destination = TaskRuntime::<2, 2>::new(1_000, 1).unwrap();
        let sender = source
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        let migrated = source
            .create(Priority::RESPONSIVE, context(0x30_0000, 0x50_0000))
            .unwrap();
        let retained_capability = source
            .capabilities_mut(migrated)
            .unwrap()
            .insert(ObjectId(77), Rights::READ)
            .unwrap();
        let endpoint_object = ObjectId(88);
        let endpoint_capability = source
            .capabilities_mut(sender)
            .unwrap()
            .insert(endpoint_object, Rights::SIGNAL)
            .unwrap();
        let mut endpoint = Endpoint::new(endpoint_object);
        source
            .deliver_ipc(
                sender,
                migrated,
                &mut endpoint,
                endpoint_capability,
                b"migrate",
                &[],
            )
            .unwrap();
        assert!(matches!(source.start(), ScheduleOutcome::Switch { to, .. } if to == sender));

        let mut ticket = Some(source.detach_domain(migrated).unwrap());
        assert_eq!(source.context(migrated), Err(TaskRuntimeError::MissingTask));
        let migration = destination.attach_domain(&mut ticket).unwrap();
        assert!(ticket.is_none());
        let admitted = migration.destination();
        assert_eq!(
            destination.context(admitted).unwrap().instruction_pointer(),
            0x50_0000
        );
        assert_eq!(
            destination
                .capabilities_mut(admitted)
                .unwrap()
                .authorize(retained_capability, Rights::READ),
            Ok(ObjectId(77))
        );
        assert_eq!(
            destination.receive_ipc(admitted).unwrap().payload(),
            b"migrate"
        );
    }

    #[test]
    fn full_runtime_returns_the_complete_domain_for_retry() {
        let mut source = TaskRuntime::<1, 1>::new(1_000, 1).unwrap();
        let mut full = TaskRuntime::<1, 1>::new(1_000, 1).unwrap();
        let mut fallback = TaskRuntime::<1, 1>::new(1_000, 1).unwrap();
        let task = source
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        full.create(Priority::NORMAL, context(0x30_0000, 0x50_0000))
            .unwrap();

        let mut ticket = Some(source.detach_domain(task).unwrap());
        assert_eq!(full.attach_domain(&mut ticket), Err(TaskRuntimeError::Full));
        assert!(ticket.is_some());
        let migration = fallback.attach_domain(&mut ticket).unwrap();
        assert!(ticket.is_none());
        assert_eq!(
            fallback
                .context(migration.destination())
                .unwrap()
                .page_table(),
            PhysAddr::new(0x20_0000).unwrap()
        );
    }

    #[test]
    fn running_domain_cannot_detach_and_remains_fully_live() {
        let mut runtime = TaskRuntime::<1, 1>::new(1_000, 1).unwrap();
        let task = runtime
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        let capability = runtime
            .capabilities_mut(task)
            .unwrap()
            .insert(ObjectId(7), Rights::READ)
            .unwrap();
        runtime.start();

        assert!(matches!(
            runtime.detach_domain(task),
            Err(TaskRuntimeError::Scheduler(
                KernelScheduleError::CurrentTaskCannotMigrate
            ))
        ));
        assert_eq!(runtime.current(), Some(task));
        assert_eq!(
            runtime
                .capabilities_mut(task)
                .unwrap()
                .authorize(capability, Rights::READ),
            Ok(ObjectId(7))
        );
        assert_eq!(
            runtime.context(task).unwrap().instruction_pointer(),
            0x40_0000
        );
    }

    #[test]
    fn complete_domain_crosses_typed_linear_mailbox() {
        let mut source = TaskRuntime::<1, 1>::new(1_000, 1).unwrap();
        let mut destination = TaskRuntime::<1, 1>::new(1_000, 1).unwrap();
        let task = source
            .create(Priority::RESPONSIVE, context(0x20_0000, 0x40_0000))
            .unwrap();
        let mailbox = OwnershipMailbox::new();
        mailbox
            .publish(source.detach_domain(task).unwrap())
            .ok()
            .unwrap();

        let mut ticket = mailbox.take();
        let migration = destination.attach_domain(&mut ticket).unwrap();
        assert!(ticket.is_none());
        assert_eq!(migration.source(), task);
        assert_eq!(migration.priority(), Priority::RESPONSIVE);
        assert_eq!(
            destination
                .context(migration.destination())
                .unwrap()
                .page_table(),
            PhysAddr::new(0x20_0000).unwrap()
        );
        assert!(mailbox.take().is_none());
    }

    #[test]
    fn runtime_rebalance_selects_only_an_identity_with_a_complete_domain() {
        let mut source = TaskRuntime::<3, 1>::new(1_000, 1).unwrap();
        let mut destination = TaskRuntime::<2, 1>::new(1_000, 1).unwrap();
        let current = source
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        source
            .create(Priority::RESPONSIVE, context(0x30_0000, 0x50_0000))
            .unwrap();
        source
            .create(Priority::BACKGROUND, context(0x40_0000, 0x60_0000))
            .unwrap();
        destination
            .create(Priority::NORMAL, context(0x50_0000, 0x70_0000))
            .unwrap();
        source.start();

        let mut ticket = source
            .detach_domain_for_rebalance(destination.scheduler.load())
            .unwrap();
        let migration = destination.attach_domain(&mut ticket).unwrap();
        assert_ne!(migration.source(), current);
        assert_eq!(migration.priority(), Priority::RESPONSIVE);
        assert_eq!(
            destination
                .context(migration.destination())
                .unwrap()
                .instruction_pointer(),
            0x50_0000
        );
    }

    #[test]
    fn periodic_domain_balancer_retries_linear_ticket_before_new_work() {
        let mut source = TaskRuntime::<3, 1>::new(1_000, 1).unwrap();
        let current = source
            .create(Priority::NORMAL, context(0x20_0000, 0x40_0000))
            .unwrap();
        let migrating = source
            .create(Priority::RESPONSIVE, context(0x30_0000, 0x50_0000))
            .unwrap();
        let capability = source
            .capabilities_mut(migrating)
            .unwrap()
            .insert(ObjectId(19), Rights::READ)
            .unwrap();
        source
            .create(Priority::BACKGROUND, context(0x40_0000, 0x60_0000))
            .unwrap();
        assert!(matches!(source.start(), ScheduleOutcome::Switch { to, .. } if to == current));

        let mut blocker = TaskRuntime::<1, 1>::new(1_000, 1).unwrap();
        let blocker_task = blocker
            .create(Priority::NORMAL, context(0x50_0000, 0x70_0000))
            .unwrap();
        let local_mailbox = OwnershipMailbox::new();
        let remote_mailbox = OwnershipMailbox::new();
        remote_mailbox
            .publish(blocker.detach_domain(blocker_task).unwrap())
            .ok()
            .unwrap();
        let mailboxes = [&local_mailbox, &remote_mailbox];
        let loads = [source.load(), SchedulerLoad::new(0, 0, 3).unwrap()];
        let mut balancer = PeriodicDomainBalancer::<2, 1>::new(0, 1).unwrap();

        assert!(matches!(
            balancer
                .poll_and_publish(1, 0, &loads, &mut source, &mailboxes)
                .unwrap(),
            DomainBalanceOutcome::RetryPending(target) if target.cpu() == 1
        ));
        assert_eq!(balancer.pending_target().unwrap().cpu(), 1);
        assert_eq!(source.load().runnable(), 2);

        let _occupied = remote_mailbox.take().unwrap();
        assert!(matches!(
            balancer
                .poll_and_publish(2, 0, &loads, &mut source, &mailboxes)
                .unwrap(),
            DomainBalanceOutcome::Published(target) if target.cpu() == 1
        ));
        assert!(balancer.pending_target().is_none());

        let mut destination = TaskRuntime::<1, 1>::new(1_000, 1).unwrap();
        let mut ticket = remote_mailbox.take();
        let migration = destination.attach_domain(&mut ticket).unwrap();
        assert_eq!(migration.source(), migrating);
        assert_eq!(
            destination
                .context(migration.destination())
                .unwrap()
                .page_table(),
            PhysAddr::new(0x30_0000).unwrap()
        );
        assert_eq!(
            destination
                .capabilities_mut(migration.destination())
                .unwrap()
                .authorize(capability, Rights::READ),
            Ok(ObjectId(19))
        );
    }

    #[test]
    fn periodic_domain_balancer_rejects_invalid_cadence() {
        assert!(matches!(
            PeriodicDomainBalancer::<1, 0>::new(0, 1),
            Err(DomainBalanceError::Policy(
                BalancePolicyError::InvalidCpuCount
            ))
        ));
        assert!(matches!(
            PeriodicDomainBalancer::<2, 0>::new(0, 0),
            Err(DomainBalanceError::Policy(
                BalancePolicyError::InvalidInterval
            ))
        ));
    }
}
