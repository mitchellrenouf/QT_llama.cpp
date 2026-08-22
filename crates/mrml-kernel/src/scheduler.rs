use core::{
    array,
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicU8, Ordering},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TaskId {
    slot: u32,
    generation: u32,
}

impl TaskId {
    pub const fn token(self) -> u64 {
        ((self.generation as u64) << 32) | self.slot as u64
    }

    pub(crate) const fn from_token(token: u64) -> Self {
        Self {
            slot: token as u32,
            generation: (token >> 32) as u32,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskState {
    Runnable,
    Blocked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Priority(u8);

impl Priority {
    pub const BACKGROUND: Self = Self(0);
    pub const NORMAL: Self = Self(1);
    pub const RESPONSIVE: Self = Self(2);
    pub const REALTIME: Self = Self(3);

    const fn budget(self) -> u8 {
        1 << self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulerError {
    Full,
    InvalidTask,
    RetiredSlot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelScheduleError {
    InvalidTimerFrequency,
    InvalidQuantum,
    TickExhausted,
    NoCurrentTask,
    CurrentTaskCannotMigrate,
    NoMigrationCandidate,
    Scheduler(SchedulerError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerLoad {
    occupied: usize,
    runnable: usize,
    capacity: usize,
}

impl SchedulerLoad {
    pub const fn new(occupied: usize, runnable: usize, capacity: usize) -> Option<Self> {
        if runnable > occupied || occupied > capacity {
            return None;
        }
        Some(Self {
            occupied,
            runnable,
            capacity,
        })
    }

    pub const fn occupied(self) -> usize {
        self.occupied
    }

    pub const fn runnable(self) -> usize {
        self.runnable
    }

    pub const fn capacity(self) -> usize {
        self.capacity
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BalanceOutcome {
    Balanced,
    Migrated(TaskMigration),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TaskMigration {
    source: TaskId,
    destination: TaskId,
    state: TaskState,
    priority: Priority,
}

#[derive(Debug, Eq, PartialEq)]
pub struct DetachedTask {
    source: TaskId,
    state: TaskState,
    priority: Priority,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnershipMailboxError {
    Occupied,
}

pub type MigrationMailboxError = OwnershipMailboxError;

#[derive(Debug, Eq, PartialEq)]
pub struct TaskAttachError {
    error: KernelScheduleError,
    task: DetachedTask,
}

impl TaskAttachError {
    pub const fn error(&self) -> KernelScheduleError {
        self.error
    }

    pub fn into_task(self) -> DetachedTask {
        self.task
    }
}

const MAILBOX_EMPTY: u8 = 0;
const MAILBOX_WRITING: u8 = 1;
const MAILBOX_FULL: u8 = 2;
const MAILBOX_READING: u8 = 3;

/// Allocation-free multi-producer, multi-consumer ownership transfer slot.
/// Atomic state transitions ensure only the successful publisher writes the
/// slot and only the successful receiver takes its non-copyable task ticket.
pub struct OwnershipMailbox<T> {
    state: AtomicU8,
    task: UnsafeCell<MaybeUninit<T>>,
}

// SAFETY: access to `task` is exclusively granted by the atomic state machine.
unsafe impl<T: Send> Sync for OwnershipMailbox<T> {}

impl<T> OwnershipMailbox<T> {
    pub const fn new() -> Self {
        Self {
            state: AtomicU8::new(MAILBOX_EMPTY),
            task: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    pub fn publish(&self, task: T) -> Result<(), (OwnershipMailboxError, T)> {
        if self
            .state
            .compare_exchange(
                MAILBOX_EMPTY,
                MAILBOX_WRITING,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return Err((OwnershipMailboxError::Occupied, task));
        }

        // SAFETY: the EMPTY -> WRITING transition grants this publisher the
        // only access to the uninitialized slot until FULL is published.
        unsafe { (*self.task.get()).write(task) };
        self.state.store(MAILBOX_FULL, Ordering::Release);
        Ok(())
    }

    pub fn take(&self) -> Option<T> {
        self.state
            .compare_exchange(
                MAILBOX_FULL,
                MAILBOX_READING,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .ok()?;
        // SAFETY: the FULL -> READING transition grants this receiver the only
        // access, and Acquire observes the publisher's initialized bytes.
        let task = unsafe { (*self.task.get()).assume_init_read() };
        self.state.store(MAILBOX_EMPTY, Ordering::Release);
        Some(task)
    }
}

impl<T> Default for OwnershipMailbox<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for OwnershipMailbox<T> {
    fn drop(&mut self) {
        if *self.state.get_mut() == MAILBOX_FULL {
            // SAFETY: exclusive mailbox ownership prevents concurrent state
            // transitions, and FULL proves the slot contains one value.
            unsafe { self.task.get_mut().assume_init_drop() };
        }
    }
}

pub type MigrationMailbox = OwnershipMailbox<DetachedTask>;

impl TaskMigration {
    pub const fn source(self) -> TaskId {
        self.source
    }

    pub const fn destination(self) -> TaskId {
        self.destination
    }

    pub const fn state(self) -> TaskState {
        self.state
    }

    pub const fn priority(self) -> Priority {
        self.priority
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScheduleOutcome {
    Idle,
    Continue(TaskId),
    Switch { from: Option<TaskId>, to: TaskId },
}

#[derive(Clone, Copy)]
struct Task {
    generation: u32,
    state: TaskState,
    priority: Priority,
    budget: u8,
    occupied: bool,
}

impl Task {
    const EMPTY: Self = Self {
        generation: 1,
        state: TaskState::Blocked,
        priority: Priority::NORMAL,
        budget: 0,
        occupied: false,
    };
}

/// Fixed-capacity weighted round-robin scheduler policy. Priorities affect CPU
/// share but cannot completely starve a lower-priority runnable task.
pub struct Scheduler<const TASKS: usize> {
    tasks: [Task; TASKS],
    cursor: usize,
}

impl<const TASKS: usize> Scheduler<TASKS> {
    pub fn new() -> Self {
        Self {
            tasks: array::from_fn(|_| Task::EMPTY),
            cursor: 0,
        }
    }

    pub fn create(&mut self, priority: Priority) -> Result<TaskId, SchedulerError> {
        let (slot, task) = self
            .tasks
            .iter_mut()
            .enumerate()
            .find(|(_, task)| !task.occupied && task.generation != 0)
            .ok_or(SchedulerError::Full)?;
        task.occupied = true;
        task.state = TaskState::Runnable;
        task.priority = priority;
        task.budget = priority.budget();
        Ok(TaskId {
            slot: slot as u32,
            generation: task.generation,
        })
    }

    pub fn set_state(&mut self, id: TaskId, state: TaskState) -> Result<(), SchedulerError> {
        let task = self.task_mut(id)?;
        task.state = state;
        if state == TaskState::Runnable && task.budget == 0 {
            task.budget = task.priority.budget();
        }
        Ok(())
    }

    pub fn remove(&mut self, id: TaskId) -> Result<(), SchedulerError> {
        let task = self.task_mut(id)?;
        task.occupied = false;
        task.state = TaskState::Blocked;
        task.budget = 0;
        task.generation = task.generation.checked_add(1).unwrap_or(0);
        Ok(())
    }

    pub fn schedule(&mut self) -> Option<TaskId> {
        if let Some(id) = self.select_with_budget() {
            return Some(id);
        }
        let mut runnable = false;
        for task in &mut self.tasks {
            if task.occupied && task.state == TaskState::Runnable {
                task.budget = task.priority.budget();
                runnable = true;
            }
        }
        runnable.then(|| self.select_with_budget()).flatten()
    }

    fn select_with_budget(&mut self) -> Option<TaskId> {
        for distance in 0..TASKS {
            let slot = (self.cursor + distance) % TASKS;
            let task = &mut self.tasks[slot];
            if task.occupied && task.state == TaskState::Runnable && task.budget != 0 {
                task.budget -= 1;
                self.cursor = (slot + 1) % TASKS;
                return Some(TaskId {
                    slot: slot as u32,
                    generation: task.generation,
                });
            }
        }
        None
    }

    fn task_mut(&mut self, id: TaskId) -> Result<&mut Task, SchedulerError> {
        self.tasks
            .get_mut(id.slot as usize)
            .filter(|task| task.occupied && task.generation == id.generation)
            .ok_or(SchedulerError::InvalidTask)
    }

    fn task(&self, id: TaskId) -> Result<&Task, SchedulerError> {
        self.tasks
            .get(id.slot as usize)
            .filter(|task| task.occupied && task.generation == id.generation)
            .ok_or(SchedulerError::InvalidTask)
    }

    fn load(&self) -> SchedulerLoad {
        let occupied = self.tasks.iter().filter(|task| task.occupied).count();
        let runnable = self
            .tasks
            .iter()
            .filter(|task| task.occupied && task.state == TaskState::Runnable)
            .count();
        SchedulerLoad {
            occupied,
            runnable,
            capacity: TASKS,
        }
    }

    fn migration_candidate(&self, excluded: Option<TaskId>) -> Option<TaskId> {
        for distance in 0..TASKS {
            let slot = (self.cursor + distance) % TASKS;
            let task = &self.tasks[slot];
            let id = TaskId {
                slot: slot as u32,
                generation: task.generation,
            };
            if task.occupied && task.state == TaskState::Runnable && Some(id) != excluded {
                return Some(id);
            }
        }
        None
    }
}

impl<const TASKS: usize> Default for Scheduler<TASKS> {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer-driven owner of scheduler state. Exactly one current task is tracked;
/// exceptions can revoke it without leaving a stale identity resumable.
pub struct KernelScheduler<const TASKS: usize> {
    scheduler: Scheduler<TASKS>,
    current: Option<TaskId>,
    ticks: u64,
    quantum_ticks: u32,
    remaining: u32,
}

impl<const TASKS: usize> KernelScheduler<TASKS> {
    pub fn new(ticks_per_second: u32, quantum_ticks: u32) -> Result<Self, KernelScheduleError> {
        if !(10..=100_000).contains(&ticks_per_second) {
            return Err(KernelScheduleError::InvalidTimerFrequency);
        }
        if quantum_ticks == 0 || quantum_ticks > ticks_per_second {
            return Err(KernelScheduleError::InvalidQuantum);
        }
        Ok(Self {
            scheduler: Scheduler::new(),
            current: None,
            ticks: 0,
            quantum_ticks,
            remaining: quantum_ticks,
        })
    }

    pub fn create(&mut self, priority: Priority) -> Result<TaskId, KernelScheduleError> {
        self.scheduler
            .create(priority)
            .map_err(KernelScheduleError::Scheduler)
    }

    pub const fn current(&self) -> Option<TaskId> {
        self.current
    }

    pub const fn ticks(&self) -> u64 {
        self.ticks
    }

    pub fn load(&self) -> SchedulerLoad {
        self.scheduler.load()
    }

    pub fn start(&mut self) -> ScheduleOutcome {
        if let Some(current) = self.current {
            return ScheduleOutcome::Continue(current);
        }
        self.select_replacement(None)
    }

    pub fn timer_tick(&mut self) -> Result<ScheduleOutcome, KernelScheduleError> {
        self.ticks = self
            .ticks
            .checked_add(1)
            .ok_or(KernelScheduleError::TickExhausted)?;
        let Some(current) = self.current else {
            return Ok(self.select_replacement(None));
        };
        self.remaining -= 1;
        if self.remaining != 0 {
            return Ok(ScheduleOutcome::Continue(current));
        }
        Ok(self.select_replacement(Some(current)))
    }

    pub fn yield_current(&mut self) -> Result<ScheduleOutcome, KernelScheduleError> {
        let current = self.current.ok_or(KernelScheduleError::NoCurrentTask)?;
        Ok(self.select_replacement(Some(current)))
    }

    pub fn block_current(&mut self) -> Result<ScheduleOutcome, KernelScheduleError> {
        let current = self
            .current
            .take()
            .ok_or(KernelScheduleError::NoCurrentTask)?;
        self.scheduler
            .set_state(current, TaskState::Blocked)
            .map_err(KernelScheduleError::Scheduler)?;
        Ok(self.select_replacement(Some(current)))
    }

    pub fn terminate_current(&mut self) -> Result<ScheduleOutcome, KernelScheduleError> {
        let current = self
            .current
            .take()
            .ok_or(KernelScheduleError::NoCurrentTask)?;
        self.scheduler
            .remove(current)
            .map_err(KernelScheduleError::Scheduler)?;
        Ok(self.select_replacement(Some(current)))
    }

    pub fn wake(&mut self, task: TaskId) -> Result<(), KernelScheduleError> {
        self.scheduler
            .set_state(task, TaskState::Runnable)
            .map_err(KernelScheduleError::Scheduler)
    }

    /// Moves a non-running task between exclusively owned schedulers. The
    /// destination is constructed before the source identity is retired, so a
    /// full destination cannot lose the source task. An unexpected retirement
    /// failure removes the new identity before returning.
    pub fn migrate_to<const DESTINATION_TASKS: usize>(
        &mut self,
        destination: &mut KernelScheduler<DESTINATION_TASKS>,
        task: TaskId,
    ) -> Result<TaskMigration, KernelScheduleError> {
        if self.current == Some(task) {
            return Err(KernelScheduleError::CurrentTaskCannotMigrate);
        }

        let source = self
            .scheduler
            .task(task)
            .map_err(KernelScheduleError::Scheduler)?;
        let state = source.state;
        let priority = source.priority;

        let destination_task = destination.create(priority)?;
        if state == TaskState::Blocked {
            let state_result = destination
                .scheduler
                .set_state(destination_task, TaskState::Blocked);
            if let Err(error) = state_result {
                let _ = destination.scheduler.remove(destination_task);
                return Err(KernelScheduleError::Scheduler(error));
            }
        }

        if let Err(error) = self.scheduler.remove(task) {
            let _ = destination.scheduler.remove(destination_task);
            return Err(KernelScheduleError::Scheduler(error));
        }

        Ok(TaskMigration {
            source: task,
            destination: destination_task,
            state,
            priority,
        })
    }

    /// Moves at most one runnable, non-current task when this scheduler has at
    /// least two more runnable tasks than the destination. The hysteresis
    /// prevents equal-load peers from repeatedly moving the same work.
    pub fn rebalance_to<const DESTINATION_TASKS: usize>(
        &mut self,
        destination: &mut KernelScheduler<DESTINATION_TASKS>,
    ) -> Result<BalanceOutcome, KernelScheduleError> {
        let source_load = self.load();
        let destination_load = destination.load();
        if source_load.runnable <= destination_load.runnable.saturating_add(1) {
            return Ok(BalanceOutcome::Balanced);
        }
        let candidate = self
            .scheduler
            .migration_candidate(self.current)
            .ok_or(KernelScheduleError::NoMigrationCandidate)?;
        self.migrate_to(destination, candidate)
            .map(BalanceOutcome::Migrated)
    }

    /// Retires a non-running local identity and returns its scheduling policy
    /// as a linear ticket suitable for release/acquire transfer to another CPU.
    pub fn detach(&mut self, task: TaskId) -> Result<DetachedTask, KernelScheduleError> {
        if self.current == Some(task) {
            return Err(KernelScheduleError::CurrentTaskCannotMigrate);
        }
        let source = self
            .scheduler
            .task(task)
            .map_err(KernelScheduleError::Scheduler)?;
        let detached = DetachedTask {
            source: task,
            state: source.state,
            priority: source.priority,
        };
        self.scheduler
            .remove(task)
            .map_err(KernelScheduleError::Scheduler)?;
        Ok(detached)
    }

    /// Selects and detaches at most one task using a remotely published load
    /// snapshot. A stale optimistic snapshot is safe: destination admission
    /// still owns final capacity enforcement and returns the ticket on failure.
    pub fn detach_for_rebalance(
        &mut self,
        destination: SchedulerLoad,
    ) -> Result<Option<DetachedTask>, KernelScheduleError> {
        match self.rebalance_candidate(destination)? {
            Some(candidate) => self.detach(candidate).map(Some),
            None => Ok(None),
        }
    }

    /// Selects without mutating, allowing a higher-level runtime to verify its
    /// complete domain exists before retiring the scheduler identity.
    pub fn rebalance_candidate(
        &self,
        destination: SchedulerLoad,
    ) -> Result<Option<TaskId>, KernelScheduleError> {
        let source = self.load();
        if destination.occupied == destination.capacity
            || source.runnable <= destination.runnable.saturating_add(1)
        {
            return Ok(None);
        }
        self.scheduler
            .migration_candidate(self.current)
            .map(Some)
            .ok_or(KernelScheduleError::NoMigrationCandidate)
    }

    /// Admits an exclusively owned migration ticket. Failure returns the same
    /// ticket so the caller can retry, route elsewhere, or restore it.
    pub fn attach(&mut self, task: DetachedTask) -> Result<TaskMigration, TaskAttachError> {
        let destination = match self.create(task.priority) {
            Ok(destination) => destination,
            Err(error) => return Err(TaskAttachError { error, task }),
        };
        if task.state == TaskState::Blocked {
            let state_result = self.scheduler.set_state(destination, TaskState::Blocked);
            if let Err(error) = state_result {
                let _ = self.scheduler.remove(destination);
                return Err(TaskAttachError {
                    error: KernelScheduleError::Scheduler(error),
                    task,
                });
            }
        }
        Ok(TaskMigration {
            source: task.source,
            destination,
            state: task.state,
            priority: task.priority,
        })
    }

    fn select_replacement(&mut self, from: Option<TaskId>) -> ScheduleOutcome {
        self.remaining = self.quantum_ticks;
        match self.scheduler.schedule() {
            Some(to) => {
                self.current = Some(to);
                if from == Some(to) {
                    ScheduleOutcome::Continue(to)
                } else {
                    ScheduleOutcome::Switch { from, to }
                }
            }
            None => {
                self.current = None;
                ScheduleOutcome::Idle
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_task_ids_cannot_control_reused_slots() {
        let mut scheduler = Scheduler::<1>::new();
        let old = scheduler.create(Priority::NORMAL).unwrap();
        scheduler.remove(old).unwrap();
        let replacement = scheduler.create(Priority::NORMAL).unwrap();
        assert_ne!(old, replacement);
        assert_eq!(
            scheduler.set_state(old, TaskState::Blocked),
            Err(SchedulerError::InvalidTask)
        );
    }

    #[test]
    fn weighted_priority_preserves_a_bounded_share_for_background_tasks() {
        let mut scheduler = Scheduler::<2>::new();
        let background = scheduler.create(Priority::BACKGROUND).unwrap();
        let realtime = scheduler.create(Priority::REALTIME).unwrap();
        let mut background_runs = 0;
        let mut realtime_runs = 0;
        for _ in 0..9 {
            match scheduler.schedule().unwrap() {
                id if id == background => background_runs += 1,
                id if id == realtime => realtime_runs += 1,
                _ => unreachable!(),
            }
        }
        assert_eq!(background_runs, 1);
        assert_eq!(realtime_runs, 8);
    }

    #[test]
    fn blocked_tasks_are_not_selected() {
        let mut scheduler = Scheduler::<1>::new();
        let task = scheduler.create(Priority::NORMAL).unwrap();
        scheduler.set_state(task, TaskState::Blocked).unwrap();
        assert_eq!(scheduler.schedule(), None);
        scheduler.set_state(task, TaskState::Runnable).unwrap();
        assert_eq!(scheduler.schedule(), Some(task));
    }

    #[test]
    fn timer_preempts_only_at_the_validated_quantum() {
        let mut kernel = KernelScheduler::<2>::new(1_000, 3).unwrap();
        let first = kernel.create(Priority::NORMAL).unwrap();
        let second = kernel.create(Priority::NORMAL).unwrap();
        assert_eq!(
            kernel.start(),
            ScheduleOutcome::Switch {
                from: None,
                to: first
            }
        );
        assert_eq!(kernel.timer_tick(), Ok(ScheduleOutcome::Continue(first)));
        assert_eq!(kernel.timer_tick(), Ok(ScheduleOutcome::Continue(first)));
        assert_eq!(
            kernel.timer_tick(),
            Ok(ScheduleOutcome::Switch {
                from: Some(first),
                to: second
            })
        );
        assert_eq!(kernel.ticks(), 3);
    }

    #[test]
    fn voluntary_yield_selects_the_next_runnable_task() {
        let mut kernel = KernelScheduler::<2>::new(1_000, 10).unwrap();
        let first = kernel.create(Priority::NORMAL).unwrap();
        let second = kernel.create(Priority::NORMAL).unwrap();
        assert_eq!(
            kernel.start(),
            ScheduleOutcome::Switch {
                from: None,
                to: first
            }
        );
        assert_eq!(
            kernel.yield_current().unwrap(),
            ScheduleOutcome::Switch {
                from: Some(first),
                to: second
            }
        );
    }

    #[test]
    fn faulted_current_identity_is_retired_before_replacement() {
        let mut kernel = KernelScheduler::<2>::new(100, 1).unwrap();
        let faulted = kernel.create(Priority::NORMAL).unwrap();
        let survivor = kernel.create(Priority::NORMAL).unwrap();
        assert_eq!(
            kernel.start(),
            ScheduleOutcome::Switch {
                from: None,
                to: faulted
            }
        );
        assert_eq!(
            kernel.terminate_current(),
            Ok(ScheduleOutcome::Switch {
                from: Some(faulted),
                to: survivor
            })
        );
        assert_eq!(
            kernel.wake(faulted),
            Err(KernelScheduleError::Scheduler(SchedulerError::InvalidTask))
        );
    }

    #[test]
    fn blocked_and_idle_transitions_are_explicit() {
        let mut kernel = KernelScheduler::<1>::new(100, 5).unwrap();
        let task = kernel.create(Priority::BACKGROUND).unwrap();
        assert_eq!(
            kernel.start(),
            ScheduleOutcome::Switch {
                from: None,
                to: task
            }
        );
        assert_eq!(kernel.block_current(), Ok(ScheduleOutcome::Idle));
        assert_eq!(kernel.timer_tick(), Ok(ScheduleOutcome::Idle));
        kernel.wake(task).unwrap();
        assert_eq!(
            kernel.start(),
            ScheduleOutcome::Switch {
                from: None,
                to: task
            }
        );
    }

    #[test]
    fn invalid_timer_policy_fails_closed() {
        assert!(matches!(
            KernelScheduler::<1>::new(9, 1),
            Err(KernelScheduleError::InvalidTimerFrequency)
        ));
        assert!(matches!(
            KernelScheduler::<1>::new(100, 0),
            Err(KernelScheduleError::InvalidQuantum)
        ));
        assert!(matches!(
            KernelScheduler::<1>::new(100, 101),
            Err(KernelScheduleError::InvalidQuantum)
        ));
    }

    #[test]
    fn migration_preserves_policy_and_retires_the_source_identity() {
        let mut source = KernelScheduler::<2>::new(1_000, 3).unwrap();
        let mut destination = KernelScheduler::<2>::new(1_000, 7).unwrap();
        let task = source.create(Priority::RESPONSIVE).unwrap();

        let migration = source.migrate_to(&mut destination, task).unwrap();
        assert_eq!(migration.source(), task);
        assert_eq!(migration.state(), TaskState::Runnable);
        assert_eq!(migration.priority(), Priority::RESPONSIVE);
        assert_eq!(
            source.wake(task),
            Err(KernelScheduleError::Scheduler(SchedulerError::InvalidTask))
        );
        assert_eq!(
            destination.start(),
            ScheduleOutcome::Switch {
                from: None,
                to: migration.destination(),
            }
        );
    }

    #[test]
    fn migration_preserves_blocked_state() {
        let mut source = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let mut destination = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let task = source.create(Priority::BACKGROUND).unwrap();
        source
            .scheduler
            .set_state(task, TaskState::Blocked)
            .unwrap();

        let migration = source.migrate_to(&mut destination, task).unwrap();
        assert_eq!(migration.state(), TaskState::Blocked);
        assert_eq!(destination.start(), ScheduleOutcome::Idle);
        destination.wake(migration.destination()).unwrap();
        assert!(
            matches!(destination.start(), ScheduleOutcome::Switch { to, .. } if to == migration.destination())
        );
    }

    #[test]
    fn migration_rejects_the_running_task_without_mutation() {
        let mut source = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let mut destination = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let task = source.create(Priority::NORMAL).unwrap();
        source.start();

        assert_eq!(
            source.migrate_to(&mut destination, task),
            Err(KernelScheduleError::CurrentTaskCannotMigrate)
        );
        assert_eq!(source.current(), Some(task));
        assert_eq!(destination.start(), ScheduleOutcome::Idle);
    }

    #[test]
    fn full_destination_leaves_the_source_task_intact() {
        let mut source = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let mut destination = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let task = source.create(Priority::NORMAL).unwrap();
        destination.create(Priority::NORMAL).unwrap();

        assert_eq!(
            source.migrate_to(&mut destination, task),
            Err(KernelScheduleError::Scheduler(SchedulerError::Full))
        );
        assert_eq!(
            source.start(),
            ScheduleOutcome::Switch {
                from: None,
                to: task
            }
        );
    }

    #[test]
    fn detached_ticket_crosses_mailbox_and_attaches_once() {
        let mut source = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let mut destination = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let task = source.create(Priority::REALTIME).unwrap();
        let mailbox = MigrationMailbox::new();

        mailbox.publish(source.detach(task).unwrap()).unwrap();
        assert_eq!(source.start(), ScheduleOutcome::Idle);
        let migration = destination.attach(mailbox.take().unwrap()).unwrap();
        assert_eq!(migration.source(), task);
        assert_eq!(migration.priority(), Priority::REALTIME);
        assert!(mailbox.take().is_none());
    }

    #[test]
    fn occupied_mailbox_returns_the_unpublished_ticket() {
        let mut first = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let mut second = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let mailbox = MigrationMailbox::new();
        let first_id = first.create(Priority::NORMAL).unwrap();
        mailbox.publish(first.detach(first_id).unwrap()).unwrap();
        let second_id = second.create(Priority::BACKGROUND).unwrap();

        let (error, returned) = mailbox
            .publish(second.detach(second_id).unwrap())
            .unwrap_err();
        assert_eq!(error, MigrationMailboxError::Occupied);
        assert_eq!(returned.source, second_id);
    }

    #[test]
    fn failed_attach_returns_ticket_for_another_destination() {
        let mut source = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let mut full = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let mut fallback = KernelScheduler::<1>::new(1_000, 3).unwrap();
        let task = source.create(Priority::RESPONSIVE).unwrap();
        full.create(Priority::NORMAL).unwrap();

        let error = full.attach(source.detach(task).unwrap()).unwrap_err();
        assert_eq!(
            error.error(),
            KernelScheduleError::Scheduler(SchedulerError::Full)
        );
        let migration = fallback.attach(error.into_task()).unwrap();
        assert_eq!(migration.source(), task);
    }

    #[test]
    fn load_snapshot_counts_blocked_capacity_without_callers_inspecting_tasks() {
        let mut scheduler = KernelScheduler::<3>::new(1_000, 3).unwrap();
        let blocked = scheduler.create(Priority::NORMAL).unwrap();
        scheduler.create(Priority::NORMAL).unwrap();
        scheduler
            .scheduler
            .set_state(blocked, TaskState::Blocked)
            .unwrap();

        assert_eq!(
            scheduler.load(),
            SchedulerLoad {
                occupied: 2,
                runnable: 1,
                capacity: 3,
            }
        );
    }

    #[test]
    fn rebalance_moves_one_non_running_task_only_across_real_imbalance() {
        let mut source = KernelScheduler::<3>::new(1_000, 3).unwrap();
        let mut destination = KernelScheduler::<3>::new(1_000, 3).unwrap();
        let current = source.create(Priority::NORMAL).unwrap();
        source.create(Priority::RESPONSIVE).unwrap();
        source.create(Priority::BACKGROUND).unwrap();
        assert!(matches!(source.start(), ScheduleOutcome::Switch { to, .. } if to == current));

        let migration = match source.rebalance_to(&mut destination).unwrap() {
            BalanceOutcome::Migrated(migration) => migration,
            BalanceOutcome::Balanced => panic!("imbalanced schedulers were not balanced"),
        };
        assert_ne!(migration.source(), current);
        assert_eq!(source.load().runnable(), 2);
        assert_eq!(destination.load().runnable(), 1);
        assert_eq!(
            source.rebalance_to(&mut destination),
            Ok(BalanceOutcome::Balanced)
        );
    }

    #[test]
    fn rebalance_does_not_move_work_between_nearly_equal_peers() {
        let mut source = KernelScheduler::<2>::new(1_000, 3).unwrap();
        let mut destination = KernelScheduler::<2>::new(1_000, 3).unwrap();
        source.create(Priority::NORMAL).unwrap();
        destination.create(Priority::NORMAL).unwrap();

        assert_eq!(
            source.rebalance_to(&mut destination),
            Ok(BalanceOutcome::Balanced)
        );
    }

    #[test]
    fn distributed_rebalance_detaches_only_for_valid_available_remote_load() {
        assert!(SchedulerLoad::new(0, 1, 2).is_none());
        assert!(SchedulerLoad::new(3, 2, 2).is_none());
        let mut source = KernelScheduler::<3>::new(1_000, 3).unwrap();
        source.create(Priority::NORMAL).unwrap();
        source.create(Priority::RESPONSIVE).unwrap();
        source.create(Priority::BACKGROUND).unwrap();
        source.start();

        let balanced = SchedulerLoad::new(2, 2, 3).unwrap();
        assert!(source.detach_for_rebalance(balanced).unwrap().is_none());
        let full = SchedulerLoad::new(1, 1, 1).unwrap();
        assert!(source.detach_for_rebalance(full).unwrap().is_none());
        let underloaded = SchedulerLoad::new(1, 1, 2).unwrap();
        let detached = source.detach_for_rebalance(underloaded).unwrap().unwrap();
        assert_eq!(detached.priority, Priority::RESPONSIVE);
        assert_eq!(source.load().runnable(), 2);
    }
}
