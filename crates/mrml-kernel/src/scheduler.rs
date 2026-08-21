use core::array;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TaskId {
    slot: u32,
    generation: u32,
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
    Scheduler(SchedulerError),
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
}
