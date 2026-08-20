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
}
