use core::array;
use core::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use super::{ApStartupToken, MAX_X86_64_CPUS};

const OFFLINE: u8 = 0;
const PREPARING: u8 = 1;
const ARMED: u8 = 2;
const ONLINE: u8 = 3;
const FAILED: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApOnlineError {
    InvalidCpu,
    InvalidGeneration,
    InvalidState,
    StaleStartup,
}

/// Concurrent BSP/AP rendezvous. The BSP publishes a generational token before
/// SIPI; the AP acknowledges only that exact generation after entering the
/// relocated kernel entry, which proves it no longer executes from the shared
/// low trampoline page.
pub struct ApOnlineTable<const CPUS: usize> {
    generations: [AtomicU32; CPUS],
    states: [AtomicU8; CPUS],
}

impl<const CPUS: usize> ApOnlineTable<CPUS> {
    pub const fn empty() -> Self {
        assert!(CPUS != 0 && CPUS <= MAX_X86_64_CPUS);
        Self {
            generations: [const { AtomicU32::new(0) }; CPUS],
            states: [const { AtomicU8::new(OFFLINE) }; CPUS],
        }
    }

    pub fn new() -> Result<Self, ApOnlineError> {
        if CPUS == 0 || CPUS > MAX_X86_64_CPUS {
            return Err(ApOnlineError::InvalidCpu);
        }
        Ok(Self {
            generations: array::from_fn(|_| AtomicU32::new(0)),
            states: array::from_fn(|_| AtomicU8::new(OFFLINE)),
        })
    }

    pub fn matches_armed(&self, cpu: usize, generation: u32) -> bool {
        cpu < CPUS
            && generation != 0
            && self.states[cpu].load(Ordering::Acquire) == ARMED
            && self.generations[cpu].load(Ordering::Acquire) == generation
    }

    pub fn arm(&self, token: ApStartupToken) -> Result<(), ApOnlineError> {
        let cpu = usize::from(token.slot());
        if cpu >= CPUS {
            return Err(ApOnlineError::InvalidCpu);
        }
        if token.generation() == 0 {
            return Err(ApOnlineError::InvalidGeneration);
        }
        self.states[cpu]
            .compare_exchange(OFFLINE, PREPARING, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| ApOnlineError::InvalidState)?;
        self.generations[cpu].store(token.generation(), Ordering::Relaxed);
        self.states[cpu].store(ARMED, Ordering::Release);
        Ok(())
    }

    pub fn acknowledge(&self, cpu: usize, generation: u32) -> Result<(), ApOnlineError> {
        if cpu >= CPUS {
            return Err(ApOnlineError::InvalidCpu);
        }
        if generation == 0 || self.generations[cpu].load(Ordering::Acquire) != generation {
            return Err(ApOnlineError::StaleStartup);
        }
        self.states[cpu]
            .compare_exchange(ARMED, ONLINE, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| ApOnlineError::InvalidState)?;
        Ok(())
    }

    pub fn is_online(&self, token: ApStartupToken) -> Result<bool, ApOnlineError> {
        let cpu = usize::from(token.slot());
        if cpu >= CPUS {
            return Err(ApOnlineError::InvalidCpu);
        }
        if self.generations[cpu].load(Ordering::Acquire) != token.generation() {
            return Err(ApOnlineError::StaleStartup);
        }
        Ok(self.states[cpu].load(Ordering::Acquire) == ONLINE)
    }

    pub fn fail(&self, token: ApStartupToken) -> Result<(), ApOnlineError> {
        let cpu = usize::from(token.slot());
        if cpu >= CPUS {
            return Err(ApOnlineError::InvalidCpu);
        }
        if self.generations[cpu].load(Ordering::Acquire) != token.generation() {
            return Err(ApOnlineError::StaleStartup);
        }
        self.states[cpu]
            .compare_exchange(ARMED, FAILED, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| ApOnlineError::InvalidState)?;
        Ok(())
    }

    pub fn rearm_failed(&self, token: ApStartupToken) -> Result<(), ApOnlineError> {
        let cpu = usize::from(token.slot());
        if cpu >= CPUS {
            return Err(ApOnlineError::InvalidCpu);
        }
        if self.generations[cpu].load(Ordering::Acquire) != token.generation() {
            return Err(ApOnlineError::StaleStartup);
        }
        self.states[cpu]
            .compare_exchange(FAILED, OFFLINE, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| ApOnlineError::InvalidState)?;
        self.generations[cpu].store(0, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_exact_generation_can_acknowledge_and_publish_online() {
        let table = ApOnlineTable::<4>::new().unwrap();
        let token = ApStartupToken::from_parts(2, 7).unwrap();
        table.arm(token).unwrap();
        assert_eq!(table.is_online(token), Ok(false));
        assert_eq!(table.acknowledge(2, 6), Err(ApOnlineError::StaleStartup));
        table.acknowledge(2, 7).unwrap();
        assert_eq!(table.is_online(token), Ok(true));
        assert_eq!(table.acknowledge(2, 7), Err(ApOnlineError::InvalidState));
    }

    #[test]
    fn failure_and_invalid_cpu_fail_closed() {
        let table = ApOnlineTable::<2>::new().unwrap();
        assert!(ApStartupToken::from_parts(256, 1).is_err());
        let token = ApStartupToken::from_parts(1, 3).unwrap();
        table.arm(token).unwrap();
        table.fail(token).unwrap();
        assert_eq!(table.is_online(token), Ok(false));
        assert_eq!(table.arm(token), Err(ApOnlineError::InvalidState));
        table.rearm_failed(token).unwrap();
        let retry = ApStartupToken::from_parts(1, 4).unwrap();
        table.arm(retry).unwrap();
        table.acknowledge(1, 4).unwrap();
    }
}
