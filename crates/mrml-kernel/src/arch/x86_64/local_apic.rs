use core::arch::{asm, x86_64::__cpuid};

const IA32_APIC_BASE: u32 = 0x1b;
const X2APIC_EOI: u32 = 0x80b;
const X2APIC_TASK_PRIORITY: u32 = 0x808;
const X2APIC_SPURIOUS_VECTOR: u32 = 0x80f;
const X2APIC_LVT_TIMER: u32 = 0x832;
const X2APIC_INITIAL_COUNT: u32 = 0x838;
const X2APIC_CURRENT_COUNT: u32 = 0x839;
const X2APIC_ICR: u32 = 0x830;
const X2APIC_DIVIDE_CONFIGURATION: u32 = 0x83e;
const XAPIC_BASE: usize = 0xfee0_0000;
const XAPIC_EOI: usize = 0x0b0;
const XAPIC_TASK_PRIORITY: usize = 0x080;
const XAPIC_SPURIOUS_VECTOR: usize = 0x0f0;
const XAPIC_LVT_TIMER: usize = 0x320;
const XAPIC_INITIAL_COUNT: usize = 0x380;
const XAPIC_CURRENT_COUNT: usize = 0x390;
const XAPIC_DIVIDE_CONFIGURATION: usize = 0x3e0;
const XAPIC_ICR_LOW: usize = 0x300;
const XAPIC_ICR_HIGH: usize = 0x310;
const APIC_GLOBAL_ENABLE: u64 = 1 << 11;
const X2APIC_ENABLE: u64 = 1 << 10;
const APIC_BASE_MASK: u64 = 0xffff_f000;
const LVT_MASKED: u64 = 1 << 16;
const LVT_PERIODIC: u64 = 1 << 17;
const APIC_SOFTWARE_ENABLE: u64 = 1 << 8;
const SPURIOUS_VECTOR: u64 = 0xff;
const ICR_DELIVERY_PENDING: u64 = 1 << 12;
const ICR_LEVEL_ASSERT: u64 = 1 << 14;
const ICR_TRIGGER_LEVEL: u64 = 1 << 15;
const ICR_DELIVERY_INIT: u64 = 0b101 << 8;
const ICR_DELIVERY_STARTUP: u64 = 0b110 << 8;
const ICR_POLL_LIMIT: usize = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalApicError {
    Unsupported,
    InvalidVector,
    InvalidInitialCount,
    InvalidDestination,
    DeliveryBusy,
    UncalibratedClock,
    InvalidDelay,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ApStartupTiming {
    tsc_hz: u64,
}

impl ApStartupTiming {
    pub const fn from_tsc_hz(tsc_hz: u64) -> Result<Self, LocalApicError> {
        if tsc_hz < 1_000_000 || tsc_hz > 10_000_000_000 {
            return Err(LocalApicError::UncalibratedClock);
        }
        Ok(Self { tsc_hz })
    }

    /// Detects an invariant TSC and derives its frequency only from enumerated
    /// architectural CPUID ratios. No guessed fallback is accepted.
    pub fn detect() -> Result<Self, LocalApicError> {
        let extended = __cpuid(0x8000_0000).eax;
        if extended < 0x8000_0007 || __cpuid(0x8000_0007).edx & (1 << 8) == 0 {
            return Err(LocalApicError::UncalibratedClock);
        }
        let maximum = __cpuid(0).eax;
        let mut hz = 0u64;
        if maximum >= 0x15 {
            let ratio = __cpuid(0x15);
            if ratio.eax != 0 && ratio.ebx != 0 && ratio.ecx != 0 {
                hz = u64::from(ratio.ecx)
                    .checked_mul(u64::from(ratio.ebx))
                    .and_then(|value| value.checked_div(u64::from(ratio.eax)))
                    .ok_or(LocalApicError::UncalibratedClock)?;
            }
        }
        if hz == 0 && maximum >= 0x16 {
            hz = u64::from(__cpuid(0x16).eax)
                .checked_mul(1_000_000)
                .ok_or(LocalApicError::UncalibratedClock)?;
        }
        Self::from_tsc_hz(hz)
    }

    pub const fn tsc_hz(self) -> u64 {
        self.tsc_hz
    }

    pub fn wait_after_init(self) -> Result<(), LocalApicError> {
        self.wait_micros(10_000)
    }

    pub fn wait_after_startup(self) -> Result<(), LocalApicError> {
        self.wait_micros(200)
    }

    pub fn wait_micros(self, micros: u32) -> Result<(), LocalApicError> {
        if micros == 0 || micros > 1_000_000 {
            return Err(LocalApicError::InvalidDelay);
        }
        let cycles = self
            .tsc_hz
            .checked_mul(u64::from(micros))
            .and_then(|value| value.checked_add(999_999))
            .map(|value| value / 1_000_000)
            .filter(|value| *value != 0)
            .ok_or(LocalApicError::InvalidDelay)?;
        let start = read_tsc();
        while read_tsc().wrapping_sub(start) < cycles {
            core::hint::spin_loop();
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ApicIpi {
    destination: u32,
    command: u32,
}

impl ApicIpi {
    pub const fn init(destination: u32) -> Result<Self, LocalApicError> {
        if destination == u32::MAX {
            return Err(LocalApicError::InvalidDestination);
        }
        Ok(Self {
            destination,
            command: (ICR_DELIVERY_INIT | ICR_LEVEL_ASSERT | ICR_TRIGGER_LEVEL) as u32,
        })
    }

    pub const fn init_deassert(destination: u32) -> Result<Self, LocalApicError> {
        if destination == u32::MAX {
            return Err(LocalApicError::InvalidDestination);
        }
        Ok(Self {
            destination,
            command: (ICR_DELIVERY_INIT | ICR_TRIGGER_LEVEL) as u32,
        })
    }

    pub const fn startup(destination: u32, vector: u8) -> Result<Self, LocalApicError> {
        if destination == u32::MAX {
            return Err(LocalApicError::InvalidDestination);
        }
        if vector == 0 {
            return Err(LocalApicError::InvalidVector);
        }
        Ok(Self {
            destination,
            command: ICR_DELIVERY_STARTUP as u32 | vector as u32,
        })
    }

    pub const fn destination(self) -> u32 {
        self.destination
    }

    pub const fn command(self) -> u32 {
        self.command
    }

    /// Publishes one directed INIT or SIPI command and waits for the local
    /// controller to consume it. Inter-command architectural delays remain the
    /// caller's responsibility and must be provided by a separately calibrated
    /// monotonic timer.
    ///
    /// # Safety
    ///
    /// The caller must run at CPL0 with interrupts disabled and exclusive
    /// ownership of this CPU's ICR. The xAPIC mapping requirements documented
    /// by [`LocalApicTimer::enable`] apply.
    pub unsafe fn send(self) -> Result<(), LocalApicError> {
        let base = unsafe { read_msr(IA32_APIC_BASE) };
        if base & APIC_GLOBAL_ENABLE == 0 {
            return Err(LocalApicError::Unsupported);
        }
        if base & X2APIC_ENABLE != 0 {
            unsafe {
                write_msr(
                    X2APIC_SPURIOUS_VECTOR,
                    APIC_SOFTWARE_ENABLE | SPURIOUS_VECTOR,
                )
            };
            wait_x2apic_idle()?;
            unsafe {
                write_msr(
                    X2APIC_ICR,
                    (u64::from(self.destination) << 32) | u64::from(self.command),
                )
            };
            wait_x2apic_idle()
        } else {
            if self.destination > u8::MAX as u32 || base & APIC_BASE_MASK != XAPIC_BASE as u64 {
                return Err(LocalApicError::InvalidDestination);
            }
            unsafe {
                write_xapic(
                    XAPIC_SPURIOUS_VECTOR,
                    APIC_SOFTWARE_ENABLE | SPURIOUS_VECTOR,
                )
            };
            wait_xapic_idle()?;
            unsafe { write_xapic(XAPIC_ICR_HIGH, u64::from(self.destination) << 24) };
            unsafe { write_xapic(XAPIC_ICR_LOW, u64::from(self.command)) };
            wait_xapic_idle()
        }
    }
}

fn wait_x2apic_idle() -> Result<(), LocalApicError> {
    for _ in 0..ICR_POLL_LIMIT {
        if unsafe { read_msr(X2APIC_ICR) } & ICR_DELIVERY_PENDING == 0 {
            return Ok(());
        }
        core::hint::spin_loop();
    }
    Err(LocalApicError::DeliveryBusy)
}

fn wait_xapic_idle() -> Result<(), LocalApicError> {
    for _ in 0..ICR_POLL_LIMIT {
        if u64::from(unsafe { read_xapic(XAPIC_ICR_LOW) }) & ICR_DELIVERY_PENDING == 0 {
            return Ok(());
        }
        core::hint::spin_loop();
    }
    Err(LocalApicError::DeliveryBusy)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimerDivide {
    By1,
    By2,
    By4,
    By8,
    By16,
    By32,
    By64,
    By128,
}

impl TimerDivide {
    const fn register(self) -> u64 {
        match self {
            Self::By1 => 0b1011,
            Self::By2 => 0b0000,
            Self::By4 => 0b0001,
            Self::By8 => 0b0010,
            Self::By16 => 0b0011,
            Self::By32 => 0b1000,
            Self::By64 => 0b1001,
            Self::By128 => 0b1010,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalApicTimer {
    vector: u8,
    initial_count: u32,
    divide: TimerDivide,
}

impl LocalApicTimer {
    pub const fn periodic(
        vector: u8,
        initial_count: u32,
        divide: TimerDivide,
    ) -> Result<Self, LocalApicError> {
        if vector < 32 || vector == 0xff {
            return Err(LocalApicError::InvalidVector);
        }
        if initial_count == 0 {
            return Err(LocalApicError::InvalidInitialCount);
        }
        Ok(Self {
            vector,
            initial_count,
            divide,
        })
    }

    pub const fn vector(self) -> u8 {
        self.vector
    }

    /// Enables x2APIC and arms this periodic timer. The LVT remains masked
    /// until its divisor and initial count have both been installed.
    ///
    /// # Safety
    ///
    /// The caller must execute at CPL0 after installing a present interrupt
    /// gate for `self.vector`, with a valid privilege stack and interrupts
    /// disabled. No other CPU may concurrently program this local APIC. If
    /// x2APIC is unavailable, physical `0xfee00000` must be mapped supervisor
    /// writable, uncached, and NX at the identical virtual address.
    pub unsafe fn enable(self) -> Result<(), LocalApicError> {
        let features = __cpuid(1);
        if features.edx & (1 << 9) == 0 {
            return Err(LocalApicError::Unsupported);
        }
        let base = unsafe { read_msr(IA32_APIC_BASE) };
        if features.ecx & (1 << 21) != 0 {
            unsafe { write_msr(IA32_APIC_BASE, base | APIC_GLOBAL_ENABLE | X2APIC_ENABLE) };
            unsafe {
                write_msr(
                    X2APIC_SPURIOUS_VECTOR,
                    APIC_SOFTWARE_ENABLE | SPURIOUS_VECTOR,
                )
            };
            unsafe { write_msr(X2APIC_TASK_PRIORITY, 0) };
            unsafe { write_msr(X2APIC_LVT_TIMER, LVT_MASKED | u64::from(self.vector)) };
            unsafe { write_msr(X2APIC_DIVIDE_CONFIGURATION, self.divide.register()) };
            unsafe { write_msr(X2APIC_INITIAL_COUNT, u64::from(self.initial_count)) };
            unsafe { write_msr(X2APIC_LVT_TIMER, LVT_PERIODIC | u64::from(self.vector)) };
        } else {
            if base & APIC_BASE_MASK != XAPIC_BASE as u64 {
                return Err(LocalApicError::Unsupported);
            }
            unsafe { write_msr(IA32_APIC_BASE, (base | APIC_GLOBAL_ENABLE) & !X2APIC_ENABLE) };
            unsafe {
                write_xapic(
                    XAPIC_SPURIOUS_VECTOR,
                    APIC_SOFTWARE_ENABLE | SPURIOUS_VECTOR,
                )
            };
            unsafe { write_xapic(XAPIC_TASK_PRIORITY, 0) };
            unsafe { write_xapic(XAPIC_LVT_TIMER, LVT_MASKED | u64::from(self.vector)) };
            unsafe { write_xapic(XAPIC_DIVIDE_CONFIGURATION, self.divide.register()) };
            unsafe { write_xapic(XAPIC_INITIAL_COUNT, u64::from(self.initial_count)) };
            unsafe { write_xapic(XAPIC_LVT_TIMER, LVT_PERIODIC | u64::from(self.vector)) };
        }
        Ok(())
    }

    /// Acknowledges the current local-APIC interrupt.
    ///
    /// # Safety
    ///
    /// The local APIC must be enabled on this CPU and this must be called
    /// exactly once for an accepted interrupt before permitting another one.
    /// The xAPIC mapping requirement from [`Self::enable`] also applies.
    pub unsafe fn acknowledge() {
        if unsafe { read_msr(IA32_APIC_BASE) } & X2APIC_ENABLE != 0 {
            unsafe { write_msr(X2APIC_EOI, 0) }
        } else {
            unsafe { write_xapic(XAPIC_EOI, 0) }
        }
    }

    /// Reads the live timer countdown for diagnostics and calibration.
    ///
    /// # Safety
    ///
    /// The local APIC and, when needed, its xAPIC page must satisfy the same
    /// requirements as [`Self::enable`].
    pub unsafe fn current_count() -> u32 {
        if unsafe { read_msr(IA32_APIC_BASE) } & X2APIC_ENABLE != 0 {
            unsafe { read_msr(X2APIC_CURRENT_COUNT) as u32 }
        } else {
            unsafe { read_xapic(XAPIC_CURRENT_COUNT) }
        }
    }
}

unsafe fn read_msr(register: u32) -> u64 {
    let low: u32;
    let high: u32;
    unsafe {
        asm!("rdmsr", in("ecx") register, out("eax") low, out("edx") high, options(nomem, nostack))
    };
    (u64::from(high) << 32) | u64::from(low)
}

fn read_tsc() -> u64 {
    let low: u32;
    let high: u32;
    unsafe {
        asm!("lfence", "rdtsc", out("eax") low, out("edx") high, options(nomem, nostack));
    }
    (u64::from(high) << 32) | u64::from(low)
}

unsafe fn write_msr(register: u32, value: u64) {
    unsafe {
        asm!(
            "wrmsr",
            in("ecx") register,
            in("eax") value as u32,
            in("edx") (value >> 32) as u32,
            options(nomem, nostack)
        )
    }
}

unsafe fn write_xapic(offset: usize, value: u64) {
    unsafe { ((XAPIC_BASE + offset) as *mut u32).write_volatile(value as u32) }
}

unsafe fn read_xapic(offset: usize) -> u32 {
    unsafe { ((XAPIC_BASE + offset) as *const u32).read_volatile() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timer_policy_rejects_architectural_and_spurious_vectors() {
        assert_eq!(
            LocalApicTimer::periodic(31, 1, TimerDivide::By16),
            Err(LocalApicError::InvalidVector)
        );
        assert_eq!(
            LocalApicTimer::periodic(0xff, 1, TimerDivide::By16),
            Err(LocalApicError::InvalidVector)
        );
        assert_eq!(
            LocalApicTimer::periodic(32, 0, TimerDivide::By16),
            Err(LocalApicError::InvalidInitialCount)
        );
        assert_eq!(
            LocalApicTimer::periodic(32, 100_000, TimerDivide::By16)
                .unwrap()
                .vector(),
            32
        );
    }

    #[test]
    fn divisor_encoding_matches_the_xapic_architecture() {
        assert_eq!(TimerDivide::By1.register(), 0b1011);
        assert_eq!(TimerDivide::By2.register(), 0);
        assert_eq!(TimerDivide::By4.register(), 1);
        assert_eq!(TimerDivide::By8.register(), 2);
        assert_eq!(TimerDivide::By16.register(), 3);
        assert_eq!(TimerDivide::By32.register(), 8);
        assert_eq!(TimerDivide::By64.register(), 9);
        assert_eq!(TimerDivide::By128.register(), 10);
    }

    #[test]
    fn directed_init_and_startup_encodings_are_exact() {
        let init = ApicIpi::init(0x1234).unwrap();
        assert_eq!(init.destination(), 0x1234);
        assert_eq!(init.command(), 0x0000_c500);
        let deassert = ApicIpi::init_deassert(0x1234).unwrap();
        assert_eq!(deassert.destination(), 0x1234);
        assert_eq!(deassert.command(), 0x0000_8500);
        let startup = ApicIpi::startup(7, 8).unwrap();
        assert_eq!(startup.destination(), 7);
        assert_eq!(startup.command(), 0x0000_0608);
        assert_eq!(
            ApicIpi::init(u32::MAX),
            Err(LocalApicError::InvalidDestination)
        );
        assert_eq!(
            ApicIpi::init_deassert(u32::MAX),
            Err(LocalApicError::InvalidDestination)
        );
        assert_eq!(ApicIpi::startup(1, 0), Err(LocalApicError::InvalidVector));
    }

    #[test]
    fn startup_timing_is_bounded_and_never_guessed() {
        assert_eq!(
            ApStartupTiming::from_tsc_hz(999_999),
            Err(LocalApicError::UncalibratedClock)
        );
        let timing = ApStartupTiming::from_tsc_hz(3_000_000_000).unwrap();
        assert_eq!(timing.tsc_hz(), 3_000_000_000);
        assert_eq!(timing.wait_micros(0), Err(LocalApicError::InvalidDelay));
        assert_eq!(
            timing.wait_micros(1_000_001),
            Err(LocalApicError::InvalidDelay)
        );
    }
}
