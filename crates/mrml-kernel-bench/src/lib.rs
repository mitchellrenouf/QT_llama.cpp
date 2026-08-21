#![no_std]

//! Host-only performance gates for `mrml-kernel`. Keeping this crate separate
//! prevents operating-system timing and output services from entering the UEFI
//! kernel dependency graph.

#[cfg(test)]
mod tests {
    use core::hint::black_box;
    use mrml_kernel::{
        CapabilitySpace, KernelScheduler, ObjectId, Priority, Rights, Scheduler, VmTable,
    };
    use mrml_runtime::{Instant, mrml_println};

    const ITERATIONS: u64 = 1_000_000;
    const MAX_NANOSECONDS_PER_OPERATION: u128 = 10_000;

    #[test]
    #[ignore = "manual release-mode performance gate"]
    fn capability_authorization_budget() {
        let mut space = CapabilitySpace::<64>::new();
        let capability = space.insert(ObjectId(7), Rights::READ).unwrap();
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(
                space
                    .authorize(black_box(capability), Rights::READ)
                    .unwrap(),
            );
        }
        let elapsed = start.elapsed().as_nanos();
        let per_operation = elapsed / ITERATIONS as u128;
        let picoseconds = elapsed.saturating_mul(1000) / ITERATIONS as u128;
        mrml_println!(
            "MRML_KERNEL_BENCH capability_authorize_total_ns={} capability_authorize_ps={} iterations={}",
            elapsed,
            picoseconds,
            ITERATIONS
        );
        assert!(per_operation < MAX_NANOSECONDS_PER_OPERATION);
    }

    #[test]
    #[ignore = "manual release-mode performance gate"]
    fn scheduler_selection_budget() {
        let mut scheduler = Scheduler::<64>::new();
        for index in 0..64 {
            let priority = if index == 0 {
                Priority::RESPONSIVE
            } else {
                Priority::NORMAL
            };
            scheduler.create(priority).unwrap();
        }
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(scheduler.schedule().unwrap());
        }
        let elapsed = start.elapsed().as_nanos();
        let per_operation = elapsed / ITERATIONS as u128;
        let picoseconds = elapsed.saturating_mul(1000) / ITERATIONS as u128;
        mrml_println!(
            "MRML_KERNEL_BENCH scheduler_select_total_ns={} scheduler_select_ps={} iterations={}",
            elapsed,
            picoseconds,
            ITERATIONS
        );
        assert!(per_operation < MAX_NANOSECONDS_PER_OPERATION);
    }

    #[test]
    #[ignore = "manual release-mode performance gate"]
    fn vm_exit_accounting_budget() {
        let mut table = VmTable::<8>::new();
        let id = table.create(ITERATIONS).unwrap();
        table.mark_loaded(id).unwrap();
        table.start(id).unwrap();
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(table.account_exit(black_box(id)).unwrap());
        }
        let elapsed = start.elapsed().as_nanos();
        let per_operation = elapsed / ITERATIONS as u128;
        let picoseconds = elapsed.saturating_mul(1000) / ITERATIONS as u128;
        mrml_println!(
            "MRML_KERNEL_BENCH vm_exit_account_total_ns={} vm_exit_account_ps={} iterations={}",
            elapsed,
            picoseconds,
            ITERATIONS
        );
        assert!(per_operation < MAX_NANOSECONDS_PER_OPERATION);
    }

    #[test]
    #[ignore = "manual release-mode performance gate"]
    fn timer_tick_accounting_budget() {
        let mut scheduler = KernelScheduler::<1>::new(10_000, 10_000).unwrap();
        scheduler.create(Priority::NORMAL).unwrap();
        scheduler.start();
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(scheduler.timer_tick().unwrap());
        }
        let elapsed = start.elapsed().as_nanos();
        let per_operation = elapsed / ITERATIONS as u128;
        let picoseconds = elapsed.saturating_mul(1000) / ITERATIONS as u128;
        mrml_println!(
            "MRML_KERNEL_BENCH timer_tick_total_ns={} timer_tick_ps={} iterations={}",
            elapsed,
            picoseconds,
            ITERATIONS
        );
        assert!(per_operation < MAX_NANOSECONDS_PER_OPERATION);
    }
}
