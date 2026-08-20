#![no_std]

use core::hint::black_box;
use mrml_kernel::{CapabilitySpace, ObjectId, Priority, Rights, Scheduler};
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
    let picoseconds_per_operation = elapsed.saturating_mul(1000) / ITERATIONS as u128;
    mrml_println!(
        "MRML_KERNEL_BENCH capability_authorize_total_ns={} capability_authorize_ps={} iterations={}",
        elapsed,
        picoseconds_per_operation,
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
    let picoseconds_per_operation = elapsed.saturating_mul(1000) / ITERATIONS as u128;
    mrml_println!(
        "MRML_KERNEL_BENCH scheduler_select_total_ns={} scheduler_select_ps={} iterations={}",
        elapsed,
        picoseconds_per_operation,
        ITERATIONS
    );
    assert!(per_operation < MAX_NANOSECONDS_PER_OPERATION);
}
