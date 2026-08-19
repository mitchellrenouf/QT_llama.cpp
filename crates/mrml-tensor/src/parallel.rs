use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicBool, Ordering};
use mrml_runtime::{OnceCell, Shared, Vector};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, mpsc};

struct Work {
    context: usize,
    start: usize,
    end: usize,
    run: unsafe fn(usize, usize, usize),
    done: mpsc::SyncSender<()>,
    failed: Shared<AtomicBool>,
}

unsafe impl Send for Work {}

struct Pool {
    sender: mpsc::Sender<Work>,
    workers: usize,
}

fn pool() -> &'static Pool {
    static POOL: OnceCell<Pool> = OnceCell::new();
    POOL.get_or_init(|| {
        let workers = std::thread::available_parallelism().map_or(1, usize::from);
        let (sender, receiver) = mpsc::channel::<Work>();
        let receiver = Shared::new(Mutex::new(receiver));
        for index in 0..workers {
            let receiver = receiver.clone();
            std::thread::Builder::new()
                .name(format!("mrml-worker-{index}"))
                .spawn(move || {
                    loop {
                        let work = match receiver.lock().unwrap().recv() {
                            Ok(work) => work,
                            Err(_) => break,
                        };
                        if catch_unwind(AssertUnwindSafe(|| unsafe {
                            (work.run)(work.context, work.start, work.end)
                        }))
                        .is_err()
                        {
                            work.failed.store(true, Ordering::Release);
                        }
                        let _ = work.done.send(());
                    }
                })
                .expect("failed to start MRML worker");
        }
        Pool { sender, workers }
    })
}

pub fn for_each_range<F>(len: usize, minimum_chunk: usize, operation: F)
where
    F: Fn(usize, usize) + Sync,
{
    let pool = pool();
    if pool.workers <= 1 || len <= minimum_chunk {
        operation(0, len);
        return;
    }

    unsafe fn invoke<F: Fn(usize, usize) + Sync>(context: usize, start: usize, end: usize) {
        unsafe { (&*(context as *const F))(start, end) }
    }

    let jobs = pool.workers.min(len.div_ceil(minimum_chunk));
    let chunk = len.div_ceil(jobs);
    let (done_tx, done_rx) = mpsc::sync_channel(jobs);
    let failed = Shared::new(AtomicBool::new(false));
    let context = (&operation as *const F) as usize;
    let mut submitted = 0;
    for start in (0..len).step_by(chunk) {
        let end = (start + chunk).min(len);
        pool.sender
            .send(Work {
                context,
                start,
                end,
                run: invoke::<F>,
                done: done_tx.clone(),
                failed: failed.clone(),
            })
            .expect("MRML worker pool stopped");
        submitted += 1;
    }
    for _ in 0..submitted {
        done_rx
            .recv()
            .expect("MRML worker stopped before completion");
    }
    assert!(
        !failed.load(Ordering::Acquire),
        "MRML worker operation panicked"
    );
}

pub fn map<T, F>(len: usize, minimum_chunk: usize, operation: F) -> Vector<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let mut output = Vector::<MaybeUninit<T>>::with_capacity(len)
        .expect("MRML parallel output allocation failed");
    for _ in 0..len {
        output.push(MaybeUninit::uninit());
    }
    let output_address = output.as_mut_ptr() as usize;
    for_each_range(len, minimum_chunk, |start, end| {
        for index in start..end {
            // SAFETY: workers receive disjoint ranges and output lives until all finish.
            unsafe {
                (output_address as *mut MaybeUninit<T>)
                    .add(index)
                    .write(MaybeUninit::new(operation(index)));
            }
        }
    });
    let (pointer, length, capacity) = output.into_raw_parts();
    // SAFETY: all elements were initialized, and layouts are identical.
    unsafe { Vector::from_raw_parts(pointer.cast::<T>(), length, capacity) }
}

#[cfg(test)]
mod tests {
    #[test]
    fn maps_every_item_once_in_order() {
        let values = super::map(10_000, 64, |index| index * 2);
        assert_eq!(values.len(), 10_000);
        assert!(
            values
                .iter()
                .enumerate()
                .all(|(index, value)| *value == index * 2)
        );
    }
}
