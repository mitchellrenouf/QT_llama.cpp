use crate::{Shared, SpinMutex, Vector, yield_now};
use core::fmt;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

struct Channel<T> {
    queue: SpinMutex<Vector<T>>,
    capacity: usize,
    senders: AtomicUsize,
    receivers: AtomicUsize,
    epoch: AtomicU32,
    spin_limit: usize,
}

pub struct Sender<T> {
    channel: Shared<Channel<T>>,
}

pub struct Receiver<T> {
    channel: Shared<Channel<T>>,
}

pub struct SendError<T>(pub T);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecvError;

impl<T> fmt::Debug for SendError<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SendError(..)")
    }
}

pub fn sync_channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    channel_with_spin_limit(capacity, usize::MAX)
}

pub fn blocking_channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    channel_with_spin_limit(capacity, 0)
}

fn channel_with_spin_limit<T>(capacity: usize, spin_limit: usize) -> (Sender<T>, Receiver<T>) {
    assert!(capacity > 0, "MRML channels require positive capacity");
    let channel = Shared::new(Channel {
        queue: SpinMutex::new(
            Vector::with_capacity(capacity.min(64)).expect("MRML channel allocation failed"),
        ),
        capacity,
        senders: AtomicUsize::new(1),
        receivers: AtomicUsize::new(1),
        epoch: AtomicU32::new(0),
        spin_limit,
    });
    (
        Sender {
            channel: channel.clone(),
        },
        Receiver { channel },
    )
}

impl<T> Sender<T> {
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        loop {
            if self.channel.receivers.load(Ordering::Acquire) == 0 {
                return Err(SendError(value));
            }
            {
                let mut queue = self.channel.queue.lock();
                if queue.len() < self.channel.capacity {
                    queue.push(value);
                    self.channel.epoch.fetch_add(1, Ordering::Release);
                    wake_one(&self.channel.epoch);
                    return Ok(());
                }
            }
            yield_now();
        }
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        self.channel.senders.fetch_add(1, Ordering::Relaxed);
        Self {
            channel: self.channel.clone(),
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        self.channel.senders.fetch_sub(1, Ordering::Release);
        self.channel.epoch.fetch_add(1, Ordering::Release);
        wake_all(&self.channel.epoch);
    }
}

impl<T> Receiver<T> {
    pub fn recv(&self) -> Result<T, RecvError> {
        let mut spins = 0;
        loop {
            {
                let mut queue = self.channel.queue.lock();
                if !queue.is_empty() {
                    let value = queue.remove(0);
                    self.channel.epoch.fetch_add(1, Ordering::Release);
                    wake_one(&self.channel.epoch);
                    return Ok(value);
                }
            }
            if self.channel.senders.load(Ordering::Acquire) == 0 {
                return Err(RecvError);
            }
            if spins < self.channel.spin_limit {
                spins += 1;
                yield_now();
            } else {
                let observed = self.channel.epoch.load(Ordering::Acquire);
                if self.channel.queue.lock().is_empty()
                    && self.channel.senders.load(Ordering::Acquire) != 0
                {
                    wait(&self.channel.epoch, observed);
                }
            }
        }
    }
}

impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Self {
        self.channel.receivers.fetch_add(1, Ordering::Relaxed);
        Self {
            channel: self.channel.clone(),
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.channel.receivers.fetch_sub(1, Ordering::Release);
        self.channel.epoch.fetch_add(1, Ordering::Release);
        wake_all(&self.channel.epoch);
    }
}

fn wait(epoch: &AtomicU32, expected: u32) {
    #[cfg(windows)]
    mrml_windows::wait_on_u32(epoch.as_ptr(), expected);
    #[cfg(unix)]
    mrml_linux::wait_on_u32(epoch.as_ptr(), expected);
}

fn wake_one(epoch: &AtomicU32) {
    #[cfg(windows)]
    mrml_windows::wake_one_u32(epoch.as_ptr());
    #[cfg(unix)]
    mrml_linux::wake_one_u32(epoch.as_ptr());
}

fn wake_all(epoch: &AtomicU32) {
    #[cfg(windows)]
    mrml_windows::wake_all_u32(epoch.as_ptr());
    #[cfg(unix)]
    mrml_linux::wake_all_u32(epoch.as_ptr());
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn transfers_values_and_reports_disconnects() {
        let (sender, receiver) = sync_channel(2);
        sender.send(3).unwrap();
        sender.send(5).unwrap();
        assert_eq!(receiver.recv(), Ok(3));
        assert_eq!(receiver.recv(), Ok(5));
        drop(receiver);
        assert_eq!(sender.send(7).unwrap_err().0, 7);
    }

    #[test]
    fn receives_from_native_thread() {
        let (sender, receiver) = sync_channel(1);
        let ran = Shared::new(AtomicBool::new(false));
        let worker_ran = ran.clone();
        assert!(
            crate::spawn_detached(move || {
                sender.send(11).unwrap();
                worker_ran.store(true, Ordering::Release);
            })
            .is_ok()
        );
        assert_eq!(receiver.recv(), Ok(11));
        while !ran.load(Ordering::Acquire) {
            yield_now();
        }
    }
}
