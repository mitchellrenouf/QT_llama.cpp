use core::future::Future;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

/// Runs a future to completion on the current thread without a hosted async runtime.
pub fn block_on<F: Future>(future: F) -> F::Output {
    unsafe fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(core::ptr::null(), &VTABLE)
    }
    unsafe fn no_op(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
    let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) };
    let mut context = Context::from_waker(&waker);
    let mut future = core::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => mrml_runtime::yield_now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::block_on;

    #[test]
    fn repolls_pending_futures() {
        struct PendingOnce(bool);
        impl core::future::Future for PendingOnce {
            type Output = usize;
            fn poll(
                mut self: core::pin::Pin<&mut Self>,
                context: &mut core::task::Context<'_>,
            ) -> core::task::Poll<Self::Output> {
                if self.0 {
                    core::task::Poll::Ready(42)
                } else {
                    self.0 = true;
                    context.waker().wake_by_ref();
                    core::task::Poll::Pending
                }
            }
        }
        assert_eq!(block_on(PendingOnce(false)), 42);
    }
}
