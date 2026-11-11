//! Completion of a serialized native invocation; no host tensor readback.
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::task::{Context, Poll, Waker};

use super::Buffer;

type Outcome<T> = Result<Vec<T>, String>;
type Job = Box<dyn FnOnce() + Send>;

#[derive(Clone)]
pub(super) struct Queue(mpsc::Sender<Job>);
impl Queue {
    pub fn new() -> Result<Self, String> {
        let (send, receive) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("baux-program".into())
            .spawn(move || {
                while let Ok(job) = receive.recv() {
                    job();
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self(send))
    }

    pub fn submit<T: Send + 'static>(
        &self,
        job: impl FnOnce() -> Outcome<T> + Send + 'static,
    ) -> Result<Pending<T>, String> {
        let shared = Arc::new(Completion {
            state: Mutex::new(State {
                value: None,
                waker: None,
            }),
            ready: Condvar::new(),
        });
        let completion = shared.clone();
        self.0
            .send(Box::new(move || {
                let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job))
                    .unwrap_or_else(|_| Err("native invocation panicked".into()));
                let waker = {
                    let mut state = completion.state.lock().unwrap_or_else(|e| e.into_inner());
                    state.value = Some(value);
                    state.waker.take()
                };
                completion.ready.notify_all();
                if let Some(waker) = waker {
                    waker.wake();
                }
            }))
            .map_err(|_| "native execution queue is closed")?;
        Ok(Pending(shared))
    }
}

struct State<T> {
    value: Option<Outcome<T>>,
    waker: Option<Waker>,
}
struct Completion<T> {
    state: Mutex<State<T>>,
    ready: Condvar,
}

/// One invocation. Dropping it discards interest, not an in-flight device call.
/// Queued work retains its input buffers until native completion.
pub struct Pending<T = Buffer>(Arc<Completion<T>>);
impl<T> Pending<T> {
    pub fn is_ready(&self) -> bool {
        self.0
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .value
            .is_some()
    }

    pub fn wait(self) -> Outcome<T> {
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        while state.value.is_none() {
            state = self.0.ready.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state.value.take().expect("completion is ready")
    }
}
impl<T> Future for Pending<T> {
    type Output = Outcome<T>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Outcome<T>> {
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        match state.value.take() {
            Some(value) => Poll::Ready(value),
            None => {
                state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}
