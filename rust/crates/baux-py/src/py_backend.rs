//! Rust-owned serialized invocation queue around Python compiler bindings.
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyList};
use std::sync::{Arc, Condvar, Mutex, mpsc};

type ResultSet = PyResult<Vec<Py<PyAny>>>;

struct Completion {
    result: Mutex<Option<ResultSet>>,
    ready: Condvar,
}

impl Completion {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            ready: Condvar::new(),
        }
    }

    fn publish(&self, result: ResultSet) {
        let mut slot = self
            .result
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *slot = Some(result);
        self.ready.notify_all();
    }

    fn is_ready(&self) -> bool {
        self.result
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_some()
    }

    fn wait(&self) -> ResultSet {
        let mut slot = self
            .result
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        while slot.is_none() {
            slot = self
                .ready
                .wait(slot)
                .unwrap_or_else(|error| error.into_inner());
        }
        slot.take().expect("completion became ready")
    }
}

struct Job {
    inputs: Vec<Py<PyAny>>,
    completion: Arc<Completion>,
}

#[pyclass(name = "BackendQueue", frozen)]
pub struct PyBackendQueue {
    sender: mpsc::Sender<Job>,
}

#[pymethods]
impl PyBackendQueue {
    #[new]
    fn new(engine: Py<PyAny>) -> Self {
        let (sender, receiver) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("baux-backend-queue".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    let result = Python::attach(|py| {
                        let inputs = PyList::new(py, job.inputs)?;
                        engine
                            .bind(py)
                            .call_method1("run", (inputs,))?
                            .extract::<Vec<Py<PyAny>>>()
                    });
                    job.completion.publish(result);
                }
            })
            .expect("backend queue worker thread");
        Self { sender }
    }

    fn submit(&self, inputs: Vec<Py<PyAny>>) -> PyResult<PyBackendCall> {
        let completion = Arc::new(Completion::new());
        self.sender
            .send(Job {
                inputs,
                completion: completion.clone(),
            })
            .map_err(|_| PyRuntimeError::new_err("backend queue worker stopped"))?;
        Ok(PyBackendCall {
            completion: Mutex::new(Some(completion)),
        })
    }

    fn run(&self, py: Python<'_>, inputs: Vec<Py<PyAny>>) -> PyResult<Vec<Py<PyAny>>> {
        self.submit(inputs)?.wait(py)
    }
}

#[pyclass(name = "BackendCall", frozen)]
pub struct PyBackendCall {
    completion: Mutex<Option<Arc<Completion>>>,
}

#[pymethods]
impl PyBackendCall {
    fn ready(&self) -> bool {
        self.completion
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .is_some_and(|completion| completion.is_ready())
    }

    fn wait(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let completion = self
            .completion
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("completion was already consumed"))?;
        py.detach(|| completion.wait())
    }
}
