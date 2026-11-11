//! Direct Rust execution tests, independent of the Python and OCaml bindings.
use baux_program::{Buffer, Executable, Tensor};

fn spec() -> Tensor {
    Tensor::new(vec![4], "float32").unwrap()
}
fn load(device: &str) -> Executable {
    if device == "tvm" {
        return Executable::load_tvm(
            &std::env::var("BAUX_TVM").unwrap(),
            "main",
            "cpu",
            vec![spec(), spec()],
            vec![spec(), spec()],
        )
        .unwrap();
    }
    let key = if device == "cpu" {
        "BAUX_VMFB_CPU"
    } else {
        "BAUX_VMFB_METAL"
    };
    let bytes = std::fs::read(std::env::var(key).unwrap()).unwrap();
    Executable::load(
        &bytes,
        "round.main",
        device,
        vec![spec(), spec()],
        vec![spec(), spec()],
    )
    .unwrap()
}
fn put(executable: &Executable, values: [f32; 4]) -> Buffer {
    let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_ne_bytes).collect();
    executable.put(spec(), &bytes).unwrap()
}
fn values(buffer: &Buffer) -> Vec<f32> {
    buffer
        .read()
        .unwrap()
        .chunks_exact(4)
        .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
        .collect()
}

#[test]
fn rounds() {
    for device in ["cpu", "metal", "tvm"] {
        let executable = load(device);
        let weights = put(&executable, [1., -2., 3., 0.]);
        let noise = put(&executable, [0.25, -0.5, 0., 2.]);
        let first = executable.run(&[weights.clone(), noise.clone()]).unwrap();
        let second = executable
            .submit(vec![first[0].clone(), noise.clone()])
            .unwrap();
        drop(first);
        drop(executable);
        drop(weights);
        drop(noise);
        let second = second.wait().unwrap();
        assert_eq!(values(&second[0]), [1.5, -3., 3., 4.]);
        assert_eq!(values(&second[1]), [2.25, 9., 9., 16.]);
    }
}

#[test]
fn tvm() {
    let path = std::env::var("BAUX_TVM").unwrap();
    assert!(
        Executable::load_tvm("/missing/baux-program.so", "main", "cpu", vec![], vec![]).is_err()
    );
    assert!(Executable::load_tvm(&path, "missing", "cpu", vec![], vec![]).is_err());
    let program = load("tvm");
    let input = put(&program, [1.; 4]);
    assert!(program.put(spec(), &[0; 3]).is_err());
    assert!(program.run(&[]).is_err());
    let foreign = put(&load("cpu"), [0.; 4]);
    assert!(program.run(&[input.clone(), foreign]).is_err());
    assert!(unsafe { input.metal() }.is_err());
    let wrong = Executable::load_tvm(
        &path,
        "main",
        "cpu",
        vec![spec(), spec()],
        vec![Tensor::new(vec![2], "float32").unwrap(), spec()],
    )
    .unwrap();
    let value = put(&wrong, [0.; 4]);
    assert!(
        wrong
            .submit(vec![value.clone(), value])
            .unwrap()
            .wait()
            .is_err()
    );
    let calls: Vec<_> = (0..8)
        .map(|_| program.submit(vec![input.clone(), input.clone()]).unwrap())
        .collect();
    drop((program, input));
    for call in calls {
        assert_eq!(values(&call.wait().unwrap()[0]), [2.; 4]);
    }
}

#[test]
fn contracts() {
    assert!(Executable::load(b"invalid", "round.main", "cpu", vec![], vec![]).is_err());
    let executable = load("cpu");
    assert!(executable.put(spec(), &[0; 3]).is_err());
    assert!(executable.run(&[]).is_err());
    let other = load("cpu");
    let foreign = put(&other, [0.; 4]);
    assert!(executable.run(&[foreign.clone(), foreign]).is_err());
    let wrong = executable
        .put(Tensor::new(vec![2], "float32").unwrap(), &[0; 8])
        .unwrap();
    assert!(executable.submit(vec![wrong.clone(), wrong]).is_err());
    let bytes = std::fs::read(std::env::var("BAUX_VMFB_CPU").unwrap()).unwrap();
    assert!(Executable::load(&bytes, "round.missing", "cpu", vec![], vec![]).is_err());
    let mismatch = Executable::load(
        &bytes,
        "round.main",
        "cpu",
        vec![spec(), spec()],
        vec![Tensor::new(vec![2], "float32").unwrap(), spec()],
    )
    .unwrap();
    let input = put(&mismatch, [0.; 4]);
    assert!(
        mismatch
            .submit(vec![input.clone(), input])
            .unwrap()
            .wait()
            .is_err()
    );
}

#[test]
fn queued() {
    let executable = load("metal");
    let input = put(&executable, [2.; 4]);
    for _ in 0..8 {
        drop(
            executable
                .submit(vec![input.clone(), input.clone()])
                .unwrap(),
        );
    }
    let calls: Vec<_> = (0..8)
        .map(|_| {
            let pending = executable
                .submit(vec![input.clone(), input.clone()])
                .unwrap();
            std::thread::spawn(move || values(&pending.wait().unwrap()[0]))
        })
        .collect();
    drop(input);
    drop(executable);
    for call in calls {
        assert_eq!(call.join().unwrap(), [4.; 4]);
    }
}

#[test]
fn native() {
    let executable = Executable::native("metal", vec![spec()], vec![spec()], |_, inputs| {
        Ok(vec![inputs[0].clone()])
    })
    .unwrap();
    let input = put(&executable, [1., 2., 3., 4.]);
    let output = executable.submit(vec![input]).unwrap().wait().unwrap();
    let compiled = load("metal");
    let adopted = compiled.adopt(&output[0]).unwrap();
    drop(output);
    drop(executable);
    let results = compiled.run(&[adopted.clone(), adopted]).unwrap();
    assert_eq!(values(&results[0]), [2., 4., 6., 8.]);
    assert_eq!(values(&results[1]), [4., 16., 36., 64.]);
}

#[test]
fn imports() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Lease(Buffer, Arc<AtomicUsize>);
    impl Drop for Lease {
        fn drop(&mut self) {
            self.1.fetch_add(1, Ordering::SeqCst);
        }
    }
    let source = load("metal");
    let buffer = put(&source, [1., 2., 3., 4.]);
    let (handle, offset) = unsafe { buffer.metal().unwrap() };
    let target = load("metal");
    let dropped = Arc::new(AtomicUsize::new(0));
    let lease = Arc::new(Lease(buffer, dropped.clone()));
    assert_eq!(values(&lease.0), [1., 2., 3., 4.]);
    // The lease retains the immutable source; imports cannot outlive its
    // storage.
    let view = unsafe {
        target
            .import_metal(spec(), handle, offset as usize, lease.clone())
            .unwrap()
    };
    let sub = unsafe {
        target
            .import_metal(
                Tensor::new(vec![2], "float32").unwrap(),
                handle,
                offset as usize + 4,
                lease.clone(),
            )
            .unwrap()
    };
    assert_eq!(values(&sub), [2., 3.]);
    assert!(unsafe { target.import_metal(spec(), handle, usize::MAX, lease.clone()) }.is_err());
    let output = target.run(&[view.clone(), view.clone()]).unwrap();
    let third = load("metal");
    let adopted = third.adopt(&output[0]).unwrap();
    drop((source, target, view, sub, output, lease));
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    assert_eq!(values(&adopted), [2., 4., 6., 8.]);
    drop(adopted);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn native_contracts() {
    let wrong = Executable::native("cpu", vec![spec()], vec![spec()], |_, _| Ok(vec![])).unwrap();
    assert!(wrong.run(&[put(&wrong, [0.; 4])]).is_err());
    let foreign = load("cpu");
    let foreign = put(&foreign, [0.; 4]);
    let wrong = Executable::native("cpu", vec![], vec![spec()], move |_, _| {
        Ok(vec![foreign.clone()])
    })
    .unwrap();
    assert!(wrong.run(&[]).is_err());
}

#[test]
fn abandoned() {
    use std::sync::{Arc, Barrier, mpsc};
    struct Lease {
        _buffer: Buffer,
        release: mpsc::Sender<()>,
    }
    impl Drop for Lease {
        fn drop(&mut self) {
            let _ = self.release.send(());
        }
    }
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let gate = (entered.clone(), release.clone());
    let program = Executable::native("metal", vec![spec()], vec![spec()], move |program, _| {
        gate.0.wait();
        gate.1.wait();
        Ok(vec![put(program, [0.; 4])])
    })
    .unwrap();
    let source = load("metal");
    let buffer = put(&source, [1.; 4]);
    let (handle, offset) = unsafe { buffer.metal().unwrap() };
    let (send, receive) = mpsc::channel();
    let lease = Arc::new(Lease {
        _buffer: buffer,
        release: send,
    });
    let input = unsafe {
        program
            .import_metal(spec(), handle, offset as usize, lease)
            .unwrap()
    };
    let pending = program.submit(vec![input]).unwrap();
    entered.wait();
    drop((pending, program, source));
    assert!(matches!(receive.try_recv(), Err(mpsc::TryRecvError::Empty)));
    release.wait();
    receive
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
}
