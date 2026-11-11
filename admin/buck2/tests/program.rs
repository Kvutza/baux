//! Direct Rust execution tests, independent of the Python and OCaml bindings.
use baux_program::{Buffer, Executable, Tensor, TvmFfiVersion, TvmFunction, TvmModule};

fn spec() -> Tensor {
    Tensor::new(vec![4], "float32").unwrap()
}
fn load(device: &str) -> Executable {
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
    for device in ["cpu", "metal"] {
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
fn links_tvm() {
    let version = TvmFfiVersion::linked();
    assert_eq!((version.major, version.minor), (0, 1));
    assert!(
        TvmFunction::global("ffi.ModuleLoadFromFile")
            .unwrap()
            .is_some()
    );
    assert!(
        TvmFunction::global("relax.BlockBuilderCreate")
            .unwrap()
            .is_some()
    );
    let path = std::env::var("BAUX_TVM_TEST_LIBRARY").unwrap();
    let _module = TvmModule::load(&path).unwrap();
    let add_one = TvmFunction::global("testing.add_one").unwrap().unwrap();
    assert_eq!(add_one.call_i64(10).unwrap(), 11);
    let input = baux_program::TvmTensor::put(spec(), &[0; 16]).unwrap();
    let echo = TvmFunction::global("testing.echo").unwrap().unwrap();
    let outputs = echo.run(&[&input]).unwrap();
    drop(input);
    assert_eq!(outputs[0].spec(), &spec());
    assert_eq!(outputs[0].read().unwrap(), vec![0; 16]);
    let array = TvmFunction::global("ffi.Array").unwrap().unwrap();
    let pair = array.run(&[&outputs[0], &outputs[0]]).unwrap();
    drop(outputs);
    assert_eq!(pair.len(), 2);
    assert_eq!(pair[1].read().unwrap(), vec![0; 16]);
}

#[test]
fn tvm_tensors() {
    use baux_program::TvmTensor;
    for dtype in [
        "bool", "int32", "int64", "uint32", "uint64", "float16", "bfloat16", "float32", "float64",
    ] {
        let width = match dtype {
            "bool" => 1,
            "float16" | "bfloat16" => 2,
            "int64" | "uint64" | "float64" => 8,
            _ => 4,
        };
        for shape in [vec![], vec![2, 3], vec![0]] {
            let length = shape.iter().product::<usize>() * width;
            let bytes = vec![1; length];
            let spec = Tensor::new(shape, dtype).unwrap();
            let tensor = TvmTensor::put(spec.clone(), &bytes).unwrap();
            assert_eq!(tensor.spec(), &spec);
            assert_eq!(tensor.read().unwrap(), bytes);
        }
    }
    assert!(TvmTensor::put(spec(), &[0; 3]).is_err());
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
