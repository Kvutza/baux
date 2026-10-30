//! Shared Apple GPU runtime used by BAUX Metal execution paths.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use metal::objc::{
    __send_message as send_message,
    runtime::{BOOL, Class, Object, Sel, YES},
};
use metal::{
    Buffer, BufferRef, CommandBufferRef, CommandQueue, ComputeCommandEncoderRef,
    ComputePipelineState, Device, Function, FunctionRef, Library,
};

#[path = "apple_gpu/archive.rs"]
mod archive;
#[allow(unused_imports)]
pub(crate) use archive::*;

static RUNTIME: OnceLock<Result<Arc<Runtime>, String>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Target {
    G13,
    G14,
    G15,
    G16,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub name: String,
    pub target: Target,
}

pub fn device_info() -> Result<DeviceInfo, String> {
    Runtime::shared().map(|runtime| runtime.info().clone())
}

pub(crate) fn thread_group(width: u64) -> metal::MTLSize {
    metal::MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

pub(crate) fn bind_buffers(encoder: &ComputeCommandEncoderRef, buffers: &[(&BufferRef, u64)]) {
    for (slot, &(buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(slot as u64, Some(buffer), offset);
    }
}

/// Issue an in-pass execution and memory barrier covering all buffer resources
/// (`[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]`).
pub(crate) fn barrier_buffers(encoder: &ComputeCommandEncoderRef) {
    static BARRIER_SEL: OnceLock<Sel> = OnceLock::new();
    let sel = BARRIER_SEL.get_or_init(|| Sel::register("memoryBarrierWithScope:"));
    unsafe {
        let _ = send_message::<Object, (u64,), ()>(
            encoder as *const ComputeCommandEncoderRef as *const Object,
            *sel,
            (1u64,),
        );
    }
}

pub(crate) fn encode_value<T>(encoder: &ComputeCommandEncoderRef, slot: u64, value: &T) {
    encoder.set_bytes(
        slot,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast(),
    );
}

pub(crate) fn buffer_slice<T: Copy>(buffer: &BufferRef, len: usize) -> Result<&[T], String> {
    let elem = std::mem::size_of::<T>();
    if elem == 0 {
        return Err("zero-sized buffer element type".into());
    }
    let cap = (buffer.length() as usize) / elem;
    if len > cap {
        return Err(format!("buffer slice length {len} exceeds capacity {cap}"));
    }
    if len == 0 {
        return Ok(&[]);
    }
    let ptr = buffer.contents().cast::<T>();
    if ptr.is_null() {
        return Err("Metal buffer host pointer is null".into());
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

#[allow(clippy::mut_from_ref)]
pub(crate) fn slice_mut<T: Copy>(buffer: &BufferRef, len: usize) -> Result<&mut [T], String> {
    let elem = std::mem::size_of::<T>();
    if elem == 0 {
        return Err("zero-sized buffer element type".into());
    }
    let cap = (buffer.length() as usize) / elem;
    if len > cap {
        return Err(format!(
            "mutable buffer slice length {len} exceeds capacity {cap}"
        ));
    }
    if len == 0 {
        return Ok(&mut []);
    }
    let ptr = buffer.contents().cast::<T>();
    if ptr.is_null() {
        return Err("Metal buffer host pointer is null".into());
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

pub(crate) fn gpu_interval(command: &CommandBufferRef) -> Option<(f64, f64)> {
    if command.status() != metal::MTLCommandBufferStatus::Completed {
        return None;
    }
    static SELECTORS: OnceLock<[Sel; 2]> = OnceLock::new();
    let selectors =
        SELECTORS.get_or_init(|| [Sel::register("GPUStartTime"), Sel::register("GPUEndTime")]);
    // These public properties are read only after successful completion.
    let read = |selector| unsafe {
        send_message::<Object, (), f64>(
            command as *const CommandBufferRef as *const Object,
            selector,
            (),
        )
        .ok()
    };
    let start = read(selectors[0])?;
    let end = read(selectors[1])?;
    (start.is_finite() && end.is_finite() && start > 0.0 && end >= start).then_some((start, end))
}

pub(crate) fn gpu_seconds(command: &CommandBufferRef) -> Option<f64> {
    gpu_interval(command).map(|(start, end)| end - start)
}

/// Read the active power-source policy without changing system settings.
pub(crate) fn power_mode() -> Result<bool, String> {
    let class = Class::get("NSProcessInfo").ok_or("NSProcessInfo unavailable")?;
    let info =
        unsafe { send_message::<Class, (), *mut Object>(class, Sel::register("processInfo"), ()) }
            .map_err(|error| format!("read process info: {error:?}"))?;
    if info.is_null() {
        return Err("NSProcessInfo returned nil".into());
    }
    let enabled = unsafe {
        send_message::<Object, (), BOOL>(info, Sel::register("isLowPowerModeEnabled"), ())
    }
    .map_err(|error| format!("read Low Power Mode: {error:?}"))?;
    Ok(enabled == YES)
}

/// Reject performance comparisons made under the OS energy-saving policy.
/// Read-only; normal optimization rounds do not call this benchmark guard.
pub(crate) fn require_power() -> Result<(), String> {
    if power_mode()? {
        return Err(
            "Low Power Mode is enabled; turn it off for kernel performance comparisons".into(),
        );
    }
    Ok(())
}

pub(crate) fn trace(stage: &str) {
    if std::env::var_os("BAUX_METAL_TRACE").is_some() {
        eprintln!("[baux-metal] {stage}");
    }
}

pub(crate) struct Runtime {
    pub(crate) device: Device,
    pub(crate) queue: CommandQueue,
    info: DeviceInfo,
    libraries: Mutex<HashMap<(String, bool, Option<u64>), Library>>,
    pipelines: Mutex<HashMap<(String, String), (ComputePipelineState, Function)>>,
    archives: Mutex<HashMap<(String, String), Arc<[u8]>>>,
    captured: Mutex<HashSet<(PathBuf, String, String)>>,
    schedules: Mutex<HashMap<u64, usize>>,
}

impl Runtime {
    pub(crate) fn shared() -> Result<Arc<Self>, String> {
        RUNTIME
            .get_or_init(|| Self::new().map(Arc::new))
            .as_ref()
            .map(Arc::clone)
            .map_err(Clone::clone)
    }

    fn new() -> Result<Self, String> {
        trace("creating system default device");
        let device = Device::system_default().ok_or("no default Metal device found")?;
        trace("creating command queue");
        let name = device.name().to_string();
        let info = DeviceInfo {
            target: target_name(&name),
            name,
        };
        let queue = device.new_command_queue();
        trace("runtime ready");
        Ok(Self {
            device,
            queue,
            info,
            libraries: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
            archives: Mutex::new(HashMap::new()),
            captured: Mutex::new(HashSet::new()),
            schedules: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub(crate) fn pipeline(
        &self,
        source: &str,
        label: &str,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        self.compile(source, label, name, true, None, &[])
    }

    pub(crate) fn precise(
        &self,
        source: &str,
        label: &str,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        self.compile(source, label, name, false, None, &[])
    }

    pub(crate) fn precise_metal4(
        &self,
        source: &str,
        label: &str,
        name: &str,
        bool_constants: &[(u64, bool)],
    ) -> Result<ComputePipelineState, String> {
        self.compile(source, label, name, false, Some(4 << 16), bool_constants)
    }

    pub(crate) fn pipeline_metal4(
        &self,
        source: &str,
        label: &str,
        name: &str,
        bool_constants: &[(u64, bool)],
    ) -> Result<ComputePipelineState, String> {
        self.compile(source, label, name, true, Some(4 << 16), bool_constants)
    }

    fn compile(
        &self,
        source: &str,
        label: &str,
        name: &str,
        fast: bool,
        language_version: Option<u64>,
        bool_constants: &[(u64, bool)],
    ) -> Result<ComputePipelineState, String> {
        let evidence = std::env::var_os("BAUX_GPU_EVIDENCE").map(PathBuf::from);
        if evidence
            .as_ref()
            .is_some_and(|path| path.as_os_str().is_empty())
        {
            return Err("BAUX_GPU_EVIDENCE must name a directory".into());
        }
        self.compile_at(
            source,
            label,
            name,
            fast,
            language_version,
            bool_constants,
            evidence.as_deref(),
        )
    }

    fn compile_at(
        &self,
        source: &str,
        label: &str,
        name: &str,
        fast: bool,
        language_version: Option<u64>,
        bool_constants: &[(u64, bool)],
        evidence: Option<&Path>,
    ) -> Result<ComputePipelineState, String> {
        let specialization = bool_constants
            .iter()
            .map(|(index, value)| format!("{index}={value}"))
            .collect::<Vec<_>>()
            .join(",");
        let key = (
            baux::artifact::runs::digest(source.as_bytes()),
            format!("{name}:{fast}:{language_version:?}:{specialization}"),
        );
        let cached_pipeline = self
            .pipelines
            .lock()
            .map_err(|_| "Apple GPU pipeline cache poisoned")?
            .get(&key)
            .map(|(pipeline, function)| (pipeline.to_owned(), function.to_owned()));
        if let Some((pipeline, function)) = cached_pipeline {
            if let Some(directory) = evidence {
                self.capture(
                    directory,
                    source,
                    name,
                    fast,
                    language_version,
                    bool_constants,
                    &key,
                    &pipeline,
                    &function,
                )?;
            }
            return Ok(pipeline);
        }
        let library_key = (key.0.clone(), fast, language_version);
        let cached_library = self
            .libraries
            .lock()
            .map_err(|_| "Apple GPU library cache poisoned")?
            .get(&library_key)
            .map(ToOwned::to_owned);
        let library = if let Some(library) = cached_library {
            library.to_owned()
        } else {
            trace(&format!("compiling Metal library for {name}"));
            let options = metal::CompileOptions::new();
            options.set_fast_math_enabled(fast);
            if let Some(version) = language_version {
                unsafe {
                    send_message::<Object, (u64,), ()>(
                        options.as_ref() as *const _ as *const Object,
                        Sel::register("setLanguageVersion:"),
                        (version,),
                    )
                    .map_err(|error| format!("{label} Metal language version: {error:?}"))?;
                }
            }
            let library = self
                .device
                .new_library_with_source(source, &options)
                .map_err(|error| format!("{label} Metal compile: {error}"))?;
            trace(&format!("Metal library ready for {name}"));
            self.libraries
                .lock()
                .map_err(|_| "Apple GPU library cache poisoned")?
                .insert(library_key, library.to_owned());
            library
        };
        trace(&format!("resolving Metal function {name}"));
        let constants = (!bool_constants.is_empty()).then(|| {
            let constants = metal::FunctionConstantValues::new();
            for (index, value) in bool_constants {
                constants.set_constant_value_at_index(
                    (value as *const bool).cast(),
                    metal::MTLDataType::Bool,
                    *index,
                );
            }
            constants
        });
        let function = library
            .get_function(name, constants)
            .map_err(|error| format!("missing Metal kernel {name}: {error}"))?;
        trace(&format!("Metal function ready for {name}"));
        trace(&format!("creating Metal pipeline for {name}"));
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| format!("Metal pipeline {name}: {error}"))?;
        trace(&format!("Metal pipeline ready for {name}"));
        if let Some(directory) = evidence {
            self.capture(
                directory,
                source,
                name,
                fast,
                language_version,
                bool_constants,
                &key,
                &pipeline,
                &function,
            )?;
        }
        self.pipelines
            .lock()
            .map_err(|_| "Apple GPU pipeline cache poisoned")?
            .insert(key, (pipeline.to_owned(), function));
        Ok(pipeline)
    }

    #[allow(clippy::too_many_arguments)]
    fn capture(
        &self,
        directory: &Path,
        source: &str,
        name: &str,
        fast: bool,
        version: Option<u64>,
        constants: &[(u64, bool)],
        key: &(String, String),
        pipeline: &ComputePipelineState,
        function: &FunctionRef,
    ) -> Result<(), String> {
        let mut captured = self
            .captured
            .lock()
            .map_err(|_| "Metal capture cache poisoned")?;
        let identity = (directory.to_owned(), key.0.clone(), key.1.clone());
        if captured.contains(&identity) {
            return Ok(());
        }
        let archive = {
            let mut archives = self
                .archives
                .lock()
                .map_err(|_| "Metal archive cache poisoned")?;
            if let Some(bytes) = archives.get(key) {
                Arc::clone(bytes)
            } else {
                let bytes: Arc<[u8]> =
                    serialize_cached(&self.device, &self.info.name, key, function, directory)?
                        .into();
                archives.insert(key.clone(), Arc::clone(&bytes));
                bytes
            }
        };
        let recipe = baux::artifact::gpu::Recipe {
            device_kind: baux::artifact::gpu::DeviceKind::Metal,
            device: self.info.name.clone(),
            target: format!("{:?}", self.info.target),
            compiler: "Metal runtime compiler (MTLBinaryArchive)".into(),
            stage: baux::artifact::gpu::Stage::Compile,
            entry: name.into(),
            input_kind: baux::artifact::gpu::Kind::MetalSource,
            output_kind: Some(baux::artifact::gpu::Kind::DeviceImage),
            options: vec![
                format!("fast_math_enabled={fast}"),
                format!("language_version={version:?}"),
                format!(
                    "max_total_threads_per_threadgroup={}",
                    pipeline.max_total_threads_per_threadgroup()
                ),
                format!(
                    "thread_execution_width={}",
                    pipeline.thread_execution_width()
                ),
                format!(
                    "static_threadgroup_memory_length={}",
                    pipeline.static_threadgroup_memory_length()
                ),
            ],
            constants: constants
                .iter()
                .map(|&(index, value)| baux::artifact::gpu::Constant {
                    index,
                    kind: "bool".into(),
                    bytes: vec![u8::from(value)],
                })
                .collect(),
        };
        baux::artifact::gpu::record(directory, &recipe, source.as_bytes(), Some(&archive))?;
        captured.insert(identity);
        Ok(())
    }

    pub(crate) fn buffer<T>(&self, elements: usize) -> Buffer {
        self.device.new_buffer(
            (elements.max(1) * size_of::<T>()) as u64,
            metal::MTLResourceOptions::StorageModeShared
                | metal::MTLResourceOptions::HazardTrackingModeTracked,
        )
    }

    pub(crate) fn buffer_with<T>(&self, values: &[T]) -> Buffer {
        if values.is_empty() {
            return self.buffer::<T>(1);
        }
        self.device.new_buffer_with_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values) as u64,
            metal::MTLResourceOptions::StorageModeShared
                | metal::MTLResourceOptions::HazardTrackingModeTracked,
        )
    }

    pub(crate) fn mmap_buffer(
        &self,
        file: &std::fs::File,
        offset: i64,
        bytes: usize,
    ) -> Option<Buffer> {
        use std::os::fd::AsRawFd;
        unsafe extern "C" {
            fn mmap(
                addr: *mut std::ffi::c_void,
                len: usize,
                prot: i32,
                flags: i32,
                fd: i32,
                offset: i64,
            ) -> *mut std::ffi::c_void;
        }
        let aligned = bytes.next_multiple_of(16_384);
        if aligned == 0 || offset % 16_384 != 0 {
            return None;
        }
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                aligned,
                0x03,
                0x0002,
                file.as_raw_fd(),
                offset,
            )
        };
        if ptr.is_null() || ptr == usize::MAX as *mut std::ffi::c_void {
            return None;
        }
        let buf = self.device.new_buffer_with_bytes_no_copy(
            ptr,
            aligned as u64,
            metal::MTLResourceOptions::StorageModeShared
                | metal::MTLResourceOptions::HazardTrackingModeTracked,
            None,
        );
        (buf.length() == aligned as u64).then_some(buf)
    }

    #[allow(dead_code)]
    pub(crate) fn schedule<F>(
        &self,
        family: &str,
        shape: &[usize],
        candidates: usize,
        mut evaluate: F,
    ) -> Result<usize, String>
    where
        F: FnMut(usize) -> Result<Option<std::time::Duration>, String>,
    {
        let key = source_hash(&format!("{family}:{shape:?}"));
        if let Some(choice) = self
            .schedules
            .lock()
            .map_err(|_| "Apple GPU schedule cache poisoned")?
            .get(&key)
        {
            return Ok(*choice);
        }
        let mut best = None;
        for candidate in 0..candidates {
            let Some(elapsed) = evaluate(candidate)? else {
                continue;
            };
            if best.is_none_or(|(_, current)| elapsed < current) {
                best = Some((candidate, elapsed));
            }
        }
        let choice = best
            .map(|(candidate, _)| candidate)
            .ok_or_else(|| format!("no valid Apple GPU schedule for {family} {shape:?}"))?;
        self.schedules
            .lock()
            .map_err(|_| "Apple GPU schedule cache poisoned")?
            .insert(key, choice);
        Ok(choice)
    }

    pub(crate) fn race<F>(
        &self,
        family: &str,
        shape: &[usize],
        candidates: usize,
        rounds: usize,
        mut evaluate: F,
    ) -> Result<usize, String>
    where
        F: FnMut(usize) -> Result<Option<std::time::Duration>, String>,
    {
        let key = source_hash(&format!("{family}:{shape:?}"));
        if let Some(choice) = self
            .schedules
            .lock()
            .map_err(|_| "Apple GPU schedule cache poisoned")?
            .get(&key)
        {
            return Ok(*choice);
        }
        if candidates == 0 || rounds == 0 {
            return Err("Apple GPU race must have candidates and rounds".to_string());
        }
        let mut times = vec![Vec::with_capacity(rounds); candidates];
        let mut valid = vec![true; candidates];
        for round in 0..rounds {
            for offset in 0..candidates {
                let candidate = (round + offset) % candidates;
                if !valid[candidate] {
                    continue;
                }
                if let Some(elapsed) = evaluate(candidate)? {
                    times[candidate].push(elapsed);
                } else {
                    valid[candidate] = false;
                    times[candidate].clear();
                }
            }
        }
        let mut best = None;
        for (candidate, elapsed) in times.iter_mut().enumerate() {
            if !valid[candidate] || elapsed.is_empty() {
                continue;
            }
            elapsed.sort_unstable();
            let median = elapsed[elapsed.len() / 2];
            if best.is_none_or(|(_, current)| median < current) {
                best = Some((candidate, median));
            }
        }
        let choice = best
            .map(|(candidate, _)| candidate)
            .ok_or_else(|| format!("no valid Apple GPU race for {family} {shape:?}"))?;
        self.schedules
            .lock()
            .map_err(|_| "Apple GPU schedule cache poisoned")?
            .insert(key, choice);
        Ok(choice)
    }
}

fn source_hash(source: &str) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    std::hash::Hash::hash(&source, &mut hasher);
    std::hash::Hasher::finish(&hasher)
}

fn target_name(name: &str) -> Target {
    let compact: String = name
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    if compact.starts_with("AppleM1") {
        Target::G13
    } else if compact.starts_with("AppleM2") {
        Target::G14
    } else if compact.starts_with("AppleM3") {
        Target::G15
    } else if compact.starts_with("AppleM4") {
        Target::G16
    } else {
        Target::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn metal_unavailable(error: &str) -> bool {
        error.contains("no default Metal device found")
    }

    #[test]
    fn maps_generations() {
        assert_eq!(target_name("Apple M1 Max"), Target::G13);
        assert_eq!(target_name("Apple M2"), Target::G14);
        assert_eq!(target_name("Apple M3 Pro"), Target::G15);
        assert_eq!(target_name("Apple M4"), Target::G16);
        assert_eq!(target_name("Future GPU"), Target::Unknown);
    }

    #[test]
    fn shared_buffers() {
        let runtime = match Runtime::shared() {
            Ok(runtime) => runtime,
            Err(error) if metal_unavailable(&error) => return,
            Err(error) => panic!("{error}"),
        };
        let info = match device_info() {
            Ok(info) => info,
            Err(error) if metal_unavailable(&error) => return,
            Err(error) => panic!("{error}"),
        };
        assert_eq!(info, runtime.info().clone());
        let source = "kernel void copy_one(device uint *x [[buffer(0)]]) { x[0] = x[0]; }";
        let first = runtime.pipeline(source, "test", "copy_one").unwrap();
        let second = runtime.pipeline(source, "test", "copy_one").unwrap();
        assert_eq!(
            first.thread_execution_width(),
            second.thread_execution_width()
        );
        assert_eq!(source_hash(source), source_hash(source));
        let buf = runtime.buffer::<u32>(4);
        assert!(buf.length() >= 16);
        assert!(buffer_slice::<u32>(&buf, 0).unwrap().is_empty());
        assert!(slice_mut::<u32>(&buf, 0).unwrap().is_empty());
        let mutable = slice_mut::<u32>(&buf, 4).unwrap();
        mutable.copy_from_slice(&[10, 20, 30, 40]);
        assert_eq!(buffer_slice::<u32>(&buf, 4).unwrap(), &[10, 20, 30, 40]);
        let over = (buf.length() as usize / size_of::<u32>()) + 1;
        assert!(buffer_slice::<u32>(&buf, over).is_err());
        assert!(slice_mut::<u32>(&buf, over).is_err());
        assert!(runtime.buffer_with(&[1u32, 2, 3]).length() >= 12);
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        barrier_buffers(&encoder);
        encoder.end_encoding();
    }

    #[test]
    fn capture_inputs() {
        let runtime = match Runtime::new() {
            Ok(runtime) => runtime,
            Err(error) if metal_unavailable(&error) => return,
            Err(error) => panic!("{error}"),
        };
        let directory = tempfile::tempdir().unwrap();
        let source = "#include <metal_stdlib>\nusing namespace metal;\nconstant bool enabled \
                      [[function_constant(0)]];\nkernel void capture_probe(device uint *x \
                      [[buffer(0)]]) { x[0] = enabled ? 7u : 11u; }";
        runtime
            .compile_at(
                source,
                "capture",
                "capture_probe",
                true,
                None,
                &[(0, true)],
                None,
            )
            .unwrap();
        for _ in 0..2 {
            runtime
                .compile_at(
                    source,
                    "capture",
                    "capture_probe",
                    true,
                    None,
                    &[(0, true)],
                    Some(directory.path()),
                )
                .unwrap();
        }
        let records = baux::artifact::gpu::records(directory.path()).unwrap();
        assert_eq!(
            records.len(),
            1,
            "cache hits are captured once per destination"
        );
        assert_eq!(
            records[0].1.input.sha256,
            baux::artifact::runs::digest(source.as_bytes())
        );
        assert_eq!(
            records[0].1.recipe.output_kind,
            Some(baux::artifact::gpu::Kind::DeviceImage)
        );
        let output_blob = records[0]
            .1
            .output
            .as_ref()
            .expect("MTLBinaryArchive captures an executable Apple GPU image");
        let output_bytes = std::fs::read(
            directory
                .path()
                .join("blobs")
                .join(format!("{}.bin", output_blob.sha256)),
        )
        .unwrap();
        let metrics = archive_metrics(&output_bytes)
            .expect("captured MTLBinaryArchive contains Apple GPU hardware metadata");
        assert!(metrics.text_bytes > 0);
        assert!(metrics.gpr_count > 0);
        assert_eq!(metrics.spill_bytes, 0);
        assert_eq!(records[0].1.recipe.constants[0].bytes, [1]);
        runtime
            .compile_at(
                source,
                "capture",
                "capture_probe",
                true,
                None,
                &[(0, false)],
                Some(directory.path()),
            )
            .unwrap();
        runtime
            .compile_at(
                source,
                "capture",
                "capture_probe",
                false,
                None,
                &[(0, true)],
                Some(directory.path()),
            )
            .unwrap();
        assert_eq!(
            baux::artifact::gpu::records(directory.path())
                .unwrap()
                .len(),
            3
        );
        let second = tempfile::tempdir().unwrap();
        runtime
            .compile_at(
                source,
                "capture",
                "capture_probe",
                true,
                None,
                &[(0, true)],
                Some(second.path()),
            )
            .unwrap();
        assert_eq!(
            baux::artifact::gpu::records(second.path()).unwrap()[0].0.id,
            records[0].0.id
        );
        let invalid = tempfile::tempdir().unwrap();
        assert!(
            runtime
                .compile_at(
                    "not Metal source",
                    "capture",
                    "missing",
                    true,
                    None,
                    &[],
                    Some(invalid.path())
                )
                .is_err()
        );
        assert!(
            baux::artifact::gpu::records(invalid.path())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn zero_spill() {
        let runtime = match Runtime::new() {
            Ok(runtime) => runtime,
            Err(error) if metal_unavailable(&error) => return,
            Err(error) => panic!("{error}"),
        };
        let directory = tempfile::tempdir().unwrap();
        let pisa_source = include_str!("fbt_pisa1.metal");
        runtime
            .compile_at(
                pisa_source,
                "pisa1",
                "fbt_pisa1_select_attention_q4",
                false,
                None,
                &[],
                Some(directory.path()),
            )
            .unwrap();
        let moe_source = include_str!("fbt_moe.metal");
        for entry in [
            "fbt_mhc_mix_rms",
            "fbt_mhc_mean_rms",
            "fbt_moe_residual_rms",
            "fbt_moe_ungroup_residual_rms",
            "fbt_moe_feedback_fuse_rms",
        ] {
            runtime
                .compile_at(
                    moe_source,
                    "moe",
                    entry,
                    false,
                    None,
                    &[],
                    Some(directory.path()),
                )
                .unwrap();
        }
        let records = baux::artifact::gpu::records(directory.path()).unwrap();
        assert_eq!(records.len(), 6);
        for (_, record) in records {
            let blob = record.output.expect("device image output present");
            let bytes = std::fs::read(
                directory
                    .path()
                    .join("blobs")
                    .join(format!("{}.bin", blob.sha256)),
            )
            .unwrap();
            let metrics = archive_metrics(&bytes).expect("valid Apple GPU archive metrics");
            assert!(
                metrics.text_bytes > 0,
                "{} has empty machine code",
                record.recipe.entry
            );
            assert_eq!(
                metrics.spill_bytes, 0,
                "kernel {} spilled {} bytes/thread to stack",
                record.recipe.entry, metrics.spill_bytes
            );
        }
    }
}
