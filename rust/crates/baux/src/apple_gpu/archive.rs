use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use metal::foreign_types::ForeignType;
use metal::objc::{
    __send_message as send_message,
    rc::autoreleasepool,
    runtime::{BOOL, Class, Object, Sel, YES},
};
use metal::{ComputePipelineState, Device, FunctionRef};

static NEXT_ARCHIVE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveMetrics {
    pub text_bytes: usize,
    pub gpr_count: u16,
    pub uniform_register_count: u16,
    pub spill_bytes: u16,
    pub threadgroup_memory_bytes: u16,
}

pub fn archive_key(device_name: &str, source_digest: &str, specs: &[&str]) -> String {
    baux::artifact::runs::digest(
        format!("v1:{device_name}:{source_digest}:{}", specs.join(";")).as_bytes(),
    )
}

fn temp_path(directory: &Path, prefix: &str) -> std::path::PathBuf {
    directory.join(format!(
        ".{prefix}-{}-{}.tmp",
        std::process::id(),
        NEXT_ARCHIVE.fetch_add(1, Ordering::Relaxed)
    ))
}

pub fn load_pipelines(
    device: &Device,
    cache_id: &str,
    functions: &[&FunctionRef],
) -> Result<Vec<ComputePipelineState>, String> {
    if functions.is_empty() {
        return Ok(Vec::new());
    }
    let cache_dir = Path::new(".cache/baux/metal-archives");
    let cache_path = cache_dir.join(format!("{cache_id}.bin"));
    if let Ok(pipelines) = autoreleasepool(|| {
        if cache_path
            .metadata()
            .is_ok_and(|m| m.is_file() && m.len() > 0)
            && let Some(path_str) = cache_path.to_str()
            && let Ok(warm) = unsafe { build_archived(device, Some(path_str), None, functions) }
        {
            return Ok(warm);
        }
        let tmp = std::fs::create_dir_all(cache_dir)
            .ok()
            .map(|()| temp_path(cache_dir, cache_id));
        let cold = unsafe { build_archived(device, None, tmp.as_deref(), functions) };
        if let Some(tmp_path) = &tmp {
            if cold.is_ok() && tmp_path.exists() {
                let _ = std::fs::rename(tmp_path, &cache_path);
            }
            let _ = std::fs::remove_file(tmp_path);
        }
        cold
    }) {
        return Ok(pipelines);
    }
    functions
        .iter()
        .map(|f| {
            device
                .new_compute_pipeline_state_with_function(f)
                .map_err(|error| format!("Metal pipeline: {error}"))
        })
        .collect()
}

pub fn serialize_cached(
    device: &Device,
    device_name: &str,
    key: &(String, String),
    function: &FunctionRef,
    directory: &Path,
) -> Result<Vec<u8>, String> {
    let cache_id = archive_key(device_name, &key.0, &[&key.1]);
    let cache_dir = Path::new(".cache/baux/metal-archives");
    let cache_path = cache_dir.join(format!("{cache_id}.bin"));
    if let Ok(bytes) = std::fs::read(&cache_path)
        && !bytes.is_empty()
    {
        return Ok(bytes);
    }
    let bytes = serialize_archive(device, function, directory)?;
    if std::fs::create_dir_all(cache_dir).is_ok() {
        let tmp = temp_path(cache_dir, &cache_id);
        if std::fs::write(&tmp, &bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &cache_path);
            let _ = std::fs::remove_file(&tmp);
        }
    }
    Ok(bytes)
}

pub fn serialize_archive(
    device: &Device,
    function: &FunctionRef,
    directory: &Path,
) -> Result<Vec<u8>, String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("create {}: {error}", directory.display()))?;
    let temporary = temp_path(directory, "metal-archive");
    let path_str = temporary
        .to_str()
        .ok_or("Metal archive temporary path is not valid UTF-8")?;
    let result = autoreleasepool(|| unsafe { write_archive(device, function, path_str) });
    let read_result = result.and_then(|()| {
        std::fs::read(&temporary).map_err(|error| format!("read {}: {error}", temporary.display()))
    });
    let _ = std::fs::remove_file(&temporary);
    read_result
}

unsafe fn release_obj(object: *mut Object) {
    let _ = unsafe { send_message::<Object, (), ()>(object, Sel::register("release"), ()) };
}

unsafe fn pipe_descriptor(function: &FunctionRef) -> Result<*mut Object, String> {
    let cls = Class::get("MTLComputePipelineDescriptor")
        .ok_or("MTLComputePipelineDescriptor unavailable")?;
    let desc = unsafe { send_message::<Class, (), *mut Object>(cls, Sel::register("new"), ()) }
        .ok()
        .filter(|ptr| !ptr.is_null())
        .ok_or_else(|| "MTLComputePipelineDescriptor returned nil".to_string())?;
    let _ = unsafe {
        send_message::<Object, (*const Object,), ()>(
            desc,
            Sel::register("setComputeFunction:"),
            (function as *const FunctionRef as *const Object,),
        )
    };
    Ok(desc)
}

unsafe fn build_archived(
    device: &Device,
    open_path: Option<&str>,
    save_tmp: Option<&Path>,
    functions: &[&FunctionRef],
) -> Result<Vec<ComputePipelineState>, String> {
    let archive = unsafe { create_archive(device, open_path)? };
    let result = (|| unsafe {
        if open_path.is_none() {
            for function in functions {
                add_function(archive, function)?;
            }
            if let Some(tmp_str) = save_tmp.and_then(Path::to_str) {
                let _ = save_url(archive, tmp_str);
            }
        }
        functions
            .iter()
            .map(|function| archive_pipeline(device, archive, function))
            .collect()
    })();
    unsafe { release_obj(archive) };
    result
}

unsafe fn write_archive(
    device: &Device,
    function: &FunctionRef,
    path_str: &str,
) -> Result<(), String> {
    let archive = unsafe { create_archive(device, None)? };
    let result =
        unsafe { add_function(archive, function).and_then(|()| save_url(archive, path_str)) };
    unsafe { release_obj(archive) };
    result
}

unsafe fn create_archive(device: &Device, url_path: Option<&str>) -> Result<*mut Object, String> {
    let cls =
        Class::get("MTLBinaryArchiveDescriptor").ok_or("MTLBinaryArchiveDescriptor unavailable")?;
    let desc = unsafe { send_message::<Class, (), *mut Object>(cls, Sel::register("new"), ()) }
        .ok()
        .filter(|ptr| !ptr.is_null())
        .ok_or_else(|| "MTLBinaryArchiveDescriptor returned nil".to_string())?;
    if let Some(path_str) = url_path {
        let ns_url = unsafe { file_url(path_str) }.inspect_err(|_| unsafe { release_obj(desc) })?;
        let _ = unsafe {
            send_message::<Object, (*mut Object,), ()>(desc, Sel::register("setUrl:"), (ns_url,))
        };
        unsafe { release_obj(ns_url) };
    }
    let mut error_obj: *mut Object = std::ptr::null_mut();
    let archive = unsafe {
        send_message::<Object, (*mut Object, *mut *mut Object), *mut Object>(
            device.as_ref() as *const _ as *const Object,
            Sel::register("newBinaryArchiveWithDescriptor:error:"),
            (desc, &mut error_obj),
        )
    };
    unsafe { release_obj(desc) };
    archive
        .ok()
        .filter(|ptr| !ptr.is_null())
        .ok_or_else(|| "MTLDevice newBinaryArchiveWithDescriptor failed".into())
}

unsafe fn add_function(archive: *mut Object, function: &FunctionRef) -> Result<(), String> {
    let desc = unsafe { pipe_descriptor(function)? };
    let mut error_obj: *mut Object = std::ptr::null_mut();
    let added = unsafe {
        send_message::<Object, (*mut Object, *mut *mut Object), BOOL>(
            archive,
            Sel::register("addComputePipelineFunctionsWithDescriptor:error:"),
            (desc, &mut error_obj),
        )
    };
    unsafe { release_obj(desc) };
    if matches!(added, Ok(YES)) {
        Ok(())
    } else {
        Err("MTLBinaryArchive addComputePipelineFunctionsWithDescriptor failed".into())
    }
}

unsafe fn archive_pipeline(
    device: &Device,
    archive: *mut Object,
    function: &FunctionRef,
) -> Result<ComputePipelineState, String> {
    let desc = unsafe { pipe_descriptor(function)? };
    if let Some(array_cls) = Class::get("NSArray")
        && let Ok(alloc) =
            unsafe { send_message::<Class, (), *mut Object>(array_cls, Sel::register("alloc"), ()) }
        && !alloc.is_null()
    {
        let objects: [*mut Object; 1] = [archive];
        if let Ok(archives) = unsafe {
            send_message::<Object, (*const *mut Object, usize), *mut Object>(
                alloc,
                Sel::register("initWithObjects:count:"),
                (objects.as_ptr(), 1usize),
            )
        } && !archives.is_null()
        {
            let _ = unsafe {
                send_message::<Object, (*mut Object,), ()>(
                    desc,
                    Sel::register("setBinaryArchives:"),
                    (archives,),
                )
            };
            unsafe { release_obj(archives) };
        }
    }
    let mut error_obj: *mut Object = std::ptr::null_mut();
    let pipeline = unsafe {
        send_message::<Object, (*mut Object, u64, *mut *mut Object, *mut *mut Object), *mut Object>(
            device.as_ref() as *const _ as *const Object,
            Sel::register("newComputePipelineStateWithDescriptor:options:reflection:error:"),
            (desc, 0u64, std::ptr::null_mut(), &mut error_obj),
        )
    };
    unsafe { release_obj(desc) };
    let ptr = pipeline
        .ok()
        .filter(|ptr| !ptr.is_null())
        .ok_or_else(|| "newComputePipelineStateWithDescriptor failed".to_string())?;
    Ok(unsafe { ComputePipelineState::from_ptr(ptr.cast()) })
}

unsafe fn file_url(path_str: &str) -> Result<*mut Object, String> {
    let absolute;
    let resolved = if Path::new(path_str).is_relative()
        && let Ok(cwd) = std::env::current_dir()
        && let Some(full) = cwd.join(path_str).to_str().map(str::to_owned)
    {
        absolute = full;
        absolute.as_str()
    } else {
        path_str
    };
    let str_cls = Class::get("NSString").ok_or("NSString unavailable")?;
    let str_alloc =
        unsafe { send_message::<Class, (), *mut Object>(str_cls, Sel::register("alloc"), ()) }
            .map_err(|error| format!("NSString alloc: {error:?}"))?;
    let ns_str = unsafe {
        send_message::<Object, (*const u8, usize, usize), *mut Object>(
            str_alloc,
            Sel::register("initWithBytes:length:encoding:"),
            (resolved.as_ptr(), resolved.len(), 4usize),
        )
    }
    .map_err(|error| format!("NSString initWithBytes: {error:?}"))?;
    let url_cls = Class::get("NSURL").ok_or("NSURL unavailable")?;
    let url_alloc =
        unsafe { send_message::<Class, (), *mut Object>(url_cls, Sel::register("alloc"), ()) }
            .map_err(|error| format!("NSURL alloc: {error:?}"))?;
    let ns_url = unsafe {
        send_message::<Object, (*mut Object,), *mut Object>(
            url_alloc,
            Sel::register("initFileURLWithPath:"),
            (ns_str,),
        )
    };
    unsafe { release_obj(ns_str) };
    ns_url
        .ok()
        .filter(|ptr| !ptr.is_null())
        .ok_or_else(|| "NSURL initFileURLWithPath failed".into())
}

unsafe fn save_url(archive: *mut Object, path_str: &str) -> Result<(), String> {
    let ns_url = unsafe { file_url(path_str)? };
    let mut error_obj: *mut Object = std::ptr::null_mut();
    let serialized = unsafe {
        send_message::<Object, (*mut Object, *mut *mut Object), BOOL>(
            archive,
            Sel::register("serializeToURL:error:"),
            (ns_url, &mut error_obj),
        )
    };
    unsafe { release_obj(ns_url) };
    if matches!(serialized, Ok(YES)) {
        Ok(())
    } else {
        Err("MTLBinaryArchive serializeToURL failed".into())
    }
}

fn be_u32(buf: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_be_bytes(
        buf.get(off..off.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn read_u16(buf: &[u8], off: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        buf.get(off..off.checked_add(2)?)?.try_into().ok()?,
    ))
}

fn read_u32(buf: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        buf.get(off..off.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn read_u64(buf: &[u8], off: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        buf.get(off..off.checked_add(8)?)?.try_into().ok()?,
    ))
}

pub fn kernel_metrics(
    device: &Device,
    device_name: &str,
    key: &(String, String),
    function: &FunctionRef,
) -> Result<ArchiveMetrics, String> {
    let bytes = serialize_cached(device, device_name, key, function, &std::env::temp_dir())?;
    archive_metrics(&bytes).ok_or_else(|| "missing Apple GPU archive metadata".into())
}

pub fn archive_metrics(bytes: &[u8]) -> Option<ArchiveMetrics> {
    let mut compute = None;
    visit_sections(gpu_slice(bytes)?, |seg, sect, slice| {
        if seg.starts_with(b"__TEXT\0") && sect.starts_with(b"__compute\0") && compute.is_none() {
            compute = Some(slice);
        }
    })?;
    let mut text_bytes = 0usize;
    let mut metadata = None;
    visit_sections(compute?, |seg, sect, slice| {
        if seg.starts_with(b"__TEXT\0") && sect.starts_with(b"__text\0") {
            text_bytes = slice.len();
        } else if seg.starts_with(b"__GPU_METADATA") && sect.starts_with(b"__compute\0") {
            metadata = Some(slice);
        }
    })?;
    scan_metadata(metadata?, text_bytes)
}

fn gpu_slice(bytes: &[u8]) -> Option<&[u8]> {
    let magic = be_u32(bytes, 0)?;
    if magic != 0xcbfe_babe && magic != 0xcafe_babe {
        return None;
    }
    for index in 0..be_u32(bytes, 4)? as usize {
        let entry = 8usize.checked_add(index.checked_mul(20)?)?;
        if be_u32(bytes, entry)? == 0x0100_0013 {
            let offset = be_u32(bytes, entry + 8)? as usize;
            let size = be_u32(bytes, entry + 12)? as usize;
            return bytes.get(offset..offset.checked_add(size)?);
        }
    }
    None
}

fn visit_sections<'a>(
    mach: &'a [u8],
    mut f: impl FnMut(&'a [u8], &'a [u8], &'a [u8]),
) -> Option<()> {
    if read_u32(mach, 0)? != 0xfeed_facf {
        return None;
    }
    let mut cmd_off = 32usize;
    for _ in 0..read_u32(mach, 16)? as usize {
        let cmdsize = read_u32(mach, cmd_off + 4)? as usize;
        if cmdsize < 8 {
            return None;
        }
        if read_u32(mach, cmd_off)? == 0x19 {
            for sect in 0..read_u32(mach, cmd_off + 64)? as usize {
                let so = cmd_off
                    .checked_add(72)?
                    .checked_add(sect.checked_mul(80)?)?;
                let size = read_u64(mach, so + 40)? as usize;
                let offset = read_u32(mach, so + 48)? as usize;
                f(
                    mach.get(so + 16..so + 32)?,
                    mach.get(so..so + 16)?,
                    mach.get(offset..offset.checked_add(size)?)?,
                );
            }
        }
        cmd_off = cmd_off.checked_add(cmdsize)?;
    }
    Some(())
}

fn scan_metadata(buf: &[u8], text_bytes: usize) -> Option<ArchiveMetrics> {
    let needle = b"agc.main\0";
    let mut pos = buf
        .windows(needle.len())
        .position(|window| window == needle)?;
    while pos + 4 <= buf.len() {
        if let Some(metrics) = vtable_metrics(buf, pos, text_bytes)? {
            return Some(metrics);
        }
        pos += 4;
    }
    None
}

fn vtable_metrics(buf: &[u8], pos: usize, text_bytes: usize) -> Option<Option<ArchiveMetrics>> {
    let soff = read_u32(buf, pos)? as i32 as isize;
    let vt_isize = (pos as isize).checked_sub(soff)?;
    if vt_isize < 0 || (vt_isize as usize) + 4 > buf.len() || !(vt_isize as usize).is_multiple_of(2)
    {
        return Some(None);
    }
    let vt = vt_isize as usize;
    let vlen = read_u16(buf, vt)? as usize;
    let tlen = read_u16(buf, vt + 2)? as usize;
    if vlen < 64
        || !(40..=120).contains(&tlen)
        || vt.checked_add(vlen)? > buf.len()
        || pos.checked_add(tlen)? > buf.len()
        || pos < vt + vlen
    {
        return Some(None);
    }
    let nfields = (vlen - 4) / 2;
    for field in 0..nfields {
        let fo = read_u16(buf, vt + 4 + 2 * field)? as usize;
        if fo != 0 && (fo < 4 || fo >= tlen) {
            return Some(None);
        }
    }
    let fo0 = read_u16(buf, vt + 4)? as usize;
    if fo0 == 0 {
        return Some(None);
    }
    Some(Some(ArchiveMetrics {
        text_bytes,
        gpr_count: field_val(buf, vt, pos, nfields, 0),
        uniform_register_count: field_val(buf, vt, pos, nfields, 1),
        spill_bytes: field_val(buf, vt, pos, nfields, 14),
        threadgroup_memory_bytes: field_val(buf, vt, pos, nfields, 28),
    }))
}

fn field_val(buf: &[u8], vt: usize, pos: usize, nfields: usize, field: usize) -> u16 {
    if field >= nfields {
        return 0;
    }
    let fo = read_u16(buf, vt + 4 + 2 * field).unwrap_or(0) as usize;
    if fo == 0 {
        0
    } else {
        read_u16(buf, pos + fo).unwrap_or(0)
    }
}
