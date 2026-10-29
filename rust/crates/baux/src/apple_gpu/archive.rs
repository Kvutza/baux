use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use metal::objc::{
    __send_message as send_message,
    runtime::{BOOL, Class, Object, Sel, YES},
};
use metal::{Device, FunctionRef};

static NEXT_ARCHIVE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ArchiveMetrics {
    pub(crate) text_bytes: usize,
    pub(crate) gpr_count: u16,
    pub(crate) uniform_register_count: u16,
    pub(crate) spill_bytes: u16,
    pub(crate) threadgroup_memory_bytes: u16,
}

pub(crate) fn serialize_cached(
    device: &Device,
    device_name: &str,
    key: &(String, String),
    function: &FunctionRef,
    directory: &Path,
) -> Result<Vec<u8>, String> {
    let cache_id =
        baux::artifact::runs::digest(format!("v1:{device_name}:{}:{}", key.0, key.1).as_bytes());
    let cache_dir = Path::new(".cache/baux/metal-archives");
    let cache_path = cache_dir.join(format!("{cache_id}.bin"));
    if let Ok(bytes) = std::fs::read(&cache_path)
        && !bytes.is_empty()
    {
        return Ok(bytes);
    }
    let bytes = serialize_archive(device, function, directory)?;
    if std::fs::create_dir_all(cache_dir).is_ok() {
        let tmp = cache_dir.join(format!(
            ".{cache_id}-{}-{}.tmp",
            std::process::id(),
            NEXT_ARCHIVE.fetch_add(1, Ordering::Relaxed)
        ));
        if std::fs::write(&tmp, &bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &cache_path);
        }
    }
    Ok(bytes)
}

pub(crate) fn serialize_archive(
    device: &Device,
    function: &FunctionRef,
    directory: &Path,
) -> Result<Vec<u8>, String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("create {}: {error}", directory.display()))?;
    let temporary = directory.join(format!(
        ".metal-archive-{}-{}.tmp",
        std::process::id(),
        NEXT_ARCHIVE.fetch_add(1, Ordering::Relaxed)
    ));
    let path_str = temporary
        .to_str()
        .ok_or("Metal archive temporary path is not valid UTF-8")?;
    let result = unsafe { write_archive(device, function, path_str) };
    let read_result = result.and_then(|()| {
        std::fs::read(&temporary).map_err(|error| format!("read {}: {error}", temporary.display()))
    });
    let _ = std::fs::remove_file(&temporary);
    read_result
}

unsafe fn write_archive(
    device: &Device,
    function: &FunctionRef,
    path_str: &str,
) -> Result<(), String> {
    let release = Sel::register("release");
    let archive = unsafe { create_archive(device, release)? };
    if let Err(error) = unsafe { add_function(archive, function, release) } {
        let _ = unsafe { send_message::<Object, (), ()>(archive, release, ()) };
        return Err(error);
    }
    let save_res = unsafe { save_url(archive, path_str, release) };
    let _ = unsafe { send_message::<Object, (), ()>(archive, release, ()) };
    save_res
}

unsafe fn create_archive(device: &Device, release: Sel) -> Result<*mut Object, String> {
    let arch_desc_cls =
        Class::get("MTLBinaryArchiveDescriptor").ok_or("MTLBinaryArchiveDescriptor unavailable")?;
    let arch_desc =
        unsafe { send_message::<Class, (), *mut Object>(arch_desc_cls, Sel::register("new"), ()) }
            .map_err(|error| format!("create MTLBinaryArchiveDescriptor: {error:?}"))?;
    if arch_desc.is_null() {
        return Err("MTLBinaryArchiveDescriptor returned nil".into());
    }
    let mut err: *mut Object = std::ptr::null_mut();
    let archive = unsafe {
        send_message::<Object, (*mut Object, *mut *mut Object), *mut Object>(
            device.as_ref() as *const _ as *const Object,
            Sel::register("newBinaryArchiveWithDescriptor:error:"),
            (arch_desc, &mut err),
        )
    };
    let _ = unsafe { send_message::<Object, (), ()>(arch_desc, release, ()) };
    let archive = archive.map_err(|error| format!("newBinaryArchive: {error:?}"))?;
    if archive.is_null() {
        return Err("MTLDevice newBinaryArchiveWithDescriptor returned nil".into());
    }
    Ok(archive)
}

unsafe fn add_function(
    archive: *mut Object,
    function: &FunctionRef,
    release: Sel,
) -> Result<(), String> {
    let pipe_desc_cls = Class::get("MTLComputePipelineDescriptor")
        .ok_or("MTLComputePipelineDescriptor unavailable")?;
    let pipe_desc =
        unsafe { send_message::<Class, (), *mut Object>(pipe_desc_cls, Sel::register("new"), ()) };
    let pipe_desc = match pipe_desc {
        Ok(ptr) if !ptr.is_null() => ptr,
        _ => return Err("MTLComputePipelineDescriptor returned nil".into()),
    };
    let _ = unsafe {
        send_message::<Object, (*const Object,), ()>(
            pipe_desc,
            Sel::register("setComputeFunction:"),
            (function as *const FunctionRef as *const Object,),
        )
    };
    let mut err: *mut Object = std::ptr::null_mut();
    let added = unsafe {
        send_message::<Object, (*mut Object, *mut *mut Object), BOOL>(
            archive,
            Sel::register("addComputePipelineFunctionsWithDescriptor:error:"),
            (pipe_desc, &mut err),
        )
    };
    let _ = unsafe { send_message::<Object, (), ()>(pipe_desc, release, ()) };
    if !matches!(added, Ok(YES)) {
        return Err("MTLBinaryArchive addComputePipelineFunctionsWithDescriptor failed".into());
    }
    Ok(())
}

unsafe fn save_url(archive: *mut Object, path_str: &str, release: Sel) -> Result<(), String> {
    let str_cls = Class::get("NSString").ok_or("NSString unavailable")?;
    let ns_str_alloc =
        unsafe { send_message::<Class, (), *mut Object>(str_cls, Sel::register("alloc"), ()) }
            .map_err(|error| format!("NSString alloc: {error:?}"))?;
    let ns_str = unsafe {
        send_message::<Object, (*const u8, usize, usize), *mut Object>(
            ns_str_alloc,
            Sel::register("initWithBytes:length:encoding:"),
            (path_str.as_ptr(), path_str.len(), 4usize),
        )
    }
    .map_err(|error| format!("NSString initWithBytes: {error:?}"))?;
    let url_cls = Class::get("NSURL").ok_or("NSURL unavailable")?;
    let ns_url_alloc =
        unsafe { send_message::<Class, (), *mut Object>(url_cls, Sel::register("alloc"), ()) }
            .map_err(|error| format!("NSURL alloc: {error:?}"))?;
    let ns_url = unsafe {
        send_message::<Object, (*mut Object,), *mut Object>(
            ns_url_alloc,
            Sel::register("initFileURLWithPath:"),
            (ns_str,),
        )
    }
    .map_err(|error| format!("NSURL initFileURLWithPath: {error:?}"))?;
    let mut err: *mut Object = std::ptr::null_mut();
    let serialized = unsafe {
        send_message::<Object, (*mut Object, *mut *mut Object), BOOL>(
            archive,
            Sel::register("serializeToURL:error:"),
            (ns_url, &mut err),
        )
    };
    let _ = unsafe { send_message::<Object, (), ()>(ns_url, release, ()) };
    let _ = unsafe { send_message::<Object, (), ()>(ns_str, release, ()) };
    if !matches!(serialized, Ok(YES)) {
        return Err("MTLBinaryArchive serializeToURL failed".into());
    }
    Ok(())
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

fn read_i32(buf: &[u8], off: usize) -> Option<i32> {
    Some(i32::from_le_bytes(
        buf.get(off..off.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn read_u64(buf: &[u8], off: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        buf.get(off..off.checked_add(8)?)?.try_into().ok()?,
    ))
}

pub(crate) fn archive_metrics(bytes: &[u8]) -> Option<ArchiveMetrics> {
    let outer = gpu_slice(bytes)?;
    let inner = compute_slice(outer)?;
    let (text_bytes, metadata) = inner_sections(inner)?;
    scan_metadata(metadata, text_bytes)
}

fn gpu_slice(bytes: &[u8]) -> Option<&[u8]> {
    let magic = be_u32(bytes, 0)?;
    if magic != 0xcbfe_babe && magic != 0xcafe_babe {
        return None;
    }
    let narch = be_u32(bytes, 4)? as usize;
    for index in 0..narch {
        let entry = 8usize.checked_add(index.checked_mul(20)?)?;
        let cputype = be_u32(bytes, entry)?;
        let offset = be_u32(bytes, entry + 8)? as usize;
        let size = be_u32(bytes, entry + 12)? as usize;
        if cputype == 0x0100_0013 {
            return bytes.get(offset..offset.checked_add(size)?);
        }
    }
    None
}

fn compute_slice(outer: &[u8]) -> Option<&[u8]> {
    if read_u32(outer, 0)? != 0xfeed_facf {
        return None;
    }
    let ncmds = read_u32(outer, 16)? as usize;
    let mut cmd_off = 32usize;
    for _ in 0..ncmds {
        let cmd = read_u32(outer, cmd_off)?;
        let cmdsize = read_u32(outer, cmd_off + 4)? as usize;
        if cmdsize < 8 {
            return None;
        }
        if cmd == 0x19 {
            if let Some(slice) = find_compute(outer, cmd_off)? {
                return Some(slice);
            }
        }
        cmd_off = cmd_off.checked_add(cmdsize)?;
    }
    None
}

fn find_compute(outer: &[u8], cmd_off: usize) -> Option<Option<&[u8]>> {
    let nsects = read_u32(outer, cmd_off + 64)? as usize;
    for sect in 0..nsects {
        let so = cmd_off
            .checked_add(72)?
            .checked_add(sect.checked_mul(80)?)?;
        let sectname = outer.get(so..so + 16)?;
        let segname = outer.get(so + 16..so + 32)?;
        if segname.starts_with(b"__TEXT\0") && sectname.starts_with(b"__compute\0") {
            let size = read_u64(outer, so + 40)? as usize;
            let offset = read_u32(outer, so + 48)? as usize;
            return Some(outer.get(offset..offset.checked_add(size)?));
        }
    }
    Some(None)
}

fn inner_sections(inner: &[u8]) -> Option<(usize, &[u8])> {
    if read_u32(inner, 0)? != 0xfeed_facf {
        return None;
    }
    let inner_cmds = read_u32(inner, 16)? as usize;
    let mut icmd_off = 32usize;
    let mut text_bytes = 0usize;
    let mut metadata: Option<&[u8]> = None;
    for _ in 0..inner_cmds {
        let cmd = read_u32(inner, icmd_off)?;
        let cmdsize = read_u32(inner, icmd_off + 4)? as usize;
        if cmdsize < 8 {
            return None;
        }
        if cmd == 0x19 {
            scan_segment(inner, icmd_off, &mut text_bytes, &mut metadata)?;
        }
        icmd_off = icmd_off.checked_add(cmdsize)?;
    }
    Some((text_bytes, metadata?))
}

fn scan_segment<'a>(
    inner: &'a [u8],
    icmd_off: usize,
    text_bytes: &mut usize,
    metadata: &mut Option<&'a [u8]>,
) -> Option<()> {
    let nsects = read_u32(inner, icmd_off + 64)? as usize;
    for sect in 0..nsects {
        let so = icmd_off
            .checked_add(72)?
            .checked_add(sect.checked_mul(80)?)?;
        let sectname = inner.get(so..so + 16)?;
        let segname = inner.get(so + 16..so + 32)?;
        let size = read_u64(inner, so + 40)? as usize;
        let offset = read_u32(inner, so + 48)? as usize;
        if segname.starts_with(b"__TEXT\0") && sectname.starts_with(b"__text\0") {
            *text_bytes = size;
        } else if segname.starts_with(b"__GPU_METADATA") && sectname.starts_with(b"__compute\0") {
            *metadata = inner.get(offset..offset.checked_add(size)?);
        }
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
    let soff = read_i32(buf, pos)? as isize;
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
