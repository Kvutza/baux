use super::{HazardCounts, KernelEvidence};
use std::collections::BTreeMap;

pub(super) fn u16_le(bytes: &[u8], offset: usize) -> Option<u16> {
    let slice = bytes.get(offset..offset.checked_add(2)?)?;
    Some(u16::from_le_bytes([slice[0], slice[1]]))
}

pub(super) fn u32_le(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice = bytes.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

pub(super) fn u64_le(bytes: &[u8], offset: usize) -> Option<u64> {
    let slice = bytes.get(offset..offset.checked_add(8)?)?;
    Some(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

pub(super) fn u32_be(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice = bytes.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn cstr(bytes: &[u8], offset: usize) -> Option<&str> {
    let tail = bytes.get(offset..)?;
    let end = tail.iter().position(|&b| b == 0)?;
    std::str::from_utf8(&tail[..end]).ok()
}

struct ElfLayout<'a> {
    bytes: &'a [u8],
    shoff: usize,
    shentsize: usize,
    shnum: usize,
    shstrtab: &'a [u8],
}

impl<'a> ElfLayout<'a> {
    fn shdr(&self, idx: usize) -> Option<(u32, u32, usize, usize, u32, u32)> {
        let base = self.shoff.checked_add(idx.checked_mul(self.shentsize)?)?;
        let sh_name = u32_le(self.bytes, base)?;
        let sh_type = u32_le(self.bytes, base + 4)?;
        let sh_offset = usize::try_from(u64_le(self.bytes, base + 24)?).ok()?;
        let sh_size = usize::try_from(u64_le(self.bytes, base + 32)?).ok()?;
        let sh_link = u32_le(self.bytes, base + 40)?;
        let sh_info = u32_le(self.bytes, base + 44)?;
        Some((sh_name, sh_type, sh_offset, sh_size, sh_link, sh_info))
    }
}

fn cubin_symtab(elf: &ElfLayout<'_>) -> BTreeMap<u32, String> {
    let mut sym_names = BTreeMap::new();
    for idx in 0..elf.shnum {
        let Some((sh_name, _, sh_offset, sh_size, sh_link, _)) = elf.shdr(idx) else {
            continue;
        };
        if cstr(elf.shstrtab, sh_name as usize) != Some(".symtab") {
            continue;
        }
        let Some((_, _, strtab_off, strtab_sz, _, _)) = elf.shdr(sh_link as usize) else {
            continue;
        };
        let Some(strtab) = elf
            .bytes
            .get(strtab_off..strtab_off.saturating_add(strtab_sz))
        else {
            continue;
        };
        for sym_i in 0..(sh_size / 24) {
            let entry_off = sh_offset.saturating_add(sym_i * 24);
            if let Some(st_name) = u32_le(elf.bytes, entry_off)
                && let Some(name) = cstr(strtab, st_name as usize)
                && !name.is_empty()
            {
                sym_names.insert(sym_i as u32, name.to_owned());
            }
        }
    }
    sym_names
}

fn cubin_frames(elf: &ElfLayout<'_>, sym_names: &BTreeMap<u32, String>) -> BTreeMap<String, u64> {
    let mut frame_bytes = BTreeMap::new();
    for idx in 0..elf.shnum {
        let Some((sh_name, _, sh_offset, sh_size, _, _)) = elf.shdr(idx) else {
            continue;
        };
        if cstr(elf.shstrtab, sh_name as usize) != Some(".nv.info") {
            continue;
        }
        let Some(blob) = elf.bytes.get(sh_offset..sh_offset.saturating_add(sh_size)) else {
            continue;
        };
        let mut pos = 0usize;
        while pos + 4 <= blob.len() {
            let fmt = blob[pos];
            let attr = blob[pos + 1];
            let sz = usize::from(u16_le(blob, pos + 2).unwrap_or(0));
            pos += 4;
            if fmt == 4 {
                if attr == 0x11
                    && sz == 8
                    && let (Some(func_idx), Some(frame_sz)) =
                        (u32_le(blob, pos), u32_le(blob, pos + 4))
                    && let Some(kname) = sym_names.get(&func_idx)
                {
                    frame_bytes.insert(kname.clone(), u64::from(frame_sz));
                }
                pos = pos.saturating_add(sz);
            }
        }
    }
    frame_bytes
}

fn empty_cubin(kname: &str, target: &str) -> KernelEvidence {
    KernelEvidence {
        name: kname.to_owned(),
        device_kind: "cuda".into(),
        target: target.to_owned(),
        layer: "device-binary".into(),
        code_bytes: 0,
        static_insns: 0,
        gpr_count: 0,
        uniform_regs: 0,
        shared_bytes: 0,
        frame_spill_bytes: 0,
        hazards: HazardCounts::default(),
    }
}

/// Inspect an NVIDIA ELF64 `.cubin` image (`sm_75`, `sm_80`, `sm_89`, `sm_90a`,
/// `sm_100`).
pub fn inspect_cubin(bytes: &[u8]) -> Result<Vec<KernelEvidence>, String> {
    if bytes.len() < 64 || &bytes[0..4] != b"\x7fELF" || bytes[4] != 2 || bytes[5] != 1 {
        return Err("not a little-endian ELF64 cubin image".into());
    }
    let e_flags = u32_le(bytes, 48).ok_or("truncated ELF64 e_flags")?;
    let sm_code = (e_flags >> 8) & 0xff;
    let target = if sm_code > 0 {
        format!("sm_{sm_code}")
    } else {
        format!("sm_raw_{e_flags:x}")
    };
    let shoff = usize::try_from(u64_le(bytes, 40).ok_or("truncated e_shoff")?)
        .map_err(|_| "e_shoff overflow")?;
    let shentsize = usize::from(u16_le(bytes, 58).ok_or("truncated e_shentsize")?);
    let shnum = usize::from(u16_le(bytes, 60).ok_or("truncated e_shnum")?);
    let shstrndx = usize::from(u16_le(bytes, 62).ok_or("truncated e_shstrndx")?);
    if shentsize < 64 || shstrndx >= shnum {
        return Err("invalid ELF64 section header geometry".into());
    }
    let str_base = shoff
        .checked_add(shstrndx.checked_mul(shentsize).ok_or("shstrndx overflow")?)
        .ok_or("shstrndx overflow")?;
    let str_off = usize::try_from(u64_le(bytes, str_base + 24).ok_or("truncated shstrtab")?)
        .map_err(|_| "shstrtab offset overflow")?;
    let str_sz = usize::try_from(u64_le(bytes, str_base + 32).ok_or("truncated shstrtab")?)
        .map_err(|_| "shstrtab size overflow")?;
    let shstrtab = bytes
        .get(str_off..str_off.checked_add(str_sz).ok_or("shstrtab overflow")?)
        .ok_or("truncated shstrtab")?;

    let elf = ElfLayout {
        bytes,
        shoff,
        shentsize,
        shnum,
        shstrtab,
    };
    let sym_names = cubin_symtab(&elf);
    let frame_bytes = cubin_frames(&elf, &sym_names);

    let mut map = BTreeMap::<String, KernelEvidence>::new();
    for idx in 0..shnum {
        let Some((sh_name, _, _, sh_size, _, sh_info)) = elf.shdr(idx) else {
            continue;
        };
        let Some(sec_name) = cstr(shstrtab, sh_name as usize) else {
            continue;
        };
        if let Some(kname) = sec_name.strip_prefix(".text.") {
            let entry = map
                .entry(kname.to_owned())
                .or_insert_with(|| empty_cubin(kname, &target));
            entry.code_bytes = sh_size as u64;
            entry.static_insns = (sh_size / 16) as u64;
            entry.gpr_count = u64::from((sh_info >> 24) & 0xff);
            if let Some(&fbytes) = frame_bytes.get(kname) {
                entry.frame_spill_bytes = fbytes;
            }
        } else if let Some(kname) = sec_name.strip_prefix(".nv.shared.") {
            let entry = map
                .entry(kname.to_owned())
                .or_insert_with(|| empty_cubin(kname, &target));
            entry.shared_bytes = sh_size as u64;
        }
    }
    Ok(map.into_values().collect())
}

type ArchiveSlices<'a> = ((u32, &'a [u8]), Option<&'a [u8]>);

fn archive_slices(bytes: &[u8]) -> Result<ArchiveSlices<'_>, String> {
    if bytes.len() < 8 || u32_be(bytes, 0) != Some(0xcbfe_babe) {
        return Err("not a universal Mach-O MTLBinaryArchive".into());
    }
    let nfat = usize::try_from(u32_be(bytes, 4).ok_or("truncated fat header")?)
        .map_err(|_| "nfat overflow")?;
    let mut gpu_slice: Option<(u32, &[u8])> = None;
    let mut air_slice: Option<&[u8]> = None;

    for i in 0..nfat {
        let arch_off = 8usize
            .checked_add(i.checked_mul(20).ok_or("fat overflow")?)
            .ok_or("fat overflow")?;
        let cputype = u32_be(bytes, arch_off).ok_or("truncated fat_arch")?;
        let cpusubtype = u32_be(bytes, arch_off + 4).ok_or("truncated fat_arch")?;
        let offset = usize::try_from(u32_be(bytes, arch_off + 8).ok_or("truncated fat_arch")?)
            .map_err(|_| "slice offset overflow")?;
        let size = usize::try_from(u32_be(bytes, arch_off + 12).ok_or("truncated fat_arch")?)
            .map_err(|_| "slice size overflow")?;
        let slice = bytes
            .get(offset..offset.checked_add(size).ok_or("slice overflow")?)
            .ok_or("truncated fat slice")?;
        if cputype == 0x0100_0013 {
            gpu_slice = Some((cpusubtype, slice));
        } else if cputype == 0x0100_0017 {
            air_slice = Some(slice);
        }
    }
    let gpu = gpu_slice.ok_or("no applegpu slice in MTLBinaryArchive")?;
    Ok((gpu, air_slice))
}

fn decode_flatbuffer(meta: &[u8], marker_pos: usize) -> Option<(u64, u64, u64, u64)> {
    let mut pos = marker_pos;
    while pos + 4 <= meta.len() {
        let soff = u32_le(meta, pos).unwrap_or(0) as i32 as isize;
        let vt = (pos as isize)
            .checked_sub(soff)
            .and_then(|v| usize::try_from(v).ok());
        if let Some(vt) = vt
            && vt + 4 <= meta.len()
            && vt.is_multiple_of(2)
            && let Some(vals) = probe_vtable(meta, pos, vt)
        {
            return Some(vals);
        }
        pos += 4;
    }
    None
}

fn probe_vtable(meta: &[u8], pos: usize, vt: usize) -> Option<(u64, u64, u64, u64)> {
    let vlen = usize::from(u16_le(meta, vt)?);
    let tlen = usize::from(u16_le(meta, vt + 2)?);
    if vlen < 64
        || !(40..=120).contains(&tlen)
        || vt.checked_add(vlen)? > meta.len()
        || pos.checked_add(tlen)? > meta.len()
        || pos < vt + vlen
    {
        return None;
    }
    let nfields = (vlen - 4) / 2;
    for field in 0..nfields {
        let fo = usize::from(u16_le(meta, vt + 4 + 2 * field).unwrap_or(0));
        if fo != 0 && (fo < 4 || fo >= tlen) {
            return None;
        }
    }
    if u16_le(meta, vt + 4).unwrap_or(0) == 0 {
        return None;
    }
    let read_field = |field: usize| -> u64 {
        if field >= nfields {
            return 0;
        }
        let fo = usize::from(u16_le(meta, vt + 4 + 2 * field).unwrap_or(0));
        if fo == 0 {
            0
        } else {
            u64::from(u16_le(meta, pos + fo).unwrap_or(0))
        }
    };
    Some((read_field(0), read_field(1), read_field(28), read_field(14)))
}

/// Inspect an Apple Silicon `MTLBinaryArchive` (`G13` M1 .. `G17` M5 + `air64`
/// LLVM bitcode).
pub fn inspect_archive(bytes: &[u8]) -> Result<Vec<KernelEvidence>, String> {
    let ((cpusubtype, outer), air_slice) = archive_slices(bytes)?;
    let fallback_target = match cpusubtype & 0xff {
        1..=3 | 0x60..=0x6f => "G13".into(),
        4 | 0x70..=0x7f => "G14".into(),
        5 | 0x80..=0x8f => "G15".into(),
        6 | 0x90..=0x9f => "G16".into(),
        7 | 0xa0..=0xaf => "G17".into(),
        other => format!("applegpu_sub_{other}"),
    };
    let deferred_alloca_ops = air_slice.map_or(0u64, |air| {
        let needle = b"deferred-static-alloca-size";
        air.windows(needle.len()).filter(|w| *w == needle).count() as u64
    });
    let entry_name = air_slice
        .and_then(mtlb_name)
        .or_else(|| mtlb_name(bytes))
        .unwrap_or_else(|| "agc.main".into());

    let inner = macho_section(outer, b"__TEXT", b"__compute")
        .ok_or("missing __TEXT,__compute in applegpu Mach-O")?;
    let text_sec = macho_section(inner, b"__TEXT", b"__text")
        .ok_or("missing __TEXT,__text in inner compute Mach-O")?;
    let meta = macho_section(inner, b"__GPU_METADATA", b"__compute")
        .ok_or("missing __GPU_METADATA,__compute in inner compute Mach-O")?;

    let marker = b"agc.main\0";
    let marker_pos = meta
        .windows(marker.len())
        .position(|w| w == marker)
        .ok_or("missing agc.main in __GPU_METADATA")?;
    let target = preceding_symbol(meta, marker_pos).unwrap_or(fallback_target);
    let (gpr_count, uniform_regs, shared_bytes, spill_bytes) = decode_flatbuffer(meta, marker_pos)
        .ok_or("unable to decode __GPU_METADATA FlatBuffer table")?;

    Ok(vec![KernelEvidence {
        name: entry_name,
        device_kind: "metal".into(),
        target,
        layer: "device-binary".into(),
        code_bytes: text_sec.len() as u64,
        static_insns: (text_sec.len() / 4) as u64,
        gpr_count,
        uniform_regs,
        shared_bytes,
        frame_spill_bytes: spill_bytes,
        hazards: HazardCounts {
            spill_ops: if spill_bytes > 0 {
                spill_bytes
            } else {
                deferred_alloca_ops
            },
            ..HazardCounts::default()
        },
    }])
}

fn mtlb_name(buf: &[u8]) -> Option<String> {
    let tag = b"NAME";
    for idx in 0..buf.len().saturating_sub(8) {
        if &buf[idx..idx + 4] == tag {
            let rest = &buf[idx + 6..buf.len().min(idx + 134)];
            let nul = rest.iter().position(|&b| b == 0)?;
            if nul > 0
                && rest[..nul]
                    .iter()
                    .all(|&b| b.is_ascii_alphanumeric() || b == b'_')
            {
                return std::str::from_utf8(&rest[..nul]).ok().map(str::to_owned);
            }
        }
    }
    None
}

fn preceding_symbol(meta: &[u8], marker_pos: usize) -> Option<String> {
    if marker_pos < 8 {
        return None;
    }
    let head = &meta[..marker_pos];
    let mut end = head.len();
    while end > 0 && head[end - 1] == 0 {
        end -= 1;
    }
    let mut start = end;
    while start > 0 && (head[start - 1].is_ascii_alphanumeric() || head[start - 1] == b'_') {
        start -= 1;
    }
    if end > start {
        std::str::from_utf8(&head[start..end])
            .ok()
            .map(str::to_owned)
    } else {
        None
    }
}

fn macho_section<'a>(macho: &'a [u8], seg: &[u8], sec: &[u8]) -> Option<&'a [u8]> {
    if macho.len() < 32 || u32_le(macho, 0)? != 0xfeed_facf {
        return None;
    }
    let ncmds = usize::try_from(u32_le(macho, 16)?).ok()?;
    let mut cmd_off = 32usize;
    for _ in 0..ncmds {
        let cmd = u32_le(macho, cmd_off)?;
        let cmdsize = usize::try_from(u32_le(macho, cmd_off.checked_add(4)?)?).ok()?;
        if cmdsize < 8 || cmd_off.checked_add(cmdsize)? > macho.len() {
            return None;
        }
        if cmd == 0x19 && cmdsize >= 72 {
            let nsects = usize::try_from(u32_le(macho, cmd_off + 64)?).ok()?;
            for s in 0..nsects {
                let s_off = cmd_off.checked_add(72)?.checked_add(s.checked_mul(80)?)?;
                if s_off.checked_add(80)? > cmd_off + cmdsize {
                    break;
                }
                let secname = &macho[s_off..s_off + 16];
                let segname = &macho[s_off + 16..s_off + 32];
                if prefix_eq(segname, seg) && prefix_eq(secname, sec) {
                    let offset = usize::try_from(u32_le(macho, s_off + 48)?).ok()?;
                    let size = usize::try_from(u64_le(macho, s_off + 40)?).ok()?;
                    return macho.get(offset..offset.checked_add(size)?);
                }
            }
        }
        cmd_off += cmdsize;
    }
    None
}

fn prefix_eq(fixed16: &[u8], name: &[u8]) -> bool {
    let len = fixed16
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(fixed16.len());
    &fixed16[..len] == name
}
