use super::{HazardCounts, KernelEvidence};

fn ptx_hazard(t: &str, hazards: &mut HazardCounts) {
    if t.contains("ld.local") || t.contains("st.local") {
        hazards.spill_ops += 1;
    }
    if t.contains(".f64") {
        hazards.fp64_ops += 1;
    }
    if t.starts_with("div.u32")
        || t.starts_with("rem.u32")
        || t.starts_with("div.s32")
        || t.starts_with("rem.s32")
    {
        hazards.rcp_div_ops += 1;
    }
    if t.starts_with("trap;") || t.contains(" trap;") {
        hazards.trap_ops += 1;
    }
    if t.contains("mma.sync") || t.contains("wgmma.mma_async") {
        hazards.tensor_ops += 1;
    }
    if t.contains("cp.async") {
        hazards.async_copy_ops += 1;
    }
}

/// Inspect PTX virtual ISA across any target (`sm_75`, `sm_80`, `sm_89`,
/// `sm_90a`, `sm_100`).
pub fn inspect_ptx(text: &str) -> Vec<KernelEvidence> {
    let mut target = "ptx".to_owned();
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(".target ") {
            target = rest.split_whitespace().next().unwrap_or("ptx").to_owned();
            break;
        }
    }
    let mut out = Vec::new();
    let mut parts = text.split(".entry ");
    let _header = parts.next();
    for part in parts {
        let name = part
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .find(|s| !s.is_empty())
            .unwrap_or("unknown")
            .to_owned();
        let body = part.split("\n}\n").next().unwrap_or(part);
        let mut insns = 0u64;
        let mut gpr = 0u64;
        let mut shared_bytes = 0u64;
        let mut local_bytes = 0u64;
        let mut hazards = HazardCounts::default();

        for line in body.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with("//") {
                continue;
            }
            if t.starts_with(".reg ") {
                if let Some(lt) = t.find('<')
                    && let Some(gt) = t.find('>')
                    && let Ok(n) = t[lt + 1..gt].parse::<u64>()
                {
                    gpr += n;
                }
                continue;
            }
            if t.starts_with(".shared ") {
                shared_bytes += ptx_bytes(t);
                continue;
            }
            if t.starts_with(".local ") {
                local_bytes += ptx_bytes(t);
                continue;
            }
            if t.ends_with(';') && !t.starts_with('.') {
                insns += 1;
                ptx_hazard(t, &mut hazards);
            }
        }
        out.push(KernelEvidence {
            name,
            backend: "cuda".into(),
            target: target.clone(),
            layer: "virtual-isa".into(),
            code_bytes: body.len() as u64,
            static_insns: insns,
            gpr_count: gpr,
            uniform_regs: 0,
            shared_bytes,
            frame_spill_bytes: local_bytes,
            hazards,
        });
    }
    out
}

fn ptx_bytes(line: &str) -> u64 {
    let elem = if line.contains(".b8") || line.contains(".u8") || line.contains(".s8") {
        1u64
    } else if line.contains(".b16") || line.contains(".u16") || line.contains(".f16") {
        2
    } else if line.contains(".b64") || line.contains(".u64") || line.contains(".f64") {
        8
    } else {
        4
    };
    if let Some(lb) = line.find('[')
        && let Some(rb) = line.find(']')
        && let Ok(count) = line[lb + 1..rb].parse::<u64>()
    {
        elem * count
    } else {
        elem
    }
}

/// Inspect disassembled NVIDIA SASS (`.sass`) across any SM architecture.
pub fn inspect_sass(text: &str) -> Vec<KernelEvidence> {
    let mut out = Vec::new();
    let mut parts = text.split(".text.");
    let _header = parts.next();
    for part in parts {
        let Some(colon) = part.find(':') else {
            continue;
        };
        let name = part[..colon].trim().to_owned();
        if name.is_empty() || name.contains(char::is_whitespace) {
            continue;
        }
        let body = &part[colon + 1..];
        let mut insns = 0u64;
        let mut hazards = HazardCounts::default();

        for line in body.lines() {
            let trimmed = line.trim();
            if !(trimmed.starts_with("/*") && trimmed.ends_with(';')) {
                continue;
            }
            let Some(end_comment) = trimmed.find("*/") else {
                continue;
            };
            let inst = trimmed[end_comment + 2..trimmed.len() - 1].trim();
            insns += 1;
            if has_word(inst, "LDL") || has_word(inst, "STL") {
                hazards.spill_ops += 1;
            }
            if has_word(inst, "DMUL")
                || has_word(inst, "DADD")
                || has_word(inst, "DFMA")
                || has_word(inst, "DSETP")
                || inst.contains("I2F.F64")
                || inst.contains("F2F.F64")
                || inst.contains("F2F.F32.F64")
            {
                hazards.fp64_ops += 1;
            }
            if inst.contains("MUFU.RCP") {
                hazards.rcp_div_ops += 1;
            }
            if inst.contains("BPT.TRAP") {
                hazards.trap_ops += 1;
            }
            if has_word(inst, "BSSY") || has_word(inst, "BSYNC") {
                hazards.bsync_ops += 1;
            }
            if inst.contains("HMMA.") || inst.contains("IMMA.") || inst.contains("WGMMA.") {
                hazards.tensor_ops += 1;
            }
            if inst.contains("LDGSTS") {
                hazards.async_copy_ops += 1;
            }
        }
        out.push(KernelEvidence {
            name,
            backend: "cuda".into(),
            target: "sass".into(),
            layer: "disassembly".into(),
            code_bytes: insns * 16,
            static_insns: insns,
            gpr_count: 0,
            uniform_regs: 0,
            shared_bytes: 0,
            frame_spill_bytes: 0,
            hazards,
        });
    }
    out
}

pub(super) fn has_word(text: &str, word: &str) -> bool {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|token| token == word)
}

/// Inspect LLVM IR / NVVM IR / AIR IR (`.ll`) across CUDA, Metal AIR, and
/// AMDGPU targets.
pub fn inspect_ir(text: &str) -> Vec<KernelEvidence> {
    let mut target = "llvm-ir".to_owned();
    let mut backend = "cuda".to_owned();
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("target triple =") {
            let triple = rest.trim().trim_matches('"');
            target = triple.to_owned();
            if triple.contains("air64") || triple.contains("apple") {
                backend = "metal".into();
            } else if triple.contains("amdgcn") {
                backend = "rocm".into();
            }
            break;
        }
    }
    let mut out = Vec::new();
    for line_idx in text.match_indices("define ") {
        let tail = &text[line_idx.0..];
        let Some(open_brace) = tail.find('{') else {
            continue;
        };
        let header = &tail[..open_brace];
        let Some(at_pos) = header.find('@') else {
            continue;
        };
        let after_at = &header[at_pos + 1..];
        let name = after_at
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
            .find(|s| !s.is_empty())
            .unwrap_or("unknown")
            .to_owned();
        let body_tail = &tail[open_brace + 1..];
        let end = body_tail.find("\n}\n").unwrap_or(body_tail.len());
        let body = &body_tail[..end];

        let mut insns = 0u64;
        let mut alloca_ops = 0u64;
        let mut hazards = HazardCounts::default();
        for l in body.lines() {
            let t = l.trim();
            if t.is_empty() || t.starts_with(';') || t.ends_with(':') {
                continue;
            }
            insns += 1;
            if t.contains(" = alloca ") || t.starts_with("alloca ") {
                alloca_ops += 1;
            }
            if t.contains(" double ") || t.ends_with(" double") {
                hazards.fp64_ops += 1;
            }
            if t.contains(" udiv ")
                || t.contains(" urem ")
                || t.contains(" sdiv ")
                || t.contains(" srem ")
                || t.contains(" fdiv ")
            {
                hazards.rcp_div_ops += 1;
            }
            if t.contains("llvm.trap") || t.contains("unreachable") {
                hazards.trap_ops += 1;
            }
            if t.contains("nvvm.mma")
                || t.contains("air.simdgroup_matrix")
                || t.contains("amdgcn.wmma")
                || t.contains("amdgcn.mfma")
            {
                hazards.tensor_ops += 1;
            }
        }
        hazards.spill_ops = alloca_ops;
        out.push(KernelEvidence {
            name,
            backend: backend.clone(),
            target: target.clone(),
            layer: "llvm-ir".into(),
            code_bytes: body.len() as u64,
            static_insns: insns,
            gpr_count: 0,
            uniform_regs: 0,
            shared_bytes: 0,
            frame_spill_bytes: 0,
            hazards,
        });
    }
    out
}

/// Inspect Metal (`.metal`) or OpenCL (`.cl`) shader source files.
pub fn inspect_shader(text: &str, backend: &str) -> Vec<KernelEvidence> {
    let marker = if backend == "opencl" {
        "__kernel void "
    } else {
        "kernel void "
    };
    let mut out = Vec::new();
    let mut parts = text.split(marker);
    let _prefix = parts.next();
    for part in parts {
        let name = part
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .find(|s| !s.is_empty())
            .unwrap_or("unknown")
            .to_owned();
        let Some(open_brace) = part.find('{') else {
            continue;
        };
        let body = &part[open_brace + 1..];
        let mut statements = 0u64;
        let mut hazards = HazardCounts::default();
        for line in body.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with("//") {
                continue;
            }
            if t.ends_with(';') {
                statements += 1;
            }
            if has_word(t, "double") {
                hazards.fp64_ops += 1;
            }
            if t.contains("simdgroup_multiply_accumulate") || t.contains("simdgroup_matrix") {
                hazards.tensor_ops += 1;
            }
        }
        out.push(KernelEvidence {
            name,
            backend: backend.to_owned(),
            target: format!("{backend}-source"),
            layer: "source".into(),
            code_bytes: body.len() as u64,
            static_insns: statements,
            gpr_count: 0,
            uniform_regs: 0,
            shared_bytes: 0,
            frame_spill_bytes: 0,
            hazards,
        });
    }
    out
}
