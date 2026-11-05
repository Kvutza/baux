use super::*;

pub(super) struct CandidateWeights {
    pub(super) architecture: ResidualArchitecture,
    pub(super) initialization: crate::config::ModelInitialization,
    pub(super) router: Buffer,
    pub(super) qkv: Buffer,
    pub(super) output: Buffer,
    pub(super) gate_up: Buffer,
    pub(super) down: Buffer,
    pub(super) attention_norm: Buffer,
    pub(super) ffn_norm: Buffer,
    pub(super) readout: Buffer,
    pub(super) mask_embed: Buffer,
    pub(super) index_query: Buffer,
    pub(super) feedback_state: Buffer,
    pub(super) feedback_gate: Buffer,
    pub(super) mhc_attention_predictor: Buffer,
    pub(super) mhc_attention_bias: Buffer,
    pub(super) mhc_attention_control: Buffer,
    pub(super) mhc_moe_predictor: Buffer,
    pub(super) mhc_moe_bias: Buffer,
    pub(super) mhc_moe_control: Buffer,
    pub(super) final_norm: Buffer,
    pub(super) hnet: Buffer,
}

#[derive(Clone, Copy)]
pub(super) struct Mhc4Layout {
    pub(super) architecture: ResidualArchitecture,
    pub(super) bytes: u64,
    pub(super) router: u64,
    pub(super) qkv: u64,
    pub(super) output: u64,
    pub(super) gate_up: u64,
    pub(super) down: u64,
    pub(super) attention_norm: u64,
    pub(super) ffn_norm: u64,
    pub(super) readout: u64,
    pub(super) mask_embed: u64,
    pub(super) index_query: u64,
    pub(super) feedback_state: u64,
    pub(super) feedback_gate: u64,
    pub(super) mhc_attention_predictor: u64,
    pub(super) mhc_attention_bias: u64,
    pub(super) mhc_attention_control: u64,
    pub(super) mhc_moe_predictor: u64,
    pub(super) mhc_moe_bias: u64,
    pub(super) mhc_moe_control: u64,
    pub(super) final_norm: u64,
    pub(super) hnet: u64,
}

#[derive(Clone, Copy)]
pub(super) struct CandidateRow<'a> {
    pub(super) buffer: &'a BufferRef,
    pub(super) layout: Mhc4Layout,
}

impl std::ops::Deref for CandidateRow<'_> {
    type Target = Mhc4Layout;

    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl<'a> CandidateRow<'a> {
    /// Hyper-connection parameters as typed, layer-stacked views.
    pub(super) fn hyper(self) -> crate::nn::hyper::Weights<'a> {
        use crate::nn::hyper::{SiteWeights, Weights};
        use crate::tensor::View;
        let stack = |offset, cols| View::new(self.buffer, offset, MODEL_LAYERS, cols);
        let site = |predictor, bias, control, norm| SiteWeights {
            predictor: stack(predictor, MHC_INPUT * MHC_COEFFICIENTS),
            bias: stack(bias, MHC_COEFFICIENTS),
            control: stack(control, 4),
            norm: stack(norm, WIDTH),
        };
        Weights {
            attention: site(
                self.mhc_attention_predictor,
                self.mhc_attention_bias,
                self.mhc_attention_control,
                self.attention_norm,
            ),
            ffn: site(
                self.mhc_moe_predictor,
                self.mhc_moe_bias,
                self.mhc_moe_control,
                self.ffn_norm,
            ),
            final_norm: View::new(self.buffer, self.final_norm, 1, WIDTH),
            code: self.architecture.kernel_code(),
        }
    }

    /// MoE FFN parameters as typed, layer-stacked views.
    pub(super) fn moe(self, int8_gate: Option<&'a BufferRef>) -> crate::nn::moe::Weights<'a> {
        use crate::nn::moe::{GateWeights, Weights};
        use crate::tensor::View;
        let gate_cols = routing::MOE_EXPERTS * WIDTH * routing::MOE_GATEUP;
        let gate_up = match int8_gate {
            Some(quantized) => GateWeights::I8(View::new(quantized, 0, MODEL_LAYERS, gate_cols)),
            None => GateWeights::F16(View::new(
                self.buffer,
                self.gate_up,
                MODEL_LAYERS,
                gate_cols,
            )),
        };
        Weights {
            router: View::new(
                self.buffer,
                self.router,
                MODEL_LAYERS,
                WIDTH * routing::ROUTED_EXPERTS,
            ),
            gate_up,
            down: View::new(
                self.buffer,
                self.down,
                MODEL_LAYERS,
                routing::MOE_EXPERTS * routing::ROUTED_WIDTH * WIDTH,
            ),
        }
    }

    /// Attention QKV and output projection parameters as typed, layer-stacked
    /// views.
    pub(super) fn attention(self) -> crate::nn::attention::Weights<'a> {
        use crate::nn::attention::Weights;
        use crate::tensor::View;
        Weights {
            qkv: View::new(self.buffer, self.qkv, MODEL_LAYERS * WIDTH, QKV_WIDTH),
            output: View::new(self.buffer, self.output, MODEL_LAYERS * WIDTH, WIDTH),
        }
    }

    /// Readout vocabulary embedding parameters as typed views.
    pub(super) fn readout_weights(self) -> crate::nn::loss::ReadoutWeights<'a> {
        crate::nn::loss::ReadoutWeights::new(crate::tensor::View::new(
            self.buffer,
            self.readout,
            WIDTH,
            self.architecture.vocabulary(),
        ))
    }
}

fn half_squares() -> &'static [f64; 32_768] {
    static TABLE: std::sync::OnceLock<Box<[f64; 32_768]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = Box::new([0.0_f64; 32_768]);
        for bits in 0..32_768_u16 {
            let v = decode_half(bits);
            table[usize::from(bits)] = v * v;
        }
        table
    })
}

fn tensor_rms(tensor: &[u16]) -> f64 {
    use rayon::prelude::*;
    let table = half_squares();
    let sum: f64 = if tensor.len() >= 65_536 {
        tensor
            .par_chunks(65_536)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|&bits| table[usize::from(bits & 0x7fff)])
                    .sum::<f64>()
            })
            .sum()
    } else {
        tensor
            .iter()
            .map(|&bits| table[usize::from(bits & 0x7fff)])
            .sum()
    };
    (sum / tensor.len() as f64).sqrt()
}

fn chunk_rms(chunks: &[&[u16]]) -> Vec<f64> {
    use rayon::prelude::*;
    chunks
        .par_iter()
        .map(|&tensor| tensor_rms(tensor))
        .collect()
}

fn load_rms(key: u64, count: usize) -> Option<Vec<f64>> {
    let path =
        std::path::Path::new(".cache/baux/weights").join(format!("mhc4-rms-v1-{key:016x}.bin"));
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() != count * 8 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    )
}

fn save_rms(key: u64, values: &[f64]) {
    let dir = std::path::Path::new(".cache/baux/weights");
    let _ = std::fs::create_dir_all(dir);
    let tmp = dir.join(format!("mhc4-rms-v1-{key:016x}.{}.tmp", std::process::id()));
    let path = dir.join(format!("mhc4-rms-v1-{key:016x}.bin"));
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    if std::fs::write(&tmp, &bytes).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

fn map_base(key: u64, expected: usize) -> Option<Buffer> {
    let padded = (expected * 2).next_multiple_of(16_384);
    let path =
        std::path::Path::new(".cache/baux/weights").join(format!("mhc4-base-v1-{key:016x}.bin"));
    let file = std::fs::File::open(path).ok()?;
    if file.metadata().ok()?.len() != padded as u64 {
        return None;
    }
    Runtime::shared().ok()?.mmap_buffer(&file, 0, padded)
}

fn save_base(key: u64, base: &[u16]) -> Option<Buffer> {
    use std::io::Write;
    let padded = (base.len() * 2).next_multiple_of(16_384);
    let dir = std::path::Path::new(".cache/baux/weights");
    let _ = std::fs::create_dir_all(dir);
    let tmp = dir.join(format!(
        "mhc4-base-v1-{key:016x}.{}.tmp",
        std::process::id()
    ));
    let path = dir.join(format!("mhc4-base-v1-{key:016x}.bin"));
    let raw = unsafe { std::slice::from_raw_parts(base.as_ptr().cast::<u8>(), base.len() * 2) };
    let ok = std::fs::File::create(&tmp)
        .and_then(|mut f| {
            f.write_all(raw)?;
            f.write_all(&vec![0u8; padded - raw.len()])?;
            f.flush()
        })
        .and_then(|()| std::fs::rename(&tmp, &path))
        .is_ok();
    if !ok {
        let _ = std::fs::remove_file(&tmp);
        return None;
    }
    map_base(key, base.len())
}

fn build_search(
    mapped_base: Option<Buffer>,
    use_cache: bool,
    key: u64,
    base: Vec<u16>,
    expected: usize,
    blocks: Vec<ParamBlock>,
    length: crate::trust_region::TRLengthConfig,
    perturbation: crate::ProposalKernel,
) -> Result<SearchState, String> {
    let mapped = mapped_base.or_else(|| use_cache.then(|| save_base(key, &base)).flatten());
    let mut search = if let Some(base_buf) = mapped {
        drop(base);
        SearchState::new_mapped(
            base_buf,
            expected,
            blocks,
            HISTORY_CAPACITY,
            length,
            perturbation,
        )?
    } else {
        SearchState::new_implicit(&base, blocks, HISTORY_CAPACITY, length, perturbation)?
    };
    search.set_tolerance(4)?;
    Ok(search)
}

impl CandidateWeights {
    fn cache_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.architecture.parameter_count().hash(&mut hasher);
        self.architecture.kernel_code().hash(&mut hasher);
        (self.initialization as u8).hash(&mut hasher);
        for buf in [&self.router, &self.readout, &self.final_norm] {
            let n = (buf.length() as usize / 2).min(64);
            let slice = unsafe { std::slice::from_raw_parts(buf.contents().cast::<u16>(), n) };
            slice.hash(&mut hasher);
        }
        if self.architecture.is_hnet() {
            // A hierarchy-only checkpoint update must not reuse an older BO
            // base.
            let count = self.hnet.length() as usize / 2;
            let values =
                unsafe { std::slice::from_raw_parts(self.hnet.contents().cast::<u16>(), count) };
            values.hash(&mut hasher);
        }
        hasher.finish()
    }

    fn all_rms(&self, use_cache: bool, key: u64) -> Vec<f64> {
        let tensors = self.tensors();
        let total_blocks: usize = tensors
            .iter()
            .map(|(_, buf, len)| buf.length() as usize / 2 / len)
            .sum();
        if use_cache && let Some(cached) = load_rms(key, total_blocks) {
            return cached;
        }
        let mut out = Vec::with_capacity(total_blocks);
        for (_, buffer, block_len) in tensors {
            let values = unsafe {
                std::slice::from_raw_parts(
                    buffer.contents().cast::<u16>(),
                    buffer.length() as usize / 2,
                )
            };
            let chunks: Vec<&[u16]> = values.chunks_exact(block_len).collect();
            out.extend(chunk_rms(&chunks));
        }
        if use_cache {
            save_rms(key, &out);
        }
        out
    }

    pub(super) fn row<'a>(&self, buffer: &'a BufferRef) -> Result<CandidateRow<'a>, String> {
        let row_bytes = (self.architecture.parameter_count() * 2).min(buffer.length() as usize);
        self.row_view(buffer, 0, row_bytes)
    }

    pub(super) fn row_view<'a>(
        &self,
        buffer: &'a BufferRef,
        base: usize,
        row_bytes: usize,
    ) -> Result<CandidateRow<'a>, String> {
        let layout = Mhc4Layout::from_weights(self, base as u64);
        if layout.bytes as usize != row_bytes
            || base
                .checked_add(row_bytes)
                .is_none_or(|required| required > buffer.length() as usize)
        {
            return Err(format!(
                "candidate row has {row_bytes} bytes at offset {base}, expected {} within {} bytes",
                layout.bytes,
                buffer.length(),
            ));
        }
        Ok(layout.view(buffer))
    }

    pub(super) fn search(
        &self,
        perturbation: crate::ProposalKernel,
        length: crate::trust_region::TRLengthConfig,
    ) -> Result<(SearchState, UpdateLog), String> {
        self.search_shaped(
            perturbation,
            length,
            crate::config::TrustRegionShape::TensorFamilyStatic,
        )
    }

    pub(super) fn search_shaped(
        &self,
        perturbation: crate::ProposalKernel,
        length: crate::trust_region::TRLengthConfig,
        shape: crate::config::TrustRegionShape,
    ) -> Result<(SearchState, UpdateLog), String> {
        let expected_parameters = self.architecture.parameter_count();
        let use_cache = expected_parameters > 10_000_000;
        let key = if use_cache { self.cache_key() } else { 0 };
        let mapped_base = if use_cache {
            map_base(key, expected_parameters)
        } else {
            None
        };
        let mut base = if mapped_base.is_some() {
            Vec::new()
        } else {
            Vec::with_capacity(expected_parameters)
        };
        let rms_all = self.all_rms(use_cache, key);
        let mut rms_iter = rms_all.into_iter();
        let mut offset = 0usize;
        let mut blocks = Vec::new();
        let mut updates = UpdateLog::default();
        let mhc4_policy = (self.initialization == crate::config::ModelInitialization::FbtPisa1Mhc4)
            .then(|| Mhc4Initialization::new(self.architecture))
            .transpose()?;
        for (family_code, (family, buffer, block_len)) in self.tensors().into_iter().enumerate() {
            let values = unsafe {
                std::slice::from_raw_parts(
                    buffer.contents().cast::<u16>(),
                    buffer.length() as usize / 2,
                )
            };
            for (index, tensor) in values.chunks_exact(block_len).enumerate() {
                let initial_rms = rms_iter.next().ok_or("missing block RMS")?;
                let (layer, expert) = match family {
                    "expert_gate_up" | "expert_down" => (
                        Some(index / routing::MOE_EXPERTS as usize),
                        Some(index % routing::MOE_EXPERTS as usize),
                    ),
                    "embedding_readout" | "feedback_state" | "feedback_gate" | "final_norm"
                    | "mask_embed" | "index_query" | "hnet" => (None, None),
                    _ => (Some(index), None),
                };
                let (scale, _trust_multiplier) =
                    self.block_scale(mhc4_policy, shape, family, expert, initial_rms)?;
                let logical_start = index
                    .checked_mul(block_len)
                    .ok_or("tensor block logical offset overflow")?;
                let tensor_id = crate::hash::tensor_key(family);
                updates.push_block(
                    offset,
                    tensor.len(),
                    baux::semantics::optimizer::param_tree::ParamFamily::from_expert(
                        family, expert,
                    ),
                    initial_rms,
                    scale,
                )?;
                let address = ProgramAddress::new(family_code, layer, expert)?;
                blocks.push(
                    ParamBlock::new_logical(
                        tensor_id,
                        logical_start as u64,
                        offset,
                        tensor.len(),
                        scale,
                        1.0 / (tensor.len() as f32 * scale * scale),
                    )?
                    .with_address(address),
                );
                if mapped_base.is_none() {
                    base.extend_from_slice(tensor);
                }
                offset += tensor.len();
            }
        }
        if offset != expected_parameters {
            return Err(format!(
                "Full-weight inventory has {offset} parameters, expected {expected_parameters}"
            ));
        }
        let search = build_search(
            mapped_base,
            use_cache,
            key,
            base,
            expected_parameters,
            blocks,
            length,
            perturbation,
        )?;
        Ok((search, updates))
    }
}

impl Mhc4Layout {
    pub(super) fn from_weights(weights: &CandidateWeights, base: u64) -> Self {
        let mut end = base;
        let mut next = |tensor: &Buffer| {
            let start = end;
            end += tensor.length();
            start
        };
        let router = next(&weights.router);
        let qkv = next(&weights.qkv);
        let output = next(&weights.output);
        let gate_up = next(&weights.gate_up);
        let down = next(&weights.down);
        let attention_norm = next(&weights.attention_norm);
        let ffn_norm = next(&weights.ffn_norm);
        let readout = next(&weights.readout);
        let (
            feedback_state,
            feedback_gate,
            mhc_attention_predictor,
            mhc_attention_bias,
            mhc_attention_control,
            mhc_moe_predictor,
            mhc_moe_bias,
            mhc_moe_control,
        ) = if weights.architecture.is_multistream() {
            (
                0,
                0,
                next(&weights.mhc_attention_predictor),
                next(&weights.mhc_attention_bias),
                next(&weights.mhc_attention_control),
                next(&weights.mhc_moe_predictor),
                next(&weights.mhc_moe_bias),
                next(&weights.mhc_moe_control),
            )
        } else {
            (
                next(&weights.feedback_state),
                next(&weights.feedback_gate),
                0,
                0,
                0,
                0,
                0,
                0,
            )
        };
        let final_norm = next(&weights.final_norm);
        let (mask_embed, index_query) =
            if weights.architecture == ResidualArchitecture::DiffusionMhc4 {
                (next(&weights.mask_embed), next(&weights.index_query))
            } else {
                (0, 0)
            };
        let hnet = if weights.architecture.is_hnet() {
            next(&weights.hnet)
        } else {
            0
        };
        Self {
            architecture: weights.architecture,
            bytes: end - base,
            router,
            qkv,
            output,
            gate_up,
            down,
            attention_norm,
            ffn_norm,
            readout,
            mask_embed,
            index_query,
            feedback_state,
            feedback_gate,
            mhc_attention_predictor,
            mhc_attention_bias,
            mhc_attention_control,
            mhc_moe_predictor,
            mhc_moe_bias,
            mhc_moe_control,
            final_norm,
            hnet,
        }
    }

    pub(super) fn view(self, buffer: &BufferRef) -> CandidateRow<'_> {
        CandidateRow {
            buffer,
            layout: self,
        }
    }
}

pub(super) fn candidate_ffn(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    normalized: &BufferRef,
    residual: &BufferRef,
    layer: u32,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    next_norm: Option<(&BufferRef, u64)>,
) -> Result<(), String> {
    let shape = model_shape();
    let shape_ptr = (&shape as *const MoeShape).cast();
    let shape_bytes = std::mem::size_of::<MoeShape>() as u64;
    let (router_offset, gate_up_offset, down_offset) = ffn_offsets(layer);
    pipelines.fine_grained.encode(
        command,
        &buffers.fine_grained,
        normalized,
        0,
        weights.buffer,
        weights.router + router_offset,
        weights.gate_up + gate_up_offset,
        weights.down + down_offset,
        &buffers.down,
    )?;
    if let Some((weight, weight_offset)) = next_norm {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipelines.residual_rms);
        encoder.set_buffer(0, Some(residual), 0);
        encoder.set_buffer(1, Some(&buffers.down), 0);
        encoder.set_buffer(2, Some(weight), weight_offset);
        encoder.set_buffer(3, Some(&buffers.output), 0);
        encoder.set_buffer(4, Some(&buffers.normalized), 0);
        encoder.set_bytes(5, shape_bytes, shape_ptr);
        encoder.dispatch_thread_groups(
            thread_group(u64::from(ROWS)),
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
    } else {
        encode_residual(command, pipelines, residual, &buffers.down, &buffers.output);
    }
    Ok(())
}

pub(super) fn ffn_offsets(layer: u32) -> (u64, u64, u64) {
    let router_offset =
        half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(routing::ROUTED_EXPERTS));
    let gate_up_offset = half_bytes(
        u64::from(layer)
            * u64::from(routing::MOE_EXPERTS)
            * u64::from(WIDTH)
            * u64::from(routing::MOE_GATEUP),
    );
    let down_offset = half_bytes(
        u64::from(layer)
            * u64::from(routing::MOE_EXPERTS)
            * u64::from(routing::ROUTED_WIDTH)
            * u64::from(WIDTH),
    );
    (router_offset, gate_up_offset, down_offset)
}
