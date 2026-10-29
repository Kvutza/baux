use super::VOCAB;
pub(super) use super::proposal::WeightBackup;
use super::proposal::{GaussianRandom, NeighborSampler, ProposalSpec};
use crate::{CudaResult, cuda_error, read_prefix};
use baux::artifact::json::Value;
use baux::semantics::mhc4::Initialization as Mhc4Scales;
use baux::semantics::random::splitmix64 as splitmix;
use baux::semantics::tensor::key as tensor_key;
use baux_cuda_device::fbt_model;
use cuda_core::{CudaStream, DeviceBuffer};
use deser::Deserialize;
use half::f16;
use memmap2::MmapOptions;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Header {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub(super) struct Site {
    pub predictor: DeviceBuffer<u16>,
    pub bias: DeviceBuffer<u16>,
    pub control: DeviceBuffer<u16>,
    pub norm: DeviceBuffer<u16>,
}

pub(super) struct Layer {
    pub qkv: DeviceBuffer<u16>,
    pub output: DeviceBuffer<u16>,
    pub router: DeviceBuffer<u16>,
    pub gate: DeviceBuffer<u16>,
    pub down: DeviceBuffer<u16>,
    pub attention: Site,
    pub moe: Site,
}

pub(crate) struct DiffusionWeights {
    pub mask: DeviceBuffer<u16>,
    pub index: DeviceBuffer<u16>,
}

pub(crate) struct Mhc4Weights {
    pub mask: DeviceBuffer<u16>,
    pub positions: DeviceBuffer<u16>,
}

pub(crate) struct Weights {
    pub format: &'static str,
    pub diffusion: Option<DiffusionWeights>,
    pub mhc4: Option<Mhc4Weights>,
    pub embedding: DeviceBuffer<u16>,
    pub readout: DeviceBuffer<u16>,
    pub final_norm: DeviceBuffer<u16>,
    pub layers: Vec<Layer>,
    proposal: Vec<ProposalSpec>,
    gaussian_random: Option<GaussianRandom>,
    neighbor_sampler: Option<NeighborSampler>,
}

struct Checkpoint {
    map: memmap2::Mmap,
    start: usize,
    headers: BTreeMap<String, Header>,
    format: &'static str,
}

/// Storage-only fixture through the real full-row materialization adapter.
/// Three differently sized tensors exercise partial CTAs, status-buffer reuse,
/// invalid tail coordinates, unchanged rows, rollback and subsequent recovery.
/// This is not a model forward pass or an optimizer learning experiment.
pub fn proposal_check() -> CudaResult<()> {
    super::proposal::check()
}

impl Checkpoint {
    fn open(path: &std::path::Path) -> CudaResult<Self> {
        let file = std::fs::File::open(path).map_err(cuda_error)?;
        // SAFETY: read-only mapping retained for every tensor upload.
        let map = unsafe { MmapOptions::new().map(&file) }.map_err(cuda_error)?;
        if map.len() < 8 {
            return Err("truncated Safetensors prefix".into());
        }
        let len = usize::try_from(u64::from_le_bytes(map[..8].try_into().unwrap()))
            .map_err(cuda_error)?;
        let start = len.checked_add(8).ok_or("checkpoint header overflow")?;
        if start > map.len() {
            return Err("truncated Safetensors header".into());
        }
        let mut root: BTreeMap<String, Value> =
            baux::artifact::json::from_slice(&map[8..start]).map_err(cuda_error)?;
        let metadata: BTreeMap<String, String> = baux::artifact::json::from_value(
            root.remove("__metadata__")
                .ok_or("checkpoint metadata missing")?,
        )
        .map_err(cuda_error)?;
        let format = match metadata.get("format").map(String::as_str) {
            Some("baux.fbt-pisa1-looped-mhc4-rope.v1") => "looped-mhc4",
            Some("baux.fbt-pisa1-diffusion-mhc4-rope.v1") => "diffusion-mhc4",
            Some("baux.fbt-pisa1-mhc4-rope.v1") => "mhc4",
            _ => {
                return Err("checkpoint must be a looped, diffusion, or mHC-4 RoPE model".into());
            }
        };
        for (name, expected) in [
            ("position_encoding", "rope"),
            ("rope_base", "10000"),
            ("rope_dimensions", "64"),
        ] {
            if metadata.get(name).map(String::as_str) != Some(expected) {
                return Err(format!("checkpoint {name} must be {expected}"));
            }
        }
        let mut headers = BTreeMap::new();
        let mut ranges = Vec::new();
        for (name, value) in root {
            let header: Header = baux::artifact::json::from_value(value).map_err(cuda_error)?;
            let count = header
                .shape
                .iter()
                .try_fold(1_usize, |n, d| n.checked_mul(*d))
                .ok_or("tensor shape overflow")?;
            let [a, b] = header.data_offsets;
            if header.dtype != "F16"
                || a > b
                || count.checked_mul(2) != Some(b - a)
                || start.checked_add(b).is_none_or(|end| end > map.len())
            {
                return Err(format!("invalid F16 tensor {name}"));
            }
            ranges.push((a, b));
            headers.insert(name, header);
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err("overlapping checkpoint tensors".into());
        }
        Ok(Self {
            map,
            start,
            headers,
            format,
        })
    }

    fn tensor(&self, name: &str, shape: &[usize], layer: Option<usize>) -> CudaResult<Vec<u16>> {
        let header = self
            .headers
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if header.shape != shape {
            return Err(format!(
                "tensor {name} shape {:?}, expected {shape:?}",
                header.shape
            ));
        }
        let [mut a, mut b] = header.data_offsets;
        if let Some(layer) = layer {
            let stride = (b - a) / 5;
            a += layer * stride;
            b = a + stride;
        }
        Ok(self.map[self.start + a..self.start + b]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }
}

fn spherical(
    random: &mut StdRng,
    matrices: usize,
    input: usize,
    output: usize,
    column_norm: f32,
) -> Vec<u16> {
    let elements = matrices
        .checked_mul(input)
        .and_then(|v| v.checked_mul(output))
        .expect("fixed model tensor dimensions must fit usize");
    let mut values = vec![0; elements];
    baux::semantics::mhc4::columns(
        &mut values,
        input,
        output,
        column_norm,
        || StandardNormal.sample(random),
        |v| f16::from_f32(v).to_bits(),
    )
    .expect("fixed mHC-4 Gaussian column initialization");
    values
}

fn mhc_bias(layers: usize, coefficients: usize) -> Vec<u16> {
    let mut values = vec![0; layers * coefficients];
    let input_bias = f16::from_f32(-3.0_f32.ln()).to_bits();
    for layer in 0..layers {
        values[layer * coefficients..layer * coefficients + 4].fill(input_bias);
    }
    values
}

fn mhc4_specs(
    weights: &Weights,
    stream: &CudaStream,
    policy: Mhc4Scales,
    shape: baux::semantics::mhc4::ScaleShape,
) -> CudaResult<Vec<ProposalSpec>> {
    // Order matches buffers_mut. The tied embedding is a physical transpose,
    // not another logical parameter family.
    let mut inventory = vec![
        (
            "embedding_readout",
            weights.embedding.len(),
            512 * VOCAB,
            0,
            512,
        ),
        (
            "embedding_readout",
            weights.readout.len(),
            512 * VOCAB,
            0,
            0,
        ),
        ("final_norm", weights.final_norm.len(), 512, 0, 0),
    ];
    for (index, layer) in weights.layers.iter().enumerate() {
        for (name, elements, block) in [
            ("qkv", layer.qkv.len(), 512 * 640),
            ("attention_output", layer.output.len(), 512 * 512),
            ("router", layer.router.len(), 512 * 625),
            ("expert_gate_up", layer.gate.len(), 512 * 432),
            ("expert_down", layer.down.len(), 216 * 512),
            (
                "mhc_attention_predictor",
                layer.attention.predictor.len(),
                2048 * 24,
            ),
            ("mhc_attention_bias", layer.attention.bias.len(), 24),
            ("mhc_attention_control", layer.attention.control.len(), 4),
            ("attention_norm", layer.attention.norm.len(), 512),
            ("mhc_moe_predictor", layer.moe.predictor.len(), 2048 * 24),
            ("mhc_moe_bias", layer.moe.bias.len(), 24),
            ("mhc_moe_control", layer.moe.control.len(), 4),
            ("ffn_norm", layer.moe.norm.len(), 512),
        ] {
            inventory.push((name, elements, block, index * elements, 0));
        }
    }
    inventory
        .into_iter()
        .map(|(name, elements, block_len, start, transpose)| {
            if block_len == 0 || elements % block_len != 0 {
                return Err("invalid procedural mHC-4 proposal geometry".into());
            }
            let host_scales =
                vec![policy.proposal_scale(name)? * shape.multiplier(name); elements / block_len];
            Ok(ProposalSpec {
                elements,
                block_len: block_len as u32,
                scales: DeviceBuffer::from_host(stream, &host_scales).map_err(cuda_error)?,
                host_scales,
                tensor_id: tensor_key(name),
                logical_start: start as u64,
                transpose_columns: transpose as u32,
            })
        })
        .collect()
}

fn init_layers(seed: u64, policy: Mhc4Scales, stream: &CudaStream) -> CudaResult<Vec<Layer>> {
    use baux::semantics::shape as dimensions;
    const LAYERS: usize = dimensions::MHC4_LAYERS as usize;
    const WIDTH: usize = dimensions::MHC4_WIDTH as usize;
    const ROUTED: usize = dimensions::ROUTED_COUNT as usize;
    const EXPERTS: usize = dimensions::EXPERT_COUNT as usize;
    const GATE: usize = dimensions::GATE_WIDTH as usize;
    const DOWN: usize = dimensions::EXPERT_WIDTH as usize;
    const MHC_INPUT: usize = dimensions::MHC_INPUT as usize;
    const MHC_COEFFICIENTS: usize = dimensions::MHC_COEFS as usize;
    let random = |domain| StdRng::seed_from_u64(splitmix(seed ^ domain));
    let mut router_rng = random(0x6a09_e667);
    let mut qkv_rng = random(0xbb67_ae85);
    let mut output_rng = random(0x3c6e_f372);
    let mut gate_rng = random(0xa54f_f53a);
    let mut down_rng = random(0x510e_527f);
    let mut attention_rng = random(0x428a_2f98);
    let mut moe_rng = random(0x7137_4491);
    let upload = |values: &[u16]| DeviceBuffer::from_host(stream, values).map_err(cuda_error);
    let site = |rng: &mut StdRng| -> CudaResult<Site> {
        Ok(Site {
            predictor: upload(&spherical(rng, 1, MHC_INPUT, MHC_COEFFICIENTS, 1.0))?,
            bias: upload(&mhc_bias(1, MHC_COEFFICIENTS))?,
            control: upload(&[
                f16::from_f32(policy.predictor_control).to_bits(),
                f16::from_f32(policy.predictor_control).to_bits(),
                f16::from_f32(policy.predictor_control).to_bits(),
                f16::from_f32(policy.transport_lambda).to_bits(),
            ])?,
            norm: upload(&vec![0x3c00; WIDTH])?,
        })
    };
    let mut layers = Vec::with_capacity(LAYERS);
    for _ in 0..LAYERS {
        layers.push(Layer {
            qkv: upload(&spherical(&mut qkv_rng, 1, WIDTH, 640, 1.0))?,
            output: upload(&spherical(
                &mut output_rng,
                1,
                WIDTH,
                WIDTH,
                policy.residual_gain,
            ))?,
            router: upload(&spherical(
                &mut router_rng,
                1,
                WIDTH,
                ROUTED,
                policy.router_column_norm,
            ))?,
            gate: upload(&spherical(&mut gate_rng, EXPERTS, WIDTH, GATE, 1.0))?,
            down: upload(&spherical(
                &mut down_rng,
                EXPERTS,
                DOWN,
                WIDTH,
                policy.down_norm(),
            ))?,
            attention: site(&mut attention_rng)?,
            moe: site(&mut moe_rng)?,
        });
    }
    Ok(layers)
}

impl Weights {
    /// Hash every logical parameter in packed-family order, independently of
    /// this backend's per-layer allocation order and tied physical transpose.
    pub(super) fn logical_hashes(
        &self,
        stream: &CudaStream,
    ) -> CudaResult<Vec<baux::artifact::forward::WeightHash>> {
        use baux::artifact::forward::WordHash;
        use baux::semantics::mhc4::probe::initialized::FAMILIES;
        let mut hashes = FAMILIES
            .iter()
            .map(|family| (tensor_key(family.name), WordHash::new(family.name)))
            .collect::<BTreeMap<_, _>>();
        let buffers = self.buffers();
        if buffers.len() != self.proposal.len() {
            return Err("CUDA digest omitted a physical parameter buffer".into());
        }
        for (index, buffer) in buffers.into_iter().enumerate() {
            let spec = self
                .proposal
                .get(index)
                .ok_or("CUDA digest inventory is incomplete")?;
            if spec.transpose_columns != 0 {
                continue;
            }
            let hash = hashes
                .get_mut(&spec.tensor_id)
                .ok_or("CUDA digest found an unknown parameter family")?;
            if spec.logical_start != hash.elements() || spec.elements != buffer.len() {
                return Err(
                    "CUDA logical parameter order or size differs from its descriptor".into(),
                );
            }
            hash.update(&read_prefix(buffer, stream, buffer.len())?)?;
        }
        let mut output = hashes
            .into_values()
            .map(WordHash::finish)
            .collect::<Vec<_>>();
        output.sort_by(|a, b| a.name.cmp(&b.name));
        if output.iter().map(|value| value.elements).sum::<u64>()
            != baux::semantics::shape::MHC4_PARAMS
        {
            return Err("CUDA logical parameter count differs from the mHC-4 contract".into());
        }
        // Verify the actual tied alias independently of the packing helper.
        let readout = read_prefix(&self.readout, stream, self.readout.len())?;
        let embedding = read_prefix(&self.embedding, stream, self.embedding.len())?;
        for token in 0..VOCAB {
            for dimension in 0..512 {
                if embedding[token * 512 + dimension] != readout[dimension * VOCAB + token] {
                    return Err(format!(
                        "CUDA tied embedding differs at token {token}, dimension {dimension}"
                    ));
                }
            }
        }
        Ok(output)
    }

    pub fn initialize_mhc4(seed: u64, stream: &CudaStream) -> CudaResult<Self> {
        const WIDTH: usize = baux::semantics::shape::MHC4_WIDTH as usize;
        let policy = Mhc4Scales::new();
        let random = |domain| StdRng::seed_from_u64(splitmix(seed ^ domain));
        let upload = |values: &[u16]| DeviceBuffer::from_host(stream, values).map_err(cuda_error);
        let layers = init_layers(seed, policy, stream)?;
        let mut readout_rng = random(0x9b05_688c);
        let readout_values = spherical(&mut readout_rng, 1, WIDTH, VOCAB, 1.0);
        let embedding = DeviceBuffer::from_host(
            stream,
            &crate::prefill::pack_embedding(&readout_values, WIDTH, VOCAB),
        )
        .map_err(cuda_error)?;
        let readout = upload(&readout_values)?;
        let mut mask_rng = random(0x4d41_534b);
        let mut position_rng = random(0x494e_4458);
        let mhc4 = Some(Mhc4Weights {
            mask: upload(&spherical(&mut mask_rng, 1, WIDTH, 1, 1.0))?,
            positions: upload(&spherical(&mut position_rng, 1, 64, WIDTH, 1.0))?,
        });
        let mut weights = Self {
            format: "mhc4",
            diffusion: None,
            mhc4,
            embedding,
            readout,
            final_norm: upload(&vec![0x3c00; WIDTH])?,
            layers,
            proposal: Vec::new(),
            gaussian_random: Some(GaussianRandom::new(stream)?),
            neighbor_sampler: None,
        };
        weights.proposal = mhc4_specs(
            &weights,
            stream,
            policy,
            baux::semantics::mhc4::ScaleShape::Scalar,
        )?;
        weights.neighbor_sampler = Some(NeighborSampler::new(&weights.proposal, stream)?);
        Ok(weights)
    }

    /// Resolve scales before creating the initial observation or weight backup.
    pub(super) fn set_shape(
        &mut self,
        shape: baux::semantics::mhc4::ScaleShape,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        self.proposal = mhc4_specs(self, stream, Mhc4Scales::new(), shape)?;
        self.neighbor_sampler = Some(NeighborSampler::new(&self.proposal, stream)?);
        Ok(())
    }

    pub fn open(path: &std::path::Path, stream: &CudaStream) -> CudaResult<Self> {
        let checkpoint = Checkpoint::open(path)?;
        let upload = |name: &str, shape: &[usize], layer| {
            DeviceBuffer::from_host(stream, &checkpoint.tensor(name, shape, layer)?)
                .map_err(cuda_error)
        };
        let readout = checkpoint.tensor("embedding_readout", &[512, VOCAB], None)?;
        let embedding = DeviceBuffer::from_host(
            stream,
            &crate::prefill::pack_embedding(&readout, 512, VOCAB),
        )
        .map_err(cuda_error)?;
        let readout = DeviceBuffer::from_host(stream, &readout).map_err(cuda_error)?;
        let final_norm = upload("final_norm", &[512], None)?;
        let mut layers = Vec::with_capacity(5);
        for layer in 0..5 {
            let site = |attention| -> CudaResult<Site> {
                let prefix = if attention {
                    "mhc_attention"
                } else {
                    "mhc_moe"
                };
                Ok(Site {
                    predictor: upload(&format!("{prefix}_predictor"), &[5, 2048, 24], Some(layer))?,
                    bias: upload(&format!("{prefix}_bias"), &[5, 24], Some(layer))?,
                    control: upload(&format!("{prefix}_control"), &[5, 4], Some(layer))?,
                    norm: upload(
                        if attention {
                            "attention_norm"
                        } else {
                            "ffn_norm"
                        },
                        &[5, 512],
                        Some(layer),
                    )?,
                })
            };
            layers.push(Layer {
                qkv: upload("qkv", &[5, 512, 640], Some(layer))?,
                output: upload("attention_output", &[5, 512, 512], Some(layer))?,
                router: upload("router", &[5, 512, 625], Some(layer))?,
                gate: upload("expert_gate_up", &[5, 626, 512, 432], Some(layer))?,
                down: upload("expert_down", &[5, 626, 216, 512], Some(layer))?,
                attention: site(true)?,
                moe: site(false)?,
            });
        }
        let diffusion = if checkpoint.format == "diffusion-mhc4" {
            Some(DiffusionWeights {
                mask: upload("mask_embed", &[512], None)?,
                index: upload("index_query", &[64, 64], None)?,
            })
        } else {
            None
        };
        let mhc4 = if checkpoint.format == "mhc4" && checkpoint.headers.contains_key("mask_embed") {
            Some(Mhc4Weights {
                mask: upload("mask_embed", &[512], None)?,
                positions: upload("index_query", &[64, 512], None)?,
            })
        } else {
            None
        };
        let proposal = proposal_specs(&checkpoint, stream)?;
        let neighbor_sampler = (checkpoint.format == "mhc4")
            .then(|| NeighborSampler::new(&proposal, stream))
            .transpose()?;
        Ok(Self {
            format: checkpoint.format,
            diffusion,
            mhc4,
            embedding,
            readout,
            final_norm,
            layers,
            proposal,
            gaussian_random: (checkpoint.format == "mhc4")
                .then(|| GaussianRandom::new(stream))
                .transpose()?,
            neighbor_sampler,
        })
    }

    pub(super) fn fixture(stream: &CudaStream) -> CudaResult<Self> {
        let upload =
            |value, len| DeviceBuffer::from_host(stream, &vec![value; len]).map_err(cuda_error);
        let mut layers = Vec::new();
        for _ in 0..5 {
            let site = || -> CudaResult<Site> {
                Ok(Site {
                    predictor: upload(0, 2048 * 24)?,
                    bias: upload(0, 24)?,
                    control: upload(0, 4)?,
                    norm: upload(0x3c00, 512)?,
                })
            };
            layers.push(Layer {
                qkv: upload(0x1800, 512 * 640)?,
                output: upload(0x1800, 512 * 512)?,
                router: upload(0, 512 * 625)?,
                gate: upload(0, 626 * 512 * 432)?,
                down: upload(0, 626 * 216 * 512)?,
                attention: site()?,
                moe: site()?,
            });
        }
        Ok(Self {
            format: "fixture",
            diffusion: None,
            mhc4: None,
            embedding: upload(0x3c00, VOCAB * 512)?,
            readout: upload(0x3c00, 512 * VOCAB)?,
            final_norm: upload(0x3c00, 512)?,
            layers,
            proposal: Vec::new(),
            gaussian_random: None,
            neighbor_sampler: None,
        })
    }

    fn buffers(&self) -> Vec<&DeviceBuffer<u16>> {
        let mut list = vec![&self.embedding, &self.readout, &self.final_norm];
        for layer in &self.layers {
            list.extend([
                &layer.qkv,
                &layer.output,
                &layer.router,
                &layer.gate,
                &layer.down,
            ]);
            for site in [&layer.attention, &layer.moe] {
                list.extend([&site.predictor, &site.bias, &site.control, &site.norm]);
            }
        }
        if let Some(diffusion) = &self.diffusion {
            list.extend([&diffusion.mask, &diffusion.index]);
        }
        list
    }

    fn buffers_mut(&mut self) -> Vec<&mut DeviceBuffer<u16>> {
        let mut list = vec![&mut self.embedding, &mut self.readout, &mut self.final_norm];
        for layer in &mut self.layers {
            list.extend([
                &mut layer.qkv,
                &mut layer.output,
                &mut layer.router,
                &mut layer.gate,
                &mut layer.down,
            ]);
            for site in [&mut layer.attention, &mut layer.moe] {
                list.extend([
                    &mut site.predictor,
                    &mut site.bias,
                    &mut site.control,
                    &mut site.norm,
                ]);
            }
        }
        if let Some(diffusion) = &mut self.diffusion {
            list.extend([&mut diffusion.mask, &mut diffusion.index]);
        }
        list
    }

    pub(super) fn backup(&self, stream: &CudaStream) -> CudaResult<WeightBackup> {
        WeightBackup::new(&self.buffers(), self.proposal.len(), stream)
    }

    pub(super) fn restore(
        &mut self,
        backup: &WeightBackup,
        stream: &CudaStream,
    ) -> CudaResult<f32> {
        backup.restore(&mut self.buffers_mut(), stream)
    }

    pub(super) fn commit(&self, backup: &mut WeightBackup, stream: &CudaStream) -> CudaResult<f32> {
        backup.commit(&self.buffers(), stream)
    }

    pub(super) fn reset_history(
        &mut self,
        backup: &WeightBackup,
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        self.neighbor_sampler
            .as_mut()
            .ok_or("CUDA mHC-4 neighbour sampler is missing")?
            .reset_history(backup, module, stream)
    }

    pub(super) fn pool_distances(
        &mut self,
        backup: &WeightBackup,
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
        radius: f32,
        seeds: &[u64],
        history_count: usize,
    ) -> CudaResult<Vec<Vec<f32>>> {
        let random = self
            .gaussian_random
            .as_ref()
            .ok_or("CUDA Gaussian random state is missing")?;
        self.neighbor_sampler
            .as_mut()
            .ok_or("CUDA mHC-4 neighbour sampler is missing")?
            .pool_distances(backup, random, module, stream, radius, seeds, history_count)
    }

    pub(super) fn archive_neighbor(
        &mut self,
        candidate: usize,
        row: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        self.neighbor_sampler
            .as_mut()
            .ok_or("CUDA mHC-4 neighbour sampler is missing")?
            .archive(candidate, row, stream)
    }

    pub(super) fn materialize_gaussian(
        &mut self,
        backup: &WeightBackup,
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
        radius: f32,
        seed: u64,
    ) -> CudaResult<f32> {
        let mut random = self
            .gaussian_random
            .take()
            .ok_or("CUDA Gaussian random state is missing")?;
        let specs = std::mem::take(&mut self.proposal);
        let result = random.materialize(
            &specs,
            backup,
            &mut self.buffers_mut(),
            module,
            stream,
            radius,
            seed,
        );
        // A rejected proposal must not strand the host inventory/RNG or leave
        // invalid candidate weights available to a later evaluation.
        self.proposal = specs;
        self.gaussian_random = Some(random);
        result
    }

    pub(super) fn materialize_basis0(
        &mut self,
        backup: &WeightBackup,
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
        radius: f32,
    ) -> CudaResult<f32> {
        let specs = std::mem::take(&mut self.proposal);
        let result = backup.step_basis0(&specs, &mut self.buffers_mut(), module, stream, radius);
        self.proposal = specs;
        result
    }

    pub(super) fn physical_elements(&self) -> u64 {
        self.buffers().into_iter().map(|b| b.len() as u64).sum()
    }
}

fn rms(values: &[u16], floor: f32) -> f32 {
    let sum = values
        .iter()
        .map(|&bits| {
            let value = f16::from_bits(bits).to_f32() as f64;
            value * value
        })
        .sum::<f64>();
    (sum / values.len() as f64).sqrt().max(f64::from(floor)) as f32
}

fn proposal_specs(checkpoint: &Checkpoint, stream: &CudaStream) -> CudaResult<Vec<ProposalSpec>> {
    let mut specs = Vec::new();
    let mut push = |name: &str,
                    _family: u32,
                    shape: &[usize],
                    layer: Option<usize>,
                    block_len: usize,
                    transpose_columns: usize,
                    floor: f32|
     -> CudaResult<()> {
        let values = checkpoint.tensor(name, shape, layer)?;
        if block_len == 0 || values.len() % block_len != 0 || u32::try_from(block_len).is_err() {
            return Err(format!("invalid CUDA proposal blocks for {name}"));
        }
        let scales = values
            .chunks_exact(block_len)
            .map(|block| rms(block, floor))
            .collect::<Vec<_>>();
        specs.push(ProposalSpec {
            elements: values.len(),
            block_len: block_len as u32,
            scales: DeviceBuffer::from_host(stream, &scales).map_err(cuda_error)?,
            host_scales: scales,
            tensor_id: tensor_key(name),
            logical_start: layer.map_or(0, |layer| layer * values.len()) as u64,
            transpose_columns: transpose_columns as u32,
        });
        Ok(())
    };
    let tied = 512 * VOCAB;
    push(
        "embedding_readout",
        7,
        &[512, VOCAB],
        None,
        tied,
        512,
        1.0e-6,
    )?;
    push("embedding_readout", 7, &[512, VOCAB], None, tied, 0, 1.0e-6)?;
    push("final_norm", 14, &[512], None, 512, 0, 1.0e-6)?;
    for layer in 0..5 {
        push("qkv", 1, &[5, 512, 640], Some(layer), 512 * 640, 0, 1.0e-6)?;
        push(
            "attention_output",
            2,
            &[5, 512, 512],
            Some(layer),
            512 * 512,
            0,
            1.0e-6,
        )?;
        push(
            "router",
            0,
            &[5, 512, 625],
            Some(layer),
            512 * 625,
            0,
            1.0e-6,
        )?;
        push(
            "expert_gate_up",
            3,
            &[5, 626, 512, 432],
            Some(layer),
            512 * 432,
            0,
            1.0e-6,
        )?;
        push(
            "expert_down",
            4,
            &[5, 626, 216, 512],
            Some(layer),
            216 * 512,
            0,
            1.0e-6,
        )?;
        for (prefix, predictor_family, bias_family, control_family, norm_family, floor) in [
            ("mhc_attention", 8, 9, 10, 5, 1.0e-6),
            ("mhc_moe", 11, 12, 13, 6, 1.0e-6),
        ] {
            push(
                &format!("{prefix}_predictor"),
                predictor_family,
                &[5, 2048, 24],
                Some(layer),
                2048 * 24,
                0,
                floor,
            )?;
            push(
                &format!("{prefix}_bias"),
                bias_family,
                &[5, 24],
                Some(layer),
                24,
                0,
                floor,
            )?;
            push(
                &format!("{prefix}_control"),
                control_family,
                &[5, 4],
                Some(layer),
                4,
                0,
                0.02,
            )?;
            push(
                if prefix == "mhc_attention" {
                    "attention_norm"
                } else {
                    "ffn_norm"
                },
                norm_family,
                &[5, 512],
                Some(layer),
                512,
                0,
                floor,
            )?;
        }
    }
    if checkpoint.format == "diffusion-mhc4" {
        push("mask_embed", 15, &[512], None, 512, 0, 1.0e-6)?;
        push("index_query", 16, &[64, 64], None, 4096, 0, 1.0e-6)?;
    }
    Ok(specs)
}
