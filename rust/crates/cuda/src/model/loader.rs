use super::VOCAB;
use super::proposal::{GaussianRandom, ProposalSpec};
use super::weights::{DiffusionWeights, Layer, Mhc4Weights, Site, Weights};
use crate::capture::{CudaResult, cuda_error};
use baux::artifact::json::Value;
use baux::semantics::tensor::key as tensor_key;
use cuda_core::{CudaStream, DeviceBuffer};
use deser::Deserialize;
use half::f16;
use memmap2::MmapOptions;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Header {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct Checkpoint {
    map: memmap2::Mmap,
    start: usize,
    headers: BTreeMap<String, Header>,
    format: &'static str,
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
            Some("baux.pisa-looped-mhc4-rope.v1") | Some("baux.fbt-pisa1-looped-mhc4-rope.v1") => {
                "looped-mhc4"
            }
            Some("baux.pisa-diffusion-mhc4-rope.v1")
            | Some("baux.fbt-pisa1-diffusion-mhc4-rope.v1") => "diffusion-mhc4",
            Some("baux.pisa-mhc4-rope.v1") | Some("baux.fbt-pisa1-mhc4-rope.v1") => "mhc4",
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

    fn tensor(&self, name: &str, shape: &[usize], layer: Option<usize>) -> CudaResult<Vec<f32>> {
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
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect())
    }
}

impl Weights {
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
                router: upload("router", &[5, 512, 64], Some(layer))?,
                gate: upload("expert_gate_up", &[5, 65, 512, 512], Some(layer))?,
                down: upload("expert_down", &[5, 65, 256, 512], Some(layer))?,
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
        })
    }

    pub(super) fn fixture(stream: &CudaStream) -> CudaResult<Self> {
        let upload = |value: f32, len| {
            DeviceBuffer::from_host(stream, &vec![value; len]).map_err(cuda_error)
        };
        let scale_512 = f16::from_bits(0x1800).to_f32();
        let mut layers = Vec::new();
        for _ in 0..5 {
            let site = || -> CudaResult<Site> {
                Ok(Site {
                    predictor: upload(0.0, 2048 * 24)?,
                    bias: upload(0.0, 24)?,
                    control: upload(0.0, 4)?,
                    norm: upload(1.0, 512)?,
                })
            };
            layers.push(Layer {
                qkv: upload(scale_512, 512 * 640)?,
                output: upload(scale_512, 512 * 512)?,
                router: upload(0.0, 512 * 64)?,
                gate: upload(0.0, 65 * 512 * 512)?,
                down: upload(0.0, 65 * 256 * 512)?,
                attention: site()?,
                moe: site()?,
            });
        }
        Ok(Self {
            format: "fixture",
            diffusion: None,
            mhc4: None,
            embedding: upload(1.0, VOCAB * 512)?,
            readout: upload(1.0, 512 * VOCAB)?,
            final_norm: upload(1.0, 512)?,
            layers,
            proposal: Vec::new(),
            gaussian_random: None,
        })
    }
}

fn rms(values: &[f32], floor: f32) -> f32 {
    let sum = values
        .iter()
        .map(|&v| {
            let value = f64::from(v);
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
        push("router", 0, &[5, 512, 64], Some(layer), 512 * 64, 0, 1.0e-6)?;
        push(
            "expert_gate_up",
            3,
            &[5, 65, 512, 512],
            Some(layer),
            512 * 512,
            0,
            1.0e-6,
        )?;
        push(
            "expert_down",
            4,
            &[5, 65, 256, 512],
            Some(layer),
            256 * 512,
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
