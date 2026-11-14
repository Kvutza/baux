use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResidualArchitecture {
    Legacy,
    Residual1,
    ProjectedBoundary,
    Hc4,
    Mhc4,
    LoopedMhc4,
    DiffusionMhc4,
    HnetMhc4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LayerStep {
    pub layer: u32,
    pub pass: u32,
    pub execution: u32,
}

const FULL_STACK: [LayerStep; 10] = [
    LayerStep {
        layer: 0,
        pass: 0,
        execution: 0,
    },
    LayerStep {
        layer: 1,
        pass: 0,
        execution: 1,
    },
    LayerStep {
        layer: 2,
        pass: 0,
        execution: 2,
    },
    LayerStep {
        layer: 3,
        pass: 0,
        execution: 3,
    },
    LayerStep {
        layer: 4,
        pass: 0,
        execution: 4,
    },
    LayerStep {
        layer: 0,
        pass: 1,
        execution: 5,
    },
    LayerStep {
        layer: 1,
        pass: 1,
        execution: 6,
    },
    LayerStep {
        layer: 2,
        pass: 1,
        execution: 7,
    },
    LayerStep {
        layer: 3,
        pass: 1,
        execution: 8,
    },
    LayerStep {
        layer: 4,
        pass: 1,
        execution: 9,
    },
];

static SELECTIVE_CORE: std::sync::OnceLock<Vec<LayerStep>> = std::sync::OnceLock::new();

impl ResidualArchitecture {
    const BY_MODEL: [Self; 8] = [
        Self::Legacy,
        Self::Residual1,
        Self::ProjectedBoundary,
        Self::Hc4,
        Self::Mhc4,
        Self::LoopedMhc4,
        Self::DiffusionMhc4,
        Self::HnetMhc4,
    ];

    pub(super) const fn from_model(model: crate::config::ModelArchitecture) -> Self {
        match model {
            crate::config::ModelArchitecture::FbtPisaLegacyV1 => Self::Legacy,
            crate::config::ModelArchitecture::PisaResidual1V1 => Self::Residual1,
            crate::config::ModelArchitecture::FbtPisaProjectedBoundaryV1 => Self::ProjectedBoundary,
            crate::config::ModelArchitecture::PisaHc4V1 => Self::Hc4,
            crate::config::ModelArchitecture::PisaMhc4V1 => Self::Mhc4,
            crate::config::ModelArchitecture::PisaLoopedMhc4V1 => Self::LoopedMhc4,
            crate::config::ModelArchitecture::PisaDiffusionMhc4V1 => Self::DiffusionMhc4,
            crate::config::ModelArchitecture::PisaHnetMhc4V1 => Self::HnetMhc4,
        }
    }

    pub(super) const fn is_multistream(self) -> bool {
        self.kernel_code() > 0
    }

    pub(super) const fn is_mhc4(self) -> bool {
        matches!(self, Self::Mhc4 | Self::HnetMhc4)
    }

    pub(super) const fn is_hnet(self) -> bool {
        matches!(self, Self::HnetMhc4)
    }

    pub(super) const fn parameter_count(self) -> usize {
        if self.is_hnet() {
            MHC_FULLPARAMS - WIDTH as usize * (VOCAB as usize - 256)
                + (3 * WIDTH * WIDTH + WIDTH) as usize
        } else if matches!(self, Self::DiffusionMhc4) {
            MHC_FULLPARAMS + WIDTH as usize + 64 * 64
        } else if self.is_multistream() {
            MHC_FULLPARAMS
        } else {
            FULL_PARAMETERS
        }
    }

    pub(super) const fn vocabulary(self) -> u32 {
        if self.is_hnet() { 256 } else { VOCAB }
    }

    pub(super) const fn kernel_code(self) -> u32 {
        match self {
            Self::Legacy | Self::Residual1 | Self::ProjectedBoundary => 0,
            Self::Hc4 => 1,
            Self::Mhc4 | Self::HnetMhc4 | Self::LoopedMhc4 | Self::DiffusionMhc4 => 2,
        }
    }

    pub(super) const fn feedback_transition(
        self,
        configured: crate::config::FeedbackTransition,
    ) -> crate::config::FeedbackTransition {
        match self {
            Self::Legacy => configured,
            Self::Residual1 => crate::config::FeedbackTransition::Identity,
            Self::ProjectedBoundary => crate::config::FeedbackTransition::ProjectedSigmoid,
            Self::Hc4 | Self::Mhc4 | Self::HnetMhc4 | Self::LoopedMhc4 | Self::DiffusionMhc4 => {
                crate::config::FeedbackTransition::Identity
            }
        }
    }

    pub(super) const fn patch_size(self) -> usize {
        let _ = self;
        1
    }

    pub(super) fn layer_steps(self) -> &'static [LayerStep] {
        match self {
            Self::Mhc4 | Self::HnetMhc4 | Self::LoopedMhc4 | Self::DiffusionMhc4 => SELECTIVE_CORE
                .get_or_init(|| {
                    crate::forward_program::RecurrentCore::selective_fbt()
                        .layer_visits(MODEL_LAYERS as usize, FEEDBACK_PASSES as usize)
                        .expect("built-in FBT recurrence must fit its physical layer stack")
                        .into_iter()
                        .map(|step| LayerStep {
                            layer: step.layer as u32,
                            pass: step.visit as u32,
                            execution: step.execution as u32,
                        })
                        .collect()
                }),
            _ => &FULL_STACK,
        }
    }
}

pub(super) struct Layer0Slices<'a> {
    pub(super) readout: &'a [u16],
    pub(super) qkv: &'a [u16],
    pub(super) output: &'a [u16],
    pub(super) router: &'a [u16],
    pub(super) gate_up: &'a [u16],
    pub(super) down: &'a [u16],
}

fn mean_attn(s: &Layer0Slices<'_>, width: usize, vocab: usize, qkv_w: usize) -> Vec<f32> {
    use crate::context::probe::half;
    let mut mean_e = vec![0.0f32; width];
    for b in 0..vocab {
        let mut sq = 0.0f32;
        for d in 0..width {
            let v = half(s.readout[d * vocab + b]);
            sq = v.mul_add(v, sq);
        }
        let scale = (sq / (width as f32) + 1e-5).sqrt().recip() / (vocab as f32);
        for d in 0..width {
            mean_e[d] += half(s.readout[d * vocab + b]) * scale;
        }
    }
    let mut mean_v = [0.0f32; 64];
    for (c, mv) in mean_v.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for d in 0..width {
            acc = mean_e[d].mul_add(half(s.qkv[d * qkv_w + 576 + c]), acc);
        }
        *mv = acc;
    }
    let mut mean_z = vec![0.0f32; width];
    for h in 0..8 {
        for (c, &v) in mean_v.iter().enumerate() {
            let row_off = (64 * h + c) * width;
            for d in 0..width {
                mean_z[d] = v.mul_add(half(s.output[row_off + d]), mean_z[d]);
            }
        }
    }
    mean_z
}

fn sig(x: f32) -> f32 {
    let e = (-x.abs()).exp();
    (if x >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    })
    .max(1e-20)
}

fn top3_routes(norm_mid: &[f32], router: &[u16], width: usize, routed: usize) -> [(usize, f32); 4] {
    use crate::context::probe::half;
    let mut scores = [f32::NEG_INFINITY; 3];
    let mut idx = [0usize; 3];
    for r in 0..routed {
        let mut s = 0.0f32;
        for d in 0..width {
            s = norm_mid[d].mul_add(half(router[d * routed + r]), s);
        }
        if s > scores[0] {
            scores = [s, scores[0], scores[1]];
            idx = [r, idx[0], idx[1]];
        } else if s > scores[1] {
            scores[2] = scores[1];
            idx[2] = idx[1];
            scores[1] = s;
            idx[1] = r;
        } else if s > scores[2] {
            scores[2] = s;
            idx[2] = r;
        }
    }
    let (w0, w1, w2) = (sig(scores[0]), sig(scores[1]), sig(scores[2]));
    let denom = w0 + w1 + w2;
    [
        (0, 1.0),
        (idx[0] + 1, w0 / denom),
        (idx[1] + 1, w1 / denom),
        (idx[2] + 1, w2 / denom),
    ]
}

fn byte_state(s: &Layer0Slices<'_>, mean_z: &[f32], b: usize, vocab: usize) -> Vec<f32> {
    use crate::context::probe::half;
    let width = WIDTH as usize;
    let exp_w = routing::ROUTED_WIDTH as usize;
    let gu_w = routing::MOE_GATEUP as usize;
    let mut state: Vec<f32> = (0..width)
        .map(|d| half(s.readout[d * vocab + b]) + mean_z[d])
        .collect();
    let sq: f32 = state.iter().map(|&v| v * v).sum();
    let scale = (sq / (width as f32) + 1e-5).sqrt().recip();
    let norm_mid: Vec<f32> = state.iter().map(|&v| v * scale).collect();
    let active = top3_routes(&norm_mid, s.router, width, routing::ROUTED_EXPERTS as usize);
    let mut act = [0.0f32; 256];
    for (exp, weight) in active {
        let gu = &s.gate_up[exp * width * gu_w..(exp + 1) * width * gu_w];
        let dn = &s.down[exp * exp_w * width..(exp + 1) * exp_w * width];
        for m in 0..exp_w {
            let (mut g, mut up) = (0.0f32, 0.0f32);
            for d in 0..width {
                let x = norm_mid[d];
                g = x.mul_add(half(gu[d * gu_w + m]), g);
                up = x.mul_add(half(gu[d * gu_w + exp_w + m]), up);
            }
            act[m] = g * sig(g) * up;
        }
        for (m, &am) in act.iter().enumerate() {
            let a = weight * am;
            for d in 0..width {
                state[d] = a.mul_add(half(dn[m * width + d]), state[d]);
            }
        }
    }
    let norm = state.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut state {
        *x /= norm;
    }
    state
}

pub(super) fn solve_common(s: &Layer0Slices<'_>, width: usize, vocab: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mean_z = mean_attn(s, width, vocab, QKV_WIDTH as usize);
    let cols: Vec<Vec<f32>> = (0..vocab)
        .into_par_iter()
        .map(|b| byte_state(s, &mean_z, b, vocab))
        .collect();
    let mut mat = vec![0.0f32; width * vocab];
    for (b, col) in cols.iter().enumerate() {
        for (d, &val) in col.iter().enumerate() {
            mat[d * vocab + b] = val;
        }
    }
    let (mut w, mut r) = (vec![0.0f32; vocab], vec![1.0f32; vocab]);
    let mut p = r.clone();
    let mut rho: f32 = r.iter().map(|&x| x * x).sum();
    let (mut ep, mut q) = (vec![0.0f32; width], vec![0.0f32; vocab]);
    for _ in 0..24 {
        if rho < 1e-10 {
            break;
        }
        for d in 0..width {
            ep[d] = mat[d * vocab..(d + 1) * vocab]
                .iter()
                .zip(&p)
                .map(|(&a, &b)| a * b)
                .sum();
        }
        for b in 0..vocab {
            q[b] = (0..width).fold(1e-2 * p[b], |acc, d| mat[d * vocab + b].mul_add(ep[d], acc));
        }
        let pq: f32 = p.iter().zip(&q).map(|(&a, &b)| a * b).sum();
        if pq <= 1e-12 {
            break;
        }
        let step = rho / pq;
        for b in 0..vocab {
            w[b] += step * p[b];
            r[b] -= step * q[b];
        }
        let rho_new: f32 = r.iter().map(|&x| x * x).sum();
        for b in 0..vocab {
            p[b] = r[b] + (rho_new / rho) * p[b];
        }
        rho = rho_new;
    }
    let mut u: Vec<f32> = (0..width)
        .map(|d| {
            mat[d * vocab..(d + 1) * vocab]
                .iter()
                .zip(&w)
                .map(|(&a, &b)| a * b)
                .sum()
        })
        .collect();
    let inv = u
        .iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(1e-12)
        .recip();
    for val in &mut u {
        *val *= inv;
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_sequences() {
        let full = ResidualArchitecture::Hc4.layer_steps();
        assert_eq!(full.len(), 10);
        assert_eq!(
            full.iter().map(|step| step.layer).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        );

        for arch in [ResidualArchitecture::Mhc4, ResidualArchitecture::LoopedMhc4] {
            let steps = arch.layer_steps();
            assert_eq!(steps.len(), 7);
            assert_eq!(
                steps.iter().map(|step| step.layer).collect::<Vec<_>>(),
                [0, 1, 2, 1, 2, 3, 4]
            );
            assert_eq!(
                steps.iter().map(|step| step.execution).collect::<Vec<_>>(),
                [0, 1, 2, 3, 4, 5, 6]
            );
        }
    }
}
