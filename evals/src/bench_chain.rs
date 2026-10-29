//! Chain policy architecture and Adjoint Multi-Chain perturbations.

use baux::hash::splitmix64;

/// Perturbation mode for parameter candidate exploration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerturbMode {
    /// Independent rank-1 perturbation per link without interface coupling.
    Uncoupled,
    /// Frechet Adjoint Multi-Chain Lift coupling adjacent links across hidden
    /// interfaces.
    AdjointChain,
}

/// Metadata and parameter layout for a layer link in the chain policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainLink {
    pub in_dim: usize,
    pub out_dim: usize,
    pub offset: usize,
}

impl ChainLink {
    pub fn param_count(&self) -> usize {
        self.in_dim * self.out_dim + self.out_dim
    }
}

/// Running Statistics Normalization (RSNorm) for input observations.
#[derive(Clone, Debug, PartialEq)]
pub struct RSNorm {
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
    pub count: f32,
}

impl RSNorm {
    /// Constructs a new `RSNorm` with identity statistics (`mean = 0`, `var =
    /// 1`).
    pub fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.0; dim],
            var: vec![1.0; dim],
            count: 1e-4,
        }
    }

    /// Updates running mean and variance with a single observation vector.
    pub fn observe(&mut self, obs: &[f32]) {
        assert_eq!(obs.len(), self.mean.len());
        if obs.iter().any(|x| !x.is_finite()) {
            return;
        }
        let n = self.count + 1.0;
        for i in 0..obs.len() {
            let delta = obs[i] - self.mean[i];
            let new_mean = self.mean[i] + delta / n;
            let delta2 = obs[i] - new_mean;
            self.var[i] = (self.var[i] * self.count + delta * delta2) / n;
            self.mean[i] = new_mean;
        }
        self.count = n;
    }

    /// Normalizes `obs` into `out` clamped to `[-5.0, 5.0]`.
    pub fn normalize(&self, obs: &[f32], out: &mut [f32]) {
        for i in 0..obs.len() {
            let std = (self.var[i] + 1e-8).sqrt();
            out[i] = ((obs[i] - self.mean[i]) / std).clamp(-5.0, 5.0);
        }
    }
}

/// Multi-layer feedforward policy with scratch buffers for zero-allocation
/// rollouts.
#[derive(Clone, Debug)]
pub struct ChainPolicy {
    pub links: Vec<ChainLink>,
    pub weights: Vec<f64>,
    pub simba: bool,
    pub rsnorm: RSNorm,
    act_buffers: Vec<Vec<f32>>,
}

impl ChainPolicy {
    /// Constructs a chain policy given layer dimensions.
    pub fn new(layer_dims: &[usize]) -> Self {
        assert!(layer_dims.len() >= 2);
        let mut links = Vec::with_capacity(layer_dims.len() - 1);
        let mut offset = 0;
        let mut act_buffers = Vec::with_capacity(layer_dims.len());
        for &dim in layer_dims {
            act_buffers.push(vec![0.0; dim]);
        }
        for i in 0..layer_dims.len() - 1 {
            let in_dim = layer_dims[i];
            let out_dim = layer_dims[i + 1];
            links.push(ChainLink {
                in_dim,
                out_dim,
                offset,
            });
            offset += in_dim * out_dim + out_dim;
        }
        Self {
            links,
            weights: vec![0.0; offset],
            simba: false,
            rsnorm: RSNorm::new(layer_dims[0]),
            act_buffers,
        }
    }

    /// Constructs a SimBa architecture (`[obs_dim, H, 4H, H, act_dim]`) with
    /// RSNorm and LayerNorm.
    pub fn simba(obs_dim: usize, hidden_dim: usize, act_dim: usize) -> Self {
        let mut policy = Self::new(&[obs_dim, hidden_dim, hidden_dim * 4, hidden_dim, act_dim]);
        policy.simba = true;
        policy
    }

    /// Updates `RSNorm` statistics when `simba` mode is active.
    pub fn observe_norm(&mut self, obs: &[f32]) {
        if self.simba {
            self.rsnorm.observe(obs);
        }
    }

    /// Evaluates feedforward policy in-place without heap allocations or
    /// cross-feature distortion.
    pub fn forward(&mut self, obs: &[f32], act_out: &mut [f32]) {
        if self.simba {
            self.forward_simba(obs, act_out);
        } else {
            self.forward_mlp(obs, act_out);
        }
    }

    fn forward_mlp(&mut self, obs: &[f32], act_out: &mut [f32]) {
        assert_eq!(obs.len(), self.act_buffers[0].len());
        self.act_buffers[0].copy_from_slice(obs);
        let num_links = self.links.len();
        for l in 0..num_links {
            let is_last = l == num_links - 1;
            self.run_link(l, l, l + 1);
            let out_buf = &mut self.act_buffers[l + 1];
            if is_last {
                for v in out_buf.iter_mut() {
                    *v = v.tanh();
                }
            } else {
                for v in out_buf.iter_mut() {
                    *v = *v / (1.0 + (-*v).exp());
                }
            }
        }
        let last_dim = self.links.last().unwrap().out_dim;
        act_out[..last_dim].copy_from_slice(&self.act_buffers[num_links][..last_dim]);
    }

    fn forward_simba(&mut self, obs: &[f32], act_out: &mut [f32]) {
        assert_eq!(self.links.len(), 4);
        assert_eq!(obs.len(), self.act_buffers[0].len());
        self.rsnorm.normalize(obs, &mut self.act_buffers[0]);
        self.run_link(0, 0, 1);
        let h_dim = self.links[0].out_dim;
        for i in 0..h_dim {
            self.act_buffers[3][i] = self.act_buffers[1][i];
        }
        layer_norm(&mut self.act_buffers[3]);
        self.run_link(1, 3, 2);
        for v in self.act_buffers[2].iter_mut() {
            *v = v.max(0.0);
        }
        self.run_link(2, 2, 3);
        for i in 0..h_dim {
            self.act_buffers[3][i] += self.act_buffers[1][i];
        }
        layer_norm(&mut self.act_buffers[3]);
        self.run_link(3, 3, 4);
        let act_dim = self.links[3].out_dim;
        for i in 0..act_dim {
            act_out[i] = self.act_buffers[4][i].tanh();
        }
    }

    fn run_link(&mut self, link_idx: usize, in_idx: usize, out_idx: usize) {
        let link = self.links[link_idx];
        let w_len = link.in_dim * link.out_dim;
        let w_slice = &self.weights[link.offset..link.offset + w_len];
        let b_slice = &self.weights[link.offset + w_len..link.offset + w_len + link.out_dim];
        let w_f32: Vec<f32> = w_slice.iter().map(|&x| x as f32).collect();
        let b_f32: Vec<f32> = b_slice.iter().map(|&x| x as f32).collect();
        if in_idx < out_idx {
            let (left, right) = self.act_buffers.split_at_mut(out_idx);
            linear_layer(&left[in_idx], &w_f32, &b_f32, &mut right[0]);
        } else {
            let (left, right) = self.act_buffers.split_at_mut(in_idx);
            linear_layer(&right[0], &w_f32, &b_f32, &mut left[out_idx]);
        }
    }

    /// Computes and applies perturbed weights from `base_weights`.
    pub fn apply_perturbation(
        &mut self,
        base_weights: &[f64],
        seed: u64,
        scale: f32,
        mode: PerturbMode,
    ) {
        let sign = if scale < 0.0 { -1.0 } else { 1.0 };
        self.apply_signed(base_weights, seed, scale.abs(), sign, mode);
    }

    /// Computes and applies a signed perturbation ray (`sign` in `{-1.0,
    /// +1.0}`) from `base_weights`.
    pub fn apply_signed(
        &mut self,
        base_weights: &[f64],
        seed: u64,
        scale: f32,
        sign: f32,
        mode: PerturbMode,
    ) {
        assert_eq!(self.weights.len(), base_weights.len());
        self.weights.copy_from_slice(base_weights);
        let mag = scale.abs();
        if mag == 0.0 || sign == 0.0 {
            return;
        }
        let mut delta = vec![0.0_f64; self.weights.len()];
        match mode {
            PerturbMode::Uncoupled => pert_uncoupled(&self.links, seed, mag, &mut delta),
            PerturbMode::AdjointChain => {
                pert_adjoint(&self.links, base_weights, seed, mag, &mut delta);
            }
        }
        let dir = if sign < 0.0 { -1.0_f64 } else { 1.0_f64 };
        for (w, d) in self.weights.iter_mut().zip(delta.iter()) {
            *w += dir * *d;
        }
    }
}

fn layer_norm(buf: &mut [f32]) {
    let n = (buf.len() as f32).max(1.0);
    let mean = buf.iter().sum::<f32>() / n;
    let var = buf.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    for x in buf.iter_mut() {
        *x = (*x - mean) * inv_std;
    }
}

unsafe extern "C" {
    fn mjshim_gemv(
        rows: i32,
        cols: i32,
        weights: *const f32,
        input: *const f32,
        bias: *const f32,
        output: *mut f32,
    );
    fn mjshim_gemvt(rows: i32, cols: i32, weights: *const f32, vec: *const f32, output: *mut f32);
}

fn linear_layer(input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32]) {
    unsafe {
        mjshim_gemv(
            output.len() as i32,
            input.len() as i32,
            weights.as_ptr(),
            input.as_ptr(),
            bias.as_ptr(),
            output.as_mut_ptr(),
        );
    }
}

fn matvec_trans(rows: usize, cols: usize, weights: &[f32], vec: &[f32], output: &mut [f32]) {
    unsafe {
        mjshim_gemvt(
            rows as i32,
            cols as i32,
            weights.as_ptr(),
            vec.as_ptr(),
            output.as_mut_ptr(),
        );
    }
}

fn matvec_notrans(rows: usize, cols: usize, weights: &[f32], vec: &[f32], output: &mut [f32]) {
    let zeros = vec![0.0f32; rows];
    linear_layer(vec, weights, &zeros, output);
}

fn stream_seed(seed: u64, layer: usize, tag: u64) -> u64 {
    let mixed = seed
        ^ ((layer as u64)
            .wrapping_add(1)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15))
        ^ tag.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    splitmix64(mixed)
}

fn normal_vector(seed: u64, out: &mut [f32]) {
    let mut s = seed;
    for x in out.iter_mut() {
        s = splitmix64(s);
        let u1 = ((s as f64) + 1.0) / ((u64::MAX as f64) + 2.0);
        s = splitmix64(s);
        let u2 = ((s as f64) + 1.0) / ((u64::MAX as f64) + 2.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        *x = (r * theta.cos()) as f32;
    }
}

struct RankOneLink {
    u: Vec<f32>,
    v: Vec<f32>,
    b: Vec<f32>,
    link_scale: f32,
    b_scale: f32,
}

fn sample_links(links: &[ChainLink], seed: u64, scale: f32) -> Vec<RankOneLink> {
    let mut out = Vec::with_capacity(links.len());
    for (l, link) in links.iter().enumerate() {
        let mut u = vec![0.0; link.out_dim];
        let mut v = vec![0.0; link.in_dim];
        let mut b = vec![0.0; link.out_dim];
        normal_vector(stream_seed(seed, l, 1), &mut u);
        normal_vector(stream_seed(seed, l, 2), &mut v);
        normal_vector(stream_seed(seed, l, 3), &mut b);
        normalize_unit(&mut u);
        normalize_unit(&mut v);
        normalize_unit(&mut b);
        let lscale = scale / (link.in_dim as f32).max(1.0).sqrt();
        out.push(RankOneLink {
            u,
            v,
            b,
            link_scale: lscale * ((link.in_dim * link.out_dim) as f32).sqrt(),
            b_scale: lscale * (link.out_dim as f32).sqrt(),
        });
    }
    out
}

fn pert_uncoupled(links: &[ChainLink], seed: u64, scale: f32, delta: &mut [f64]) {
    let r1 = sample_links(links, seed, scale);
    for (link, r) in links.iter().zip(r1.iter()) {
        let w_offset = link.offset;
        for j in 0..link.out_dim {
            let uj = f64::from(r.link_scale * r.u[j]);
            let row = &mut delta[w_offset + j * link.in_dim..w_offset + (j + 1) * link.in_dim];
            for (dst, &vi) in row.iter_mut().zip(r.v.iter()) {
                *dst = uj * f64::from(vi);
            }
        }
        let b_offset = w_offset + link.in_dim * link.out_dim;
        for j in 0..link.out_dim {
            delta[b_offset + j] = f64::from(r.b_scale * r.b[j]);
        }
    }
}

fn adjoint_vectors(
    links: &[ChainLink],
    l: usize,
    base_weights: &[f64],
    r1: &[RankOneLink],
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let l1 = links[l];
    let r = &r1[l];
    let w1_f64 = &base_weights[l1.offset..l1.offset + l1.in_dim * l1.out_dim];
    let w1: Vec<f32> = w1_f64.iter().map(|&x| x as f32).collect();
    let mut p = vec![0.0; l1.out_dim];
    let mut q = vec![0.0; l1.in_dim];
    matvec_trans(l1.out_dim, l1.in_dim, &w1, &r.u, &mut q);
    for (qi, &vi) in q.iter_mut().zip(r.v.iter()) {
        *qi *= r.link_scale * vi;
    }
    if l > 0 {
        let l0 = links[l - 1];
        let r0 = &r1[l - 1];
        let w0_f64 = &base_weights[l0.offset..l0.offset + l0.in_dim * l0.out_dim];
        let w0: Vec<f32> = w0_f64.iter().map(|&x| x as f32).collect();
        let mut tmp = vec![0.0; l1.in_dim];
        matvec_notrans(l1.in_dim, l0.in_dim, &w0, &r0.v, &mut tmp);
        for i in 0..l1.in_dim {
            q[i] += r0.link_scale * r0.u[i] * tmp[i];
        }
    }
    if l + 1 < links.len() {
        let l2 = links[l + 1];
        let r2 = &r1[l + 1];
        let w2_f64 = &base_weights[l2.offset..l2.offset + l2.in_dim * l2.out_dim];
        let w2: Vec<f32> = w2_f64.iter().map(|&x| x as f32).collect();
        matvec_trans(l2.out_dim, l2.in_dim, &w2, &r2.u, &mut p);
        for (pk, &vk) in p.iter_mut().zip(r2.v.iter()) {
            *pk *= r2.link_scale * vk;
        }
    } else {
        matvec_notrans(l1.out_dim, l1.in_dim, &w1, &q, &mut p);
        let dot_vq: f32 = r.v.iter().zip(q.iter()).map(|(&vi, &qi)| vi * qi).sum();
        for (pk, &uk) in p.iter_mut().zip(r.u.iter()) {
            *pk += r.link_scale * uk * dot_vq;
        }
    }
    if p.iter().map(|&x| x * x).sum::<f32>() <= 1e-12 {
        normal_vector(stream_seed(seed, l, 4), &mut p);
    }
    if q.iter().map(|&x| x * x).sum::<f32>() <= 1e-12 {
        normal_vector(stream_seed(seed, l, 5), &mut q);
    }
    normalize_unit(&mut p);
    normalize_unit(&mut q);
    (p, q)
}

fn pert_adjoint(
    links: &[ChainLink],
    base_weights: &[f64],
    seed: u64,
    scale: f32,
    delta: &mut [f64],
) {
    if links.is_empty() {
        return;
    }
    let r1 = sample_links(links, seed, scale);
    for l in 0..links.len() {
        let l1 = links[l];
        let r = &r1[l];
        let (p, q) = adjoint_vectors(links, l, base_weights, &r1, seed);
        let s = f64::from(r.link_scale * 0.707_106_77);
        for k in 0..l1.out_dim {
            let uk = s * f64::from(r.u[k]);
            let pk = s * f64::from(p[k]);
            let row = &mut delta[l1.offset + k * l1.in_dim..l1.offset + (k + 1) * l1.in_dim];
            for i in 0..l1.in_dim {
                row[i] = uk * f64::from(r.v[i]) + pk * f64::from(q[i]);
            }
        }
        let b_offset = l1.offset + l1.in_dim * l1.out_dim;
        let bs = f64::from(r.b_scale * 0.707_106_77);
        for k in 0..l1.out_dim {
            delta[b_offset + k] = bs * f64::from(r.b[k] + p[k]);
        }
    }
}

fn normalize_unit(v: &mut [f32]) {
    let norm = (v.iter().map(|&x| x * x).sum::<f32>()).sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
