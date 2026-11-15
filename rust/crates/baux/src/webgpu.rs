//! Browser WebGPU (`wasm32`) C-ABI bridge and exact `f32` compute kernels.
//!
//! When compiled for `wasm32`, this module links against host-provided
//! JS/WebGPU bridge symbols (`baux_webgpu_*`, see
//! `forest/theme/webgpu_bridge.js`). On CPU and host builds, exact `f32`
//! reference implementations back `dense_apply`, `dense_dist2`, `DenseLinear`,
//! and unit tests.

use ndarray::{Array2, ArrayView2};

use crate::Device;
use crate::error::IndexError;
use crate::hash::normal_metric;
use crate::knn::{check_rows, flatten_f32, pad_k, unpack_search};

/// DLPack device type code for WebGPU (`kDLWebGPU = 15`).
pub const WEBGPU_DLPACK_DEVICE: i32 = 15;

/// Sentinel neighbor index emitted when fewer than `k` valid entries exist.
pub const WEBGPU_SENTINEL_IDX: u32 = u32::MAX;

/// Default variance floor matching `posterior.metal` and `webgpu_bridge.js`.
pub const DEFAULT_EPSILON: f32 = 1e-9;

#[cfg(target_arch = "wasm32")]
unsafe extern "C" {
    /// Returns nonzero (`1`) if WebGPU bridge is active and initialized, `0`
    /// otherwise.
    pub fn baux_webgpu_available() -> i32;

    /// Allocates a WebGPU storage buffer of at least `bytes` bytes; returns
    /// non-zero handle or `0` on failure.
    pub fn baux_webgpu_buffer_alloc(bytes: usize) -> u32;

    /// Frees a WebGPU buffer handle; returns `0` on success, nonzero on error.
    pub fn baux_webgpu_buffer_free(handle: u32) -> i32;

    /// Copies `len` `f32` values from buffer `handle` into `out_ptr`; returns
    /// `0` on success.
    pub fn baux_webgpu_buffer_read_f32(handle: u32, out_ptr: *mut f32, len: usize) -> i32;

    /// Computes `[q, n]` squared Euclidean distances (optionally scaled by
    /// `inv_scale2[d]`); returns `0` on success.
    pub fn baux_webgpu_exact_dist(
        q: usize,
        n: usize,
        d: usize,
        queries: *const f32,
        rows: *const f32,
        inv_scale2: *const f32,
        out_dist2: *mut f32,
    ) -> i32;

    /// Reduces `[q, n]` squared distances to ascending top-`k` distances and
    /// indices after skipping `skip`; returns `0` on success.
    pub fn baux_webgpu_knn_topk(
        q: usize,
        n: usize,
        k: usize,
        skip: usize,
        dist2: *const f32,
        out_dist: *mut f32,
        out_idx: *mut u32,
    ) -> i32;

    /// Evaluates weighted ENN posterior summary `[queries * metrics * 4]`,
    /// selected IDs `[queries, k]`, and weights `[queries, k, metrics]`;
    /// returns `0` on success.
    pub fn baux_webgpu_device_posterior(
        queries: usize,
        n: usize,
        metrics: usize,
        input_k: usize,
        k: usize,
        skip: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        epsilon: f32,
        observation_noise: i32,
        distances: *const f32,
        ids: *const u32,
        outcomes: *const f32,
        variances: *const f32,
        scales: *const f32,
        out_summary: *mut f32,
        out_selected: *mut u32,
        out_weights: *mut f32,
    ) -> i32;

    /// Evaluates tiled `f32` linear projection `Y[m, n] = X[m, k] @ W[n, k]^T +
    /// bias[n]`; returns `0` on success.
    pub fn baux_webgpu_dense_apply(
        m: usize,
        n: usize,
        k: usize,
        x: *const f32,
        w: *const f32,
        bias: *const f32,
        y: *mut f32,
    ) -> i32;

    /// Evaluates pairwise `f32` squared Euclidean distance `||A[q, d] - B[n,
    /// d]||^2`; returns `0` on success.
    pub fn baux_webgpu_dense_dist2(
        q: usize,
        n: usize,
        d: usize,
        a: *const f32,
        b: *const f32,
        out_dist2: *mut f32,
    ) -> i32;

    /// Scores candidates (`mode`: 0=UCB, 1=Thompson, 2=mean), selects per-batch
    /// argmax, and computes optional softmax probabilities; returns `0` on
    /// success.
    pub fn baux_webgpu_weight_select(
        batch: usize,
        candidates: usize,
        mode: u32,
        emit_softmax: i32,
        beta: f32,
        temperature: f32,
        min_prob: f32,
        mu: *const f32,
        se: *const f32,
        z: *const f32,
        out_scores: *mut f32,
        out_probs: *mut f32,
        out_best_idx: *mut u32,
        out_best_score: *mut f32,
    ) -> i32;
}

/// Returns `true` when the runtime WebGPU bridge is active.
#[inline]
pub fn is_available() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        // SAFETY: side-effect-free capability check imported from host bridge.
        unsafe { baux_webgpu_available() != 0 }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

/// RAII handle to a host WebGPU storage buffer.
#[derive(Debug)]
pub struct WebGpuBuffer {
    handle: u32,
    bytes: usize,
}

impl WebGpuBuffer {
    /// Returns the opaque non-zero JS/WebGPU buffer handle.
    pub const fn handle(&self) -> u32 {
        self.handle
    }

    /// Returns the requested allocation size in bytes.
    pub const fn len_bytes(&self) -> usize {
        self.bytes
    }

    /// Reads `len` `f32` elements from this buffer into host/WASM memory.
    pub fn read_f32(&self, len: usize) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0f32; len];
        read_buffer_f32_into(self.handle, &mut out)?;
        Ok(out)
    }
}

impl Drop for WebGpuBuffer {
    fn drop(&mut self) {
        if self.handle != 0 {
            let _ = buffer_free_handle(self.handle);
            self.handle = 0;
        }
    }
}

/// Allocates a WebGPU storage buffer of `bytes` bytes (must be positive).
pub fn buffer_alloc(bytes: usize) -> Result<WebGpuBuffer, String> {
    if bytes == 0 {
        return Err("WebGPU buffer allocation requires bytes > 0".into());
    }
    #[cfg(target_arch = "wasm32")]
    {
        ensure_bridge_active()?;
        // SAFETY: validated positive byte count; host returns non-zero handle
        // or 0.
        let handle = unsafe { baux_webgpu_buffer_alloc(bytes) };
        if handle == 0 {
            return Err(format!("WebGPU buffer_alloc failed for {bytes} bytes"));
        }
        Ok(WebGpuBuffer { handle, bytes })
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = bytes;
        Err("webgpu backend was not compiled into this library".into())
    }
}

/// Frees a raw WebGPU buffer handle allocated by `baux_webgpu_buffer_alloc`.
pub fn buffer_free_handle(handle: u32) -> Result<(), String> {
    if handle == 0 {
        return Err("cannot free null WebGPU buffer handle (0)".into());
    }
    #[cfg(target_arch = "wasm32")]
    {
        // SAFETY: zero return indicates handle freed.
        let rc = unsafe { baux_webgpu_buffer_free(handle) };
        if rc != 0 {
            return Err(format!(
                "baux_webgpu_buffer_free({handle}) returned error {rc}"
            ));
        }
        Ok(())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = handle;
        Err("webgpu backend was not compiled into this library".into())
    }
}

/// Reads `out.len()` `f32` values from raw WebGPU buffer handle into `out`.
pub fn read_buffer_f32_into(handle: u32, out: &mut [f32]) -> Result<(), String> {
    if handle == 0 {
        return Err("invalid null WebGPU buffer handle (0)".into());
    }
    if out.is_empty() {
        return Err("WebGPU buffer read requires len > 0".into());
    }
    #[cfg(target_arch = "wasm32")]
    {
        ensure_bridge_active()?;
        // SAFETY: `out` is a valid non-empty mutable slice of `out.len()` f32
        // values.
        let rc = unsafe { baux_webgpu_buffer_read_f32(handle, out.as_mut_ptr(), out.len()) };
        if rc != 0 {
            return Err(format!(
                "baux_webgpu_buffer_read_f32({handle}, len={}) returned error {rc}",
                out.len()
            ));
        }
        Ok(())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = out;
        Err("webgpu backend was not compiled into this library".into())
    }
}

#[cfg(target_arch = "wasm32")]
fn ensure_bridge_active() -> Result<(), String> {
    if is_available() {
        Ok(())
    } else {
        Err("WebGPU runtime bridge is unavailable or uninitialized".into())
    }
}

fn checked_pos(name: &str, v: usize) -> Result<usize, String> {
    if v == 0 {
        Err(format!("webgpu: {name} must be > 0"))
    } else {
        Ok(v)
    }
}

fn checked_mul2(a: usize, b: usize, label: &str) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("webgpu: dimension overflow in {label}"))
}

fn checked_mul3(a: usize, b: usize, c: usize, label: &str) -> Result<usize, String> {
    a.checked_mul(b)
        .and_then(|ab| ab.checked_mul(c))
        .ok_or_else(|| format!("webgpu: dimension overflow in {label}"))
}

fn check_slice_len<T>(slice: &[T], expected: usize, name: &str) -> Result<(), String> {
    if slice.len() != expected {
        return Err(format!(
            "webgpu: {name} expected length {expected}, got {}",
            slice.len()
        ));
    }
    Ok(())
}

fn resolve_compute_device(device: Device) -> Result<bool, String> {
    match device {
        Device::Cpu => Ok(false),
        Device::WebGpu => {
            #[cfg(target_arch = "wasm32")]
            {
                ensure_bridge_active()?;
                Ok(true)
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                Err("webgpu backend was not compiled into this library".into())
            }
        }
        other => Err(format!(
            "unsupported device '{}' for dense f32 kernel; expected 'cpu' or 'webgpu'",
            other.name()
        )),
    }
}

fn validate_exact_dist(
    q: usize,
    n: usize,
    d: usize,
    queries: &[f32],
    rows: &[f32],
    inv_scale2: Option<&[f32]>,
    q_label: &str,
    r_label: &str,
) -> Result<usize, String> {
    checked_pos("q", q)?;
    checked_pos("n", n)?;
    checked_pos("d", d)?;
    let qd = checked_mul2(q, d, "exact_dist.queries")?;
    let nd = checked_mul2(n, d, "exact_dist.rows")?;
    let qn = checked_mul2(q, n, "exact_dist.out")?;
    check_slice_len(queries, qd, q_label)?;
    check_slice_len(rows, nd, r_label)?;
    if let Some(scale) = inv_scale2 {
        check_slice_len(scale, d, "inv_scale2")?;
    }
    Ok(qn)
}

/// Exact `f32` CPU squared Euclidean distance matching `WGSL_EXACT_DIST`.
pub fn exact_dist_cpu(
    q: usize,
    n: usize,
    d: usize,
    queries: &[f32],
    rows: &[f32],
    inv_scale2: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    let qn = validate_exact_dist(q, n, d, queries, rows, inv_scale2, "queries", "rows")?;
    let mut out = vec![0.0f32; qn];
    for qi in 0..q {
        let q_row = &queries[qi * d..(qi + 1) * d];
        for ni in 0..n {
            let r_row = &rows[ni * d..(ni + 1) * d];
            let mut acc = 0.0f32;
            match inv_scale2 {
                Some(scale) => {
                    for j in 0..d {
                        let diff = q_row[j] - r_row[j];
                        acc = (diff * diff).mul_add(scale[j], acc);
                    }
                }
                None => {
                    for j in 0..d {
                        let diff = q_row[j] - r_row[j];
                        acc = diff.mul_add(diff, acc);
                    }
                }
            }
            out[qi * n + ni] = acc;
        }
    }
    Ok(out)
}

/// Dispatches exact `f32` squared distances to WebGPU (`wasm32`) or CPU.
pub fn exact_dist(
    q: usize,
    n: usize,
    d: usize,
    queries: &[f32],
    rows: &[f32],
    inv_scale2: Option<&[f32]>,
    device: Device,
) -> Result<Vec<f32>, String> {
    if !resolve_compute_device(device)? {
        return exact_dist_cpu(q, n, d, queries, rows, inv_scale2);
    }
    let qn = validate_exact_dist(q, n, d, queries, rows, inv_scale2, "queries", "rows")?;
    #[cfg(target_arch = "wasm32")]
    {
        let mut out = vec![0.0f32; qn];
        let scale_ptr = inv_scale2.map_or(std::ptr::null(), |s| s.as_ptr());
        // SAFETY: validated non-empty slices of exact required lengths.
        let rc = unsafe {
            baux_webgpu_exact_dist(
                q,
                n,
                d,
                queries.as_ptr(),
                rows.as_ptr(),
                scale_ptr,
                out.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!("baux_webgpu_exact_dist returned error {rc}"));
        }
        Ok(out)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = qn;
        unreachable!("resolve_compute_device forbids WebGpu outside wasm32")
    }
}

fn validate_knn_topk(
    q: usize,
    n: usize,
    k: usize,
    skip: usize,
    dist2: &[f32],
) -> Result<usize, String> {
    checked_pos("q", q)?;
    checked_pos("n", n)?;
    checked_pos("k", k)?;
    if skip > n || skip.checked_add(k).is_none_or(|total| total > n) {
        return Err(format!(
            "webgpu: skip ({skip}) + k ({k}) exceeds candidate count n ({n})"
        ));
    }
    let qn = checked_mul2(q, n, "knn_topk.dist2")?;
    let qk = checked_mul2(q, k, "knn_topk.out")?;
    check_slice_len(dist2, qn, "dist2")?;
    Ok(qk)
}

/// Exact `f32` CPU top-k reduction matching `WGSL_KNN_TOPK` tie-breaking.
pub fn knn_topk_cpu(
    q: usize,
    n: usize,
    k: usize,
    skip: usize,
    dist2: &[f32],
) -> Result<(Vec<f32>, Vec<u32>), String> {
    let qk = validate_knn_topk(q, n, k, skip, dist2)?;
    let mut out_dist = vec![0.0f32; qk];
    let mut out_idx = vec![WEBGPU_SENTINEL_IDX; qk];
    let mut pairs = Vec::with_capacity(n);

    for qi in 0..q {
        let row = &dist2[qi * n..(qi + 1) * n];
        pairs.clear();
        for (j, &d) in row.iter().enumerate() {
            if !d.is_nan() {
                pairs.push((d, j as u32));
            }
        }
        pairs.sort_unstable_by(|(da, ia), (db, ib)| da.total_cmp(db).then_with(|| ia.cmp(ib)));
        for slot in 0..k {
            if let Some(&(d, idx)) = pairs.get(skip + slot) {
                let pos = qi * k + slot;
                out_dist[pos] = d;
                out_idx[pos] = idx;
            }
        }
    }
    Ok((out_dist, out_idx))
}

/// Dispatches `f32` top-`k` selection to WebGPU (`wasm32`) or CPU.
pub fn knn_topk(
    q: usize,
    n: usize,
    k: usize,
    skip: usize,
    dist2: &[f32],
    device: Device,
) -> Result<(Vec<f32>, Vec<u32>), String> {
    if !resolve_compute_device(device)? {
        return knn_topk_cpu(q, n, k, skip, dist2);
    }
    let qk = validate_knn_topk(q, n, k, skip, dist2)?;

    #[cfg(target_arch = "wasm32")]
    {
        let mut out_dist = vec![0.0f32; qk];
        let mut out_idx = vec![WEBGPU_SENTINEL_IDX; qk];
        // SAFETY: validated input and preallocated output buffers.
        let rc = unsafe {
            baux_webgpu_knn_topk(
                q,
                n,
                k,
                skip,
                dist2.as_ptr(),
                out_dist.as_mut_ptr(),
                out_idx.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!("baux_webgpu_knn_topk returned error {rc}"));
        }
        Ok((out_dist, out_idx))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = qk;
        unreachable!("resolve_compute_device forbids WebGpu outside wasm32")
    }
}

fn validate_dense_apply(
    m: usize,
    n: usize,
    k: usize,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
) -> Result<usize, String> {
    checked_pos("m", m)?;
    checked_pos("n", n)?;
    checked_pos("k", k)?;
    let mk = checked_mul2(m, k, "dense_apply.x")?;
    let nk = checked_mul2(n, k, "dense_apply.w")?;
    let mn = checked_mul2(m, n, "dense_apply.y")?;
    check_slice_len(x, mk, "x")?;
    check_slice_len(w, nk, "w")?;
    if let Some(b) = bias {
        check_slice_len(b, n, "bias")?;
    }
    Ok(mn)
}

/// Exact `f32` CPU linear layer `Y[m, n] = X[m, k] @ W[n, k]^T + bias[n]`.
pub fn dense_apply_cpu(
    m: usize,
    n: usize,
    k: usize,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    let mn = validate_dense_apply(m, n, k, x, w, bias)?;
    let mut y = vec![0.0f32; mn];
    for row in 0..m {
        let x_row = &x[row * k..(row + 1) * k];
        for col in 0..n {
            let w_row = &w[col * k..(col + 1) * k];
            let mut acc = 0.0f32;
            for j in 0..k {
                acc = x_row[j].mul_add(w_row[j], acc);
            }
            if let Some(b) = bias {
                acc += b[col];
            }
            y[row * n + col] = acc;
        }
    }
    Ok(y)
}

/// Evaluates `Y[m, n] = X[m, k] @ W[n, k]^T + bias[n]` on CPU or WebGPU.
pub fn dense_apply(
    m: usize,
    n: usize,
    k: usize,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    device: Device,
) -> Result<Vec<f32>, String> {
    if !resolve_compute_device(device)? {
        return dense_apply_cpu(m, n, k, x, w, bias);
    }
    let mn = validate_dense_apply(m, n, k, x, w, bias)?;

    #[cfg(target_arch = "wasm32")]
    {
        let mut y = vec![0.0f32; mn];
        let b_ptr = bias.map_or(std::ptr::null(), |b| b.as_ptr());
        // SAFETY: validated non-empty matrices and optional bias vector.
        let rc = unsafe {
            baux_webgpu_dense_apply(m, n, k, x.as_ptr(), w.as_ptr(), b_ptr, y.as_mut_ptr())
        };
        if rc != 0 {
            return Err(format!("baux_webgpu_dense_apply returned error {rc}"));
        }
        Ok(y)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = mn;
        unreachable!("resolve_compute_device forbids WebGpu outside wasm32")
    }
}

/// Evaluates pairwise `f32` squared distances `||A[q, d] - B[n, d]||^2` on CPU
/// or WebGPU.
pub fn dense_dist2(
    q: usize,
    n: usize,
    d: usize,
    a: &[f32],
    b: &[f32],
    device: Device,
) -> Result<Vec<f32>, String> {
    if !resolve_compute_device(device)? {
        return exact_dist_cpu(q, n, d, a, b, None);
    }
    let qn = validate_exact_dist(q, n, d, a, b, None, "a", "b")?;

    #[cfg(target_arch = "wasm32")]
    {
        let mut out = vec![0.0f32; qn];
        // SAFETY: validated non-empty matrices of exact lengths.
        let rc =
            unsafe { baux_webgpu_dense_dist2(q, n, d, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) };
        if rc != 0 {
            return Err(format!("baux_webgpu_dense_dist2 returned error {rc}"));
        }
        Ok(out)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = qn;
        unreachable!("resolve_compute_device forbids WebGpu outside wasm32")
    }
}

/// Parameters controlling candidate acquisition scoring and softmax selection.
#[derive(Debug, Clone, Copy)]
pub struct WeightSelectParams {
    pub mode: u32,
    pub emit_softmax: bool,
    pub beta: f32,
    pub temperature: f32,
    pub min_prob: f32,
}

impl Default for WeightSelectParams {
    fn default() -> Self {
        Self {
            mode: 0,
            emit_softmax: false,
            beta: 1.0,
            temperature: 1.0,
            min_prob: 0.0,
        }
    }
}

/// Output buffers produced by `weight_select`.
#[derive(Debug, Clone)]
pub struct WeightSelectOutput {
    pub scores: Vec<f32>,
    pub probs: Vec<f32>,
    pub best_idx: Vec<u32>,
    pub best_score: Vec<f32>,
}

fn validate_weight_select(
    batch: usize,
    candidates: usize,
    params: WeightSelectParams,
    mu: &[f32],
    se: &[f32],
    z: Option<&[f32]>,
) -> Result<usize, String> {
    checked_pos("batch", batch)?;
    checked_pos("candidates", candidates)?;
    if params.mode > 2 {
        return Err(format!(
            "webgpu: invalid weight_select mode {}",
            params.mode
        ));
    }
    if !(params.temperature.is_finite() && params.temperature > 0.0) {
        return Err("webgpu: temperature must be finite and > 0".into());
    }
    if !params.beta.is_finite() || !params.min_prob.is_finite() {
        return Err("webgpu: beta and min_prob must be finite".into());
    }
    let bc = checked_mul2(batch, candidates, "weight_select.bc")?;
    check_slice_len(mu, bc, "mu")?;
    check_slice_len(se, bc, "se")?;
    if params.mode == 1 {
        let z_slice =
            z.ok_or_else(|| "webgpu: z noise array required for Thompson mode".to_string())?;
        check_slice_len(z_slice, bc, "z")?;
    }
    Ok(bc)
}

/// Exact `f32` CPU acquisition scoring, argmax selection, and softmax matching
/// `WGSL_WEIGHT_SELECT`.
pub fn weight_select_cpu(
    batch: usize,
    candidates: usize,
    params: WeightSelectParams,
    mu: &[f32],
    se: &[f32],
    z: Option<&[f32]>,
) -> Result<WeightSelectOutput, String> {
    let bc = validate_weight_select(batch, candidates, params, mu, se, z)?;

    let mut scores = vec![0.0f32; bc];
    let mut probs = vec![0.0f32; bc];
    let mut best_idx = vec![WEBGPU_SENTINEL_IDX; batch];
    let mut best_score = vec![f32::MIN; batch];
    let inv_temp = 1.0f32 / params.temperature.max(1e-6);

    for b in 0..batch {
        let base = b * candidates;
        let mut b_best_i = WEBGPU_SENTINEL_IDX;
        let mut b_best_s = -f32::MAX;
        for c in 0..candidates {
            let idx = base + c;
            let score = match params.mode {
                0 => params.beta.mul_add(se[idx], mu[idx]),
                1 => z.expect("checked")[idx].mul_add(se[idx], mu[idx]),
                _ => mu[idx],
            };
            scores[idx] = score;
            if !score.is_nan()
                && (b_best_i == WEBGPU_SENTINEL_IDX
                    || score > b_best_s
                    || (score == b_best_s && (c as u32) < b_best_i))
            {
                b_best_s = score;
                b_best_i = c as u32;
            }
        }
        best_idx[b] = b_best_i;
        best_score[b] = b_best_s;
        let max_score = if b_best_i == WEBGPU_SENTINEL_IDX {
            0.0
        } else {
            b_best_s
        };
        if params.emit_softmax {
            let mut sum = 0.0f32;
            for c in 0..candidates {
                let sc = scores[base + c];
                let w = if sc.is_nan() {
                    0.0
                } else {
                    ((sc - max_score) * inv_temp).clamp(-80.0, 0.0).exp()
                };
                probs[base + c] = w;
                sum += w;
            }
            let norm = 1.0f32 / sum.max(1e-20);
            for c in 0..candidates {
                probs[base + c] = (probs[base + c] * norm).max(params.min_prob);
            }
        } else if (b_best_i as usize) < candidates {
            probs[base + b_best_i as usize] = 1.0;
        }
    }

    Ok(WeightSelectOutput {
        scores,
        probs,
        best_idx,
        best_score,
    })
}

/// Evaluates candidate selection on CPU or WebGPU (`wasm32`).
pub fn weight_select(
    batch: usize,
    candidates: usize,
    params: WeightSelectParams,
    mu: &[f32],
    se: &[f32],
    z: Option<&[f32]>,
    device: Device,
) -> Result<WeightSelectOutput, String> {
    if !resolve_compute_device(device)? {
        return weight_select_cpu(batch, candidates, params, mu, se, z);
    }
    let bc = validate_weight_select(batch, candidates, params, mu, se, z)?;

    #[cfg(target_arch = "wasm32")]
    {
        let mut scores = vec![0.0f32; bc];
        let mut probs = vec![0.0f32; bc];
        let mut best_idx = vec![WEBGPU_SENTINEL_IDX; batch];
        let mut best_score = vec![0.0f32; batch];
        let z_ptr = z.map_or(std::ptr::null(), |v| v.as_ptr());
        // SAFETY: validated batch, candidates, input slices, and output
        // allocations.
        let rc = unsafe {
            baux_webgpu_weight_select(
                batch,
                candidates,
                params.mode,
                i32::from(params.emit_softmax),
                params.beta,
                params.temperature,
                params.min_prob,
                mu.as_ptr(),
                se.as_ptr(),
                z_ptr,
                scores.as_mut_ptr(),
                probs.as_mut_ptr(),
                best_idx.as_mut_ptr(),
                best_score.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!("baux_webgpu_weight_select returned error {rc}"));
        }
        Ok(WeightSelectOutput {
            scores,
            probs,
            best_idx,
            best_score,
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = bc;
        unreachable!("resolve_compute_device forbids WebGpu outside wasm32")
    }
}

/// Immutable `f32` linear projection layer executing on CPU or WebGPU
/// (`wasm32`).
#[derive(Debug, Clone)]
pub struct DenseLinear {
    out_features: usize,
    in_features: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    device: Device,
}

impl DenseLinear {
    /// Creates a linear layer `Y = X @ weight^T + bias`.
    /// `weight` has row-major shape `(out_features, in_features)`.
    pub fn new(
        out_features: usize,
        in_features: usize,
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        device: Device,
    ) -> Result<Self, String> {
        checked_pos("out_features", out_features)?;
        checked_pos("in_features", in_features)?;
        let expected_w = checked_mul2(out_features, in_features, "DenseLinear.weight")?;
        check_slice_len(&weight, expected_w, "weight")?;
        if let Some(ref b) = bias {
            check_slice_len(b, out_features, "bias")?;
        }
        resolve_compute_device(device)?;
        Ok(Self {
            out_features,
            in_features,
            weight,
            bias,
            device,
        })
    }

    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    pub const fn device(&self) -> Device {
        self.device
    }

    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    /// Applies the linear projection to a row-major `[m, in_features]` input
    /// matrix.
    pub fn apply(&self, m: usize, x: &[f32]) -> Result<Vec<f32>, String> {
        dense_apply(
            m,
            self.out_features,
            self.in_features,
            x,
            &self.weight,
            self.bias.as_deref(),
            self.device,
        )
    }
}

/// Parameters controlling weighted WebGPU `f32` ENN posterior evaluation.
#[derive(Debug, Clone, Copy)]
pub struct PosteriorKernelSpec {
    pub queries: usize,
    pub n: usize,
    pub metrics: usize,
    pub input_k: usize,
    pub k: usize,
    pub skip: usize,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub epsilon: f32,
    pub observation_noise: bool,
}

/// Immutable WebGPU posterior storage (`f32` means, standard errors, weights,
/// l2, and selected row IDs).
#[derive(Debug, Clone)]
pub struct WebGpuPosterior {
    pub means: Vec<f32>,
    pub errors: Vec<f32>,
    pub weights: Vec<f32>,
    pub l2: Vec<f32>,
    pub indices: Vec<u32>,
    pub queries: usize,
    pub metrics: usize,
    pub k: usize,
}

impl WebGpuPosterior {
    /// Unpacks packed `[mean, se, se2, l2]` summary rows into
    /// `WebGpuPosterior`.
    pub fn from_packed(
        queries: usize,
        metrics: usize,
        k: usize,
        summary: &[f32],
        selected: Vec<u32>,
        weights: Vec<f32>,
    ) -> Result<Self, String> {
        let count = checked_mul2(queries, metrics, "WebGpuPosterior.count")?;
        let summary_len = checked_mul2(count, 4, "WebGpuPosterior.summary")?;
        let qk = checked_mul2(queries, k, "WebGpuPosterior.selected")?;
        let w_len = checked_mul2(qk, metrics, "WebGpuPosterior.weights")?;
        check_slice_len(summary, summary_len, "summary")?;
        check_slice_len(&selected, qk, "selected")?;
        check_slice_len(&weights, w_len, "weights")?;

        let mut means = Vec::with_capacity(count);
        let mut errors = Vec::with_capacity(count);
        let mut l2 = Vec::with_capacity(count);
        for chunk in summary.chunks_exact(4) {
            means.push(chunk[0]);
            errors.push(chunk[1]);
            l2.push(chunk[3]);
        }
        Ok(Self {
            means,
            errors,
            weights,
            l2,
            indices: selected,
            queries,
            metrics,
            k,
        })
    }

    /// Computes shared-seed `f32` posterior function draws matching
    /// `posterior.metal::enn_draw`.
    pub fn draw(&self, seeds: &[u64]) -> Result<Vec<f32>, String> {
        if seeds.is_empty() {
            return Err("draw seeds must be nonempty".into());
        }
        let count = checked_mul2(self.queries, self.metrics, "draw.qm")?;
        let total = checked_mul2(seeds.len(), count, "draw.total")?;
        let mut out = vec![0.0f32; total];
        for (s_idx, &seed) in seeds.iter().enumerate() {
            let s_base = s_idx * count;
            for at in 0..count {
                let q = at / self.metrics;
                let m = at % self.metrics;
                let mut sum = 0.0f32;
                for j in 0..self.k {
                    let row_id = self.indices[q * self.k + j];
                    if row_id == WEBGPU_SENTINEL_IDX {
                        continue;
                    }
                    let w = self.weights[(q * self.k + j) * self.metrics + m];
                    let z = normal_metric(seed, i64::from(row_id), m) as f32;
                    sum = w.mul_add(z, sum);
                }
                let scale = self.errors[at] / self.l2[at].max(1e-12);
                out[s_base + at] = scale.mul_add(sum, self.means[at]);
            }
        }
        Ok(out)
    }
}

/// CPU `f32` weighted posterior matching `WGSL_DEVICE_POSTERIOR` and
/// `posterior.metal`.
pub fn device_posterior_cpu(
    spec: PosteriorKernelSpec,
    distances: &[f32],
    ids: &[u32],
    outcomes: &[f32],
    variances: Option<&[f32]>,
    scales: &[f32],
) -> Result<WebGpuPosterior, String> {
    validate_posterior_inputs(&spec, distances, ids, outcomes, variances, scales)?;
    let count = spec.queries * spec.metrics;
    let mut summary = vec![0.0f32; count * 4];
    let mut selected = vec![WEBGPU_SENTINEL_IDX; spec.queries * spec.k];
    let mut weights = vec![0.0f32; spec.queries * spec.k * spec.metrics];

    for q in 0..spec.queries {
        for m in 0..spec.metrics {
            let scale = scales[m];
            let scale2 = (scale * scale).max(spec.epsilon);
            let mut sum_w = 0.0f32;
            for j in 0..spec.k {
                let at = q * spec.input_k + spec.skip + j;
                let row = ids[at];
                let w_offset = (q * spec.k + j) * spec.metrics + m;
                if m == 0 {
                    selected[q * spec.k + j] = row;
                }
                if row == WEBGPU_SENTINEL_IDX {
                    weights[w_offset] = 0.0;
                } else {
                    let r_idx = row as usize;
                    let var_norm =
                        variances.map_or(0.0, |v| v[r_idx * spec.metrics + m].max(0.0) / scale2);
                    let denom = (spec.epsilon
                        + spec.epistemic_scale * distances[at]
                        + spec.aleatoric_scale
                        + var_norm)
                        .max(spec.epsilon);
                    let w = 1.0f32 / denom;
                    weights[w_offset] = w;
                    sum_w += w;
                }
            }
            let inv_sum = if sum_w > 0.0 {
                1.0f32 / sum_w.max(spec.epsilon)
            } else {
                0.0
            };
            let mut mean = 0.0f32;
            let mut squared = 0.0f32;
            let mut noise_acc = 0.0f32;
            for j in 0..spec.k {
                let at = q * spec.input_k + spec.skip + j;
                let row = ids[at];
                let w_offset = (q * spec.k + j) * spec.metrics + m;
                if row == WEBGPU_SENTINEL_IDX || sum_w <= 0.0 {
                    weights[w_offset] = 0.0;
                } else {
                    let r_idx = row as usize;
                    let wn = weights[w_offset] * inv_sum;
                    weights[w_offset] = wn;
                    mean = wn.mul_add(outcomes[r_idx * spec.metrics + m], mean);
                    squared = wn.mul_add(wn, squared);
                    if spec.observation_noise {
                        let nv = variances
                            .map_or(0.0, |v| v[r_idx * spec.metrics + m].max(0.0) / scale2);
                        noise_acc = wn.mul_add(spec.aleatoric_scale + nv, noise_acc);
                    }
                }
            }
            let epi_term = if sum_w > 0.0 {
                inv_sum
            } else {
                1.0f32 / spec.epsilon
            };
            let se = (epi_term + noise_acc).max(spec.epsilon).sqrt() * scale.abs();
            let s_base = (q * spec.metrics + m) * 4;
            summary[s_base] = mean;
            summary[s_base + 1] = se;
            summary[s_base + 2] = se * se;
            summary[s_base + 3] = squared.sqrt();
        }
    }

    WebGpuPosterior::from_packed(
        spec.queries,
        spec.metrics,
        spec.k,
        &summary,
        selected,
        weights,
    )
}

fn validate_posterior_inputs(
    spec: &PosteriorKernelSpec,
    distances: &[f32],
    ids: &[u32],
    outcomes: &[f32],
    variances: Option<&[f32]>,
    scales: &[f32],
) -> Result<(usize, usize, usize), String> {
    checked_pos("queries", spec.queries)?;
    checked_pos("n", spec.n)?;
    checked_pos("metrics", spec.metrics)?;
    checked_pos("input_k", spec.input_k)?;
    checked_pos("k", spec.k)?;
    if !(spec.epsilon.is_finite() && spec.epsilon > 0.0) {
        return Err("webgpu: epsilon must be finite and > 0".into());
    }
    if spec.skip > spec.input_k
        || spec
            .skip
            .checked_add(spec.k)
            .is_none_or(|v| v > spec.input_k)
    {
        return Err("webgpu: posterior skip + k exceeds input_k".into());
    }
    let qi = checked_mul2(spec.queries, spec.input_k, "posterior.distances")?;
    let nm = checked_mul2(spec.n, spec.metrics, "posterior.outcomes")?;
    let qm = checked_mul2(spec.queries, spec.metrics, "posterior.qm")?;
    let summary_len = checked_mul2(qm, 4, "posterior.summary")?;
    let qk = checked_mul2(spec.queries, spec.k, "posterior.selected")?;
    let qkm = checked_mul3(spec.queries, spec.k, spec.metrics, "posterior.weights")?;
    check_slice_len(distances, qi, "distances")?;
    check_slice_len(ids, qi, "ids")?;
    check_slice_len(outcomes, nm, "outcomes")?;
    if let Some(var) = variances {
        check_slice_len(var, nm, "variances")?;
    }
    check_slice_len(scales, spec.metrics, "scales")?;
    for &id in ids {
        if id != WEBGPU_SENTINEL_IDX && (id as usize) >= spec.n {
            return Err(format!(
                "webgpu: neighbor id {id} out of range (n={})",
                spec.n
            ));
        }
    }
    Ok((summary_len, qk, qkm))
}

/// Dispatches weighted ENN posterior evaluation to WebGPU (`wasm32`) or CPU.
pub fn device_posterior_eval(
    spec: PosteriorKernelSpec,
    distances: &[f32],
    ids: &[u32],
    outcomes: &[f32],
    variances: Option<&[f32]>,
    scales: &[f32],
    device: Device,
) -> Result<WebGpuPosterior, String> {
    if !resolve_compute_device(device)? {
        return device_posterior_cpu(spec, distances, ids, outcomes, variances, scales);
    }
    let (summary_len, qk, qkm) =
        validate_posterior_inputs(&spec, distances, ids, outcomes, variances, scales)?;

    #[cfg(target_arch = "wasm32")]
    {
        let mut summary = vec![0.0f32; summary_len];
        let mut selected = vec![WEBGPU_SENTINEL_IDX; qk];
        let mut weights = vec![0.0f32; qkm];
        let var_ptr = variances.map_or(std::ptr::null(), |v| v.as_ptr());
        // SAFETY: validated shapes and bounds for all input/output buffers.
        let rc = unsafe {
            baux_webgpu_device_posterior(
                spec.queries,
                spec.n,
                spec.metrics,
                spec.input_k,
                spec.k,
                spec.skip,
                spec.epistemic_scale,
                spec.aleatoric_scale,
                spec.epsilon,
                i32::from(spec.observation_noise),
                distances.as_ptr(),
                ids.as_ptr(),
                outcomes.as_ptr(),
                var_ptr,
                scales.as_ptr(),
                summary.as_mut_ptr(),
                selected.as_mut_ptr(),
                weights.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!("baux_webgpu_device_posterior returned error {rc}"));
        }
        WebGpuPosterior::from_packed(
            spec.queries,
            spec.metrics,
            spec.k,
            &summary,
            selected,
            weights,
        )
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (summary_len, qk, qkm);
        unreachable!("resolve_compute_device forbids WebGpu outside wasm32")
    }
}

/// Exact `f32` neighbor index backed by WebGPU (`wasm32`) or CPU `f32` fallback
/// for testing.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) struct WebGpuIndex {
    rows_f32: Vec<f32>,
    keys: Vec<u64>,
    n_rows: usize,
    num_dim: usize,
    device: Device,
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
impl WebGpuIndex {
    /// Builds a WebGPU exact `f32` index when the `wasm32` WebGPU bridge is
    /// active.
    pub(crate) fn new(num_dim: usize, train_scaled: &ArrayView2<f64>) -> Result<Self, IndexError> {
        if !is_available() {
            return Err(IndexError::InvalidParameter(
                "WebGPU index is unavailable; target wasm32 with active webgpu bridge".to_string(),
            ));
        }
        Self::with_device(num_dim, train_scaled, Device::WebGpu)
    }

    fn with_device(
        num_dim: usize,
        train_scaled: &ArrayView2<f64>,
        device: Device,
    ) -> Result<Self, IndexError> {
        check_rows(train_scaled, num_dim)?;
        Ok(Self {
            rows_f32: flatten_f32(train_scaled),
            keys: (0..train_scaled.nrows()).map(|i| i as u64).collect(),
            n_rows: train_scaled.nrows(),
            num_dim,
            device,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.n_rows
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        self.rows_f32
            .len()
            .saturating_mul(std::mem::size_of::<f32>())
            + self.keys.len().saturating_mul(std::mem::size_of::<u64>())
    }

    pub(crate) fn rebuild(&mut self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        check_rows(train_scaled, self.num_dim)?;
        self.rows_f32 = flatten_f32(train_scaled);
        self.n_rows = train_scaled.nrows();
        self.keys = (0..self.n_rows).map(|i| i as u64).collect();
        Ok(())
    }

    pub(crate) fn add(
        &mut self,
        rows_scaled: &ArrayView2<f64>,
        start_key: u64,
    ) -> Result<(), IndexError> {
        check_rows(rows_scaled, self.num_dim)?;
        if rows_scaled.nrows() == 0 {
            return Ok(());
        }
        self.rows_f32.extend(flatten_f32(rows_scaled));
        let added = rows_scaled.nrows();
        self.keys.extend((0..added).map(|i| start_key + i as u64));
        self.n_rows += added;
        Ok(())
    }

    pub(crate) fn search(
        &mut self,
        queries_scaled: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        check_rows(queries_scaled, self.num_dim)?;
        let n_query = queries_scaled.nrows();
        if n_query == 0 || k_eff == 0 || self.n_rows == 0 {
            return Ok(pad_k(
                Array2::zeros((n_query, 0)),
                Array2::zeros((n_query, 0)),
                search_k,
            ));
        }
        let k = k_eff.min(self.n_rows);
        let q_f32 = flatten_f32(queries_scaled);
        let dist2 = exact_dist(
            n_query,
            self.n_rows,
            self.num_dim,
            &q_f32,
            &self.rows_f32,
            None,
            self.device,
        )
        .map_err(IndexError::InvalidParameter)?;
        let (dists, raw_idx) = knn_topk(n_query, self.n_rows, k, 0, &dist2, self.device)
            .map_err(IndexError::InvalidParameter)?;
        let labels: Vec<i64> = raw_idx
            .into_iter()
            .map(|slot| self.keys.get(slot as usize).map_or(-1, |&key| key as i64))
            .collect();
        let (dist2s, idx) = unpack_search(n_query, k, &dists, &labels);
        Ok(pad_k(dist2s, idx, search_k))
    }

    /// Computes raw `(distances, ids)` top-`(skip + k)` search needed by
    /// `device_posterior`.
    pub(crate) fn raw_topk_f32(
        &self,
        queries_f32: &[f32],
        n_query: usize,
        input_k: usize,
    ) -> Result<(Vec<f32>, Vec<u32>), String> {
        let dist2 = exact_dist(
            n_query,
            self.n_rows,
            self.num_dim,
            queries_f32,
            &self.rows_f32,
            None,
            self.device,
        )?;
        knn_topk(n_query, self.n_rows, input_k, 0, &dist2, self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn host_reports_unavailable_and_rejects_alloc() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert!(!is_available());
            assert!(buffer_alloc(16).is_err());
            assert!(buffer_free_handle(1).is_err());
            let mut out = [0.0f32; 4];
            assert!(read_buffer_f32_into(1, &mut out).is_err());
            assert!(dense_apply(1, 1, 1, &[1.0], &[1.0], None, Device::WebGpu).is_err());
        }
        assert!(buffer_alloc(0).is_err());
        assert!(buffer_free_handle(0).is_err());
    }

    #[test]
    fn exact_dist_and_topk_match_ties_and_skip() {
        let queries = vec![0.0f32, 0.0, 2.0, 0.0];
        let rows = vec![1.0f32, 0.0, -1.0, 0.0, 3.0, 0.0];
        let dist2 = exact_dist(2, 3, 2, &queries, &rows, None, Device::Cpu).unwrap();
        assert_eq!(dist2, vec![1.0, 1.0, 9.0, 1.0, 9.0, 1.0]);

        let (d_top, idx_top) = knn_topk(2, 3, 2, 1, &dist2, Device::Cpu).unwrap();
        assert_eq!(d_top, vec![1.0, 9.0, 1.0, 9.0]);
        assert_eq!(idx_top, vec![1, 2, 2, 1]);
    }

    #[test]
    fn dense_linear_and_dist2_f32() {
        let weight = vec![1.0f32, 2.0, 3.0, -1.0, 0.0, 1.0];
        let bias = Some(vec![0.5f32, -0.5]);
        let layer = DenseLinear::new(2, 3, weight.clone(), bias.clone(), Device::Cpu).unwrap();
        let x = vec![2.0f32, 1.0, -1.0, 0.0, 3.0, 2.0];
        let y = layer.apply(2, &x).unwrap();
        let expected = vec![
            2.0 * 1.0 + 1.0 * 2.0 + (-1.0) * 3.0 + 0.5,
            2.0 * (-1.0) + (-1.0) * 1.0 - 0.5,
            1.0 * 0.0 + 2.0 * 3.0 + 3.0 * 2.0 + 0.5,
            2.0 - 0.5,
        ];
        assert_eq!(y, expected);
        assert_eq!(
            dense_apply(2, 2, 3, &x, &weight, bias.as_deref(), Device::Cpu).unwrap(),
            expected
        );

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 0.0, 2.0, 4.0];
        let d2 = dense_dist2(2, 2, 2, &a, &b, Device::Cpu).unwrap();
        assert_eq!(d2, vec![4.0, 5.0, 20.0, 1.0]);
    }

    #[test]
    fn device_posterior_and_weight_select_cpu() {
        let distances = vec![0.25f32, 1.0, 4.0];
        let ids = vec![0u32, 1, 2];
        let outcomes = vec![2.0f32, 4.0, 8.0];
        let scales = vec![1.5f32];
        let spec = PosteriorKernelSpec {
            queries: 1,
            n: 3,
            metrics: 1,
            input_k: 3,
            k: 2,
            skip: 0,
            epistemic_scale: 1.0,
            aleatoric_scale: 0.0,
            epsilon: DEFAULT_EPSILON,
            observation_noise: true,
        };
        let post = device_posterior_eval(
            spec,
            &distances,
            &ids,
            &outcomes,
            None,
            &scales,
            Device::Cpu,
        )
        .unwrap();
        assert_eq!(post.queries, 1);
        assert_eq!(post.metrics, 1);
        assert_eq!(post.indices, vec![0, 1]);
        assert!((post.means[0] - 2.4).abs() < 1e-5);
        let draws = post.draw(&[11, 22]).unwrap();
        assert_eq!(draws.len(), 2);

        let sel = weight_select(
            1,
            3,
            WeightSelectParams {
                mode: 0,
                emit_softmax: true,
                beta: 2.0,
                temperature: 1.0,
                min_prob: 0.01,
            },
            &[1.0, 2.0, 2.0],
            &[0.1, 0.5, 0.5],
            None,
            Device::Cpu,
        )
        .unwrap();
        assert_eq!(sel.best_idx, vec![1]);
        assert!((sel.best_score[0] - 3.0).abs() < 1e-6);
        assert!(sel.probs.iter().all(|&p| p >= 0.01));
    }

    #[test]
    fn webgpu_index_cpu_parity() {
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]];
        let mut idx = WebGpuIndex::with_device(2, &train.view(), Device::Cpu).unwrap();
        idx.add(&array![[3.0, 0.0]].view(), 10).unwrap();
        assert_eq!(idx.len(), 4);
        let (d, ids) = idx.search(&array![[3.1, 0.0]].view(), 2, 2).unwrap();
        assert_eq!(ids[[0, 0]], 10);
        assert_eq!(ids[[0, 1]], 1);
        assert!((d[[0, 0]] - 0.01).abs() < 1e-5);
    }
}
