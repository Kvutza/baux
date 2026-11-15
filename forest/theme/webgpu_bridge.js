/**
 * Browser WebGPU f32 Compute Engine for BAUX Dedicated Web Workers.
 *
 * Implements verified WGSL compute kernels matching `baux`'s exact numeric
 * operations:
 *   - exact_dist      : Tiled f32 squared Euclidean distance with optional inv_scale2
 *   - dense_dist2     : Pairwise f32 ||A - B||^2 squared distance
 *   - knn_topk        : Parallel top-k min-distance + u32 index reduction with
 *                       deterministic (dist, idx) tie-breaking
 *   - device_posterior: Weighted Gaussian neighbor posterior matching posterior.metal
 *                       and weighted.rs semantics (summary [mu, se, se2, l2],
 *                       normalized weights, selected neighbor ids)
 *   - dense_apply     : Tiled f32 linear projection Y = X @ W^T + b
 *   - weight_select   : UCB / Thompson scoring + parallel argmax + optional softmax
 *
 * Security invariants:
 *   - Uses `Map` for all dynamic handle/pipeline tables (no bare `obj[k]` lookups).
 *   - Validates matrix dimensions (> 0, overflow-checked, storage-limit-checked).
 *   - Zero dynamic code execution (`eval`, `new Function`).
 *   - Works under strict CSP (`worker-src 'self'`, `script-src 'self' 'wasm-unsafe-eval'`).
 */

"use strict";

(function attachBauxWebGpu(scope) {
  const MAX_I32_BYTES = 0x7ffffffc;
  const DEFAULT_EPSILON = 1e-9;
  const SENTINEL_IDX = 0xffffffff;

  function align4(byteLength) {
    return (byteLength + 3) & ~3;
  }

  function assertPositiveInt(name, value, maxVal) {
    if (!Number.isSafeInteger(value) || value <= 0 || value > maxVal) {
      throw new RangeError(
        `webgpu_bridge: invalid ${name}=${String(value)} (expected 1..${maxVal})`
      );
    }
    return value;
  }

  function assertNonNegativeInt(name, value, maxVal) {
    if (!Number.isSafeInteger(value) || value < 0 || value > maxVal) {
      throw new RangeError(
        `webgpu_bridge: invalid ${name}=${String(value)} (expected 0..${maxVal})`
      );
    }
    return value;
  }

  function assertFiniteNumber(name, value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new RangeError(
        `webgpu_bridge: invalid ${name}=${String(value)} (expected finite number)`
      );
    }
    return value;
  }

  function checkedElementByteSize(a, b, bytesPerElem, maxBindingBytes, label) {
    const count = a * b;
    if (!Number.isSafeInteger(count) || count <= 0) {
      throw new RangeError(`webgpu_bridge: dimension overflow in ${label}`);
    }
    const bytes = count * bytesPerElem;
    if (!Number.isSafeInteger(bytes) || bytes <= 0 || bytes > MAX_I32_BYTES) {
      throw new RangeError(
        `webgpu_bridge: byte size overflow (${bytes}) in ${label}`
      );
    }
    if (bytes > maxBindingBytes) {
      throw new RangeError(
        `webgpu_bridge: ${label} requires ${bytes} bytes, exceeds device maxStorageBufferBindingSize (${maxBindingBytes})`
      );
    }
    return bytes;
  }

  function divCeil(n, d) {
    return Math.ceil(n / d);
  }

  // ---------------------------------------------------------------------------
  // WGSL Kernels (strict WGSL 1.0 uniform control flow, <= 8 storage bindings)
  // ---------------------------------------------------------------------------

  const WGSL_EXACT_DIST = /* wgsl */ `
struct ExactDistParams {
  q: u32,
  n: u32,
  d: u32,
  has_scale: u32,
};

@group(0) @binding(0) var<uniform> params: ExactDistParams;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read> rows: array<f32>;
@group(0) @binding(3) var<storage, read> inv_scale2: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_dist2: array<f32>;

var<workgroup> q_tile: array<array<f32, 16>, 16>;
var<workgroup> r_tile: array<array<f32, 16>, 16>;
var<workgroup> s_tile: array<f32, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let tx = local_id.x;
  let ty = local_id.y;
  let row_q = wg_id.y * 16u + ty;
  let col_n = wg_id.x * 16u + tx;

  var acc: f32 = 0.0;
  var t: u32 = 0u;
  loop {
    if (t >= params.d) {
      break;
    }
    let dim_idx = t + tx;
    let r_row = wg_id.x * 16u + ty;

    q_tile[ty][tx] = select(0.0, queries[row_q * params.d + dim_idx], row_q < params.q && dim_idx < params.d);
    r_tile[ty][tx] = select(0.0, rows[r_row * params.d + dim_idx], r_row < params.n && dim_idx < params.d);
    if (ty == 0u) {
      let use_scale = (params.has_scale != 0u) && (dim_idx < params.d);
      s_tile[tx] = select(1.0, inv_scale2[dim_idx], use_scale);
    }

    workgroupBarrier();

    let tile_len = min(16u, params.d - t);
    if (row_q < params.q && col_n < params.n) {
      for (var k: u32 = 0u; k < tile_len; k = k + 1u) {
        let diff = q_tile[ty][k] - r_tile[tx][k];
        acc = fma(diff * diff, s_tile[k], acc);
      }
    }

    workgroupBarrier();
    t = t + 16u;
  }

  if (row_q < params.q && col_n < params.n) {
    out_dist2[row_q * params.n + col_n] = acc;
  }
}
`;

  const WGSL_KNN_TOPK = /* wgsl */ `
struct KnnTopkParams {
  q: u32,
  n: u32,
  k: u32,
  skip: u32,
};

@group(0) @binding(0) var<uniform> params: KnnTopkParams;
@group(0) @binding(1) var<storage, read> dist2: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_dist: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_idx: array<u32>;

const WG_SIZE: u32 = 256u;
const INVALID_IDX: u32 = 0xffffffffu;
const MAX_F32_BITS: f32 = 3.402823466e+38;

var<workgroup> sh_dist: array<f32, 256>;
var<workgroup> sh_idx: array<u32, 256>;
var<workgroup> sh_prev_dist: f32;
var<workgroup> sh_prev_idx: u32;
var<workgroup> sh_has_prev: u32;

fn is_not_nan(v: f32) -> bool {
  let bits = bitcast<u32>(v) & 0x7fffffffu;
  return bits <= 0x7f800000u;
}

fn pair_before(d_a: f32, i_a: u32, d_b: f32, i_b: u32) -> bool {
  if (i_a == INVALID_IDX) {
    return false;
  }
  if (i_b == INVALID_IDX) {
    return true;
  }
  if (d_a < d_b) {
    return true;
  }
  if (d_a == d_b && i_a < i_b) {
    return true;
  }
  return false;
}

fn pair_after(d: f32, idx: u32, prev_d: f32, prev_idx: u32, has_prev: u32) -> bool {
  if (has_prev == 0u) {
    return true;
  }
  if (d > prev_d) {
    return true;
  }
  if (d == prev_d && idx > prev_idx) {
    return true;
  }
  return false;
}

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let q_row = wg_id.x;
  if (q_row >= params.q) {
    return;
  }
  let tid = local_id.x;
  let row_base = q_row * params.n;
  let total_steps = params.skip + params.k;

  if (tid == 0u) {
    sh_prev_dist = 0.0;
    sh_prev_idx = 0u;
    sh_has_prev = 0u;
  }
  workgroupBarrier();

  var step: u32 = 0u;
  loop {
    if (step >= total_steps) {
      break;
    }

    let prev_d = sh_prev_dist;
    let prev_i = sh_prev_idx;
    let has_prev = sh_has_prev;

    var best_d: f32 = MAX_F32_BITS;
    var best_i: u32 = INVALID_IDX;

    var j: u32 = tid;
    loop {
      if (j >= params.n) {
        break;
      }
      let candidate_d = dist2[row_base + j];
      if (is_not_nan(candidate_d) && pair_after(candidate_d, j, prev_d, prev_i, has_prev)) {
        if (pair_before(candidate_d, j, best_d, best_i)) {
          best_d = candidate_d;
          best_i = j;
        }
      }
      j = j + WG_SIZE;
    }

    sh_dist[tid] = best_d;
    sh_idx[tid] = best_i;
    workgroupBarrier();

    var stride: u32 = WG_SIZE >> 1u;
    loop {
      if (stride == 0u) {
        break;
      }
      if (tid < stride) {
        let other_d = sh_dist[tid + stride];
        let other_i = sh_idx[tid + stride];
        if (pair_before(other_d, other_i, sh_dist[tid], sh_idx[tid])) {
          sh_dist[tid] = other_d;
          sh_idx[tid] = other_i;
        }
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }

    if (tid == 0u) {
      let win_d = sh_dist[0];
      let win_i = sh_idx[0];
      if (step >= params.skip) {
        let out_slot = q_row * params.k + (step - params.skip);
        out_dist[out_slot] = select(0.0, win_d, win_i != INVALID_IDX);
        out_idx[out_slot] = win_i;
      }
      if (win_i != INVALID_IDX) {
        sh_prev_dist = win_d;
        sh_prev_idx = win_i;
        sh_has_prev = 1u;
      } else {
        sh_prev_dist = MAX_F32_BITS;
        sh_prev_idx = INVALID_IDX;
        sh_has_prev = 1u;
      }
    }
    workgroupBarrier();
    step = step + 1u;
  }
}
`;

  // Matches posterior.metal + weighted.rs exact posterior weighting.
  // Packed out_summary: 4 f32s per (q, m): [mu, se, se2, l2] -> stays <= 8 storage bindings!
  const WGSL_DEVICE_POSTERIOR = /* wgsl */ `
struct PosteriorParams {
  queries: u32,
  metrics: u32,
  input_k: u32,
  k: u32,
  skip: u32,
  has_var: u32,
  observation_noise: u32,
  _pad0: u32,
  epistemic_scale: f32,
  aleatoric_scale: f32,
  epsilon: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: PosteriorParams;
@group(0) @binding(1) var<storage, read> distances: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read> outcomes: array<f32>;
@group(0) @binding(4) var<storage, read> variances: array<f32>;
@group(0) @binding(5) var<storage, read> scales: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_summary: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_selected: array<u32>;
@group(0) @binding(8) var<storage, read_write> out_weights: array<f32>;

const INVALID_IDX: u32 = 0xffffffffu;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
  let gid = gid3.x;
  let total = params.queries * params.metrics;
  if (gid >= total) {
    return;
  }

  let q = gid / params.metrics;
  let m = gid - q * params.metrics;
  let scale = scales[m];
  let scale2 = max(scale * scale, params.epsilon);

  var sum_w: f32 = 0.0;
  for (var j: u32 = 0u; j < params.k; j = j + 1u) {
    let at = q * params.input_k + params.skip + j;
    let row = ids[at];
    let w_offset = (q * params.k + j) * params.metrics + m;

    if (m == 0u) {
      out_selected[q * params.k + j] = row;
    }

    if (row == INVALID_IDX) {
      out_weights[w_offset] = 0.0;
    } else {
      var var_norm: f32 = 0.0;
      if (params.has_var != 0u) {
        var_norm = max(variances[row * params.metrics + m], 0.0) / scale2;
      }
      let denom = max(
        params.epsilon + params.epistemic_scale * distances[at] + params.aleatoric_scale + var_norm,
        params.epsilon
      );
      let w = 1.0 / denom;
      out_weights[w_offset] = w;
      sum_w = sum_w + w;
    }
  }

  let inv_sum = select(0.0, 1.0 / max(sum_w, params.epsilon), sum_w > 0.0);
  var mean: f32 = 0.0;
  var squared: f32 = 0.0;
  var noise_acc: f32 = 0.0;

  for (var j: u32 = 0u; j < params.k; j = j + 1u) {
    let at = q * params.input_k + params.skip + j;
    let row = ids[at];
    let w_offset = (q * params.k + j) * params.metrics + m;

    if (row == INVALID_IDX || sum_w <= 0.0) {
      out_weights[w_offset] = 0.0;
    } else {
      let wn = out_weights[w_offset] * inv_sum;
      out_weights[w_offset] = wn;
      mean = fma(wn, outcomes[row * params.metrics + m], mean);
      squared = fma(wn, wn, squared);
      if (params.observation_noise != 0u) {
        var nv: f32 = 0.0;
        if (params.has_var != 0u) {
          nv = max(variances[row * params.metrics + m], 0.0) / scale2;
        }
        noise_acc = fma(wn, params.aleatoric_scale + nv, noise_acc);
      }
    }
  }

  let epi_term = select(1.0 / params.epsilon, inv_sum, sum_w > 0.0);
  let se = sqrt(max(epi_term + noise_acc, params.epsilon)) * abs(scale);
  let se2 = se * se;
  let l2 = sqrt(squared);

  let s_base = gid * 4u;
  out_summary[s_base + 0u] = mean;
  out_summary[s_base + 1u] = se;
  out_summary[s_base + 2u] = se2;
  out_summary[s_base + 3u] = l2;
}
`;

  const WGSL_DENSE_APPLY = /* wgsl */ `
struct DenseApplyParams {
  m: u32,
  n: u32,
  k: u32,
  has_bias: u32,
};

@group(0) @binding(0) var<uniform> params: DenseApplyParams;
@group(0) @binding(1) var<storage, read> x_mat: array<f32>;
@group(0) @binding(2) var<storage, read> w_mat: array<f32>;
@group(0) @binding(3) var<storage, read> bias_vec: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_y: array<f32>;

var<workgroup> x_tile: array<array<f32, 16>, 16>;
var<workgroup> w_tile: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let tx = local_id.x;
  let ty = local_id.y;
  let row = wg_id.y * 16u + ty;
  let col = wg_id.x * 16u + tx;

  var acc: f32 = 0.0;
  var t: u32 = 0u;
  loop {
    if (t >= params.k) {
      break;
    }
    let k_idx = t + tx;
    let w_row = wg_id.x * 16u + ty;

    x_tile[ty][tx] = select(0.0, x_mat[row * params.k + k_idx], row < params.m && k_idx < params.k);
    w_tile[ty][tx] = select(0.0, w_mat[w_row * params.k + k_idx], w_row < params.n && k_idx < params.k);

    workgroupBarrier();

    let tile_len = min(16u, params.k - t);
    if (row < params.m && col < params.n) {
      for (var i: u32 = 0u; i < tile_len; i = i + 1u) {
        acc = fma(x_tile[ty][i], w_tile[tx][i], acc);
      }
    }

    workgroupBarrier();
    t = t + 16u;
  }

  if (row < params.m && col < params.n) {
    let b = select(0.0, bias_vec[col], params.has_bias != 0u);
    out_y[row * params.n + col] = acc + b;
  }
}
`;

  const WGSL_WEIGHT_SELECT = /* wgsl */ `
struct WeightSelectParams {
  batch: u32,
  candidates: u32,
  mode: u32,
  emit_softmax: u32,
  beta: f32,
  temperature: f32,
  min_prob: f32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: WeightSelectParams;
@group(0) @binding(1) var<storage, read> mu: array<f32>;
@group(0) @binding(2) var<storage, read> se: array<f32>;
@group(0) @binding(3) var<storage, read> z_noise: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_scores: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_probs: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_best_idx: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_best_score: array<f32>;

const WG_SIZE: u32 = 256u;
const INVALID_IDX: u32 = 0xffffffffu;
const NEG_INF_FALLBACK: f32 = -3.402823466e+38;

var<workgroup> sh_val: array<f32, 256>;
var<workgroup> sh_idx: array<u32, 256>;
var<workgroup> sh_scalar: f32;

fn is_not_nan(v: f32) -> bool {
  let bits = bitcast<u32>(v) & 0x7fffffffu;
  return bits <= 0x7f800000u;
}

fn better_score(s_a: f32, i_a: u32, s_b: f32, i_b: u32) -> bool {
  if (i_a == INVALID_IDX) {
    return false;
  }
  if (i_b == INVALID_IDX) {
    return true;
  }
  if (s_a > s_b) {
    return true;
  }
  if (s_a == s_b && i_a < i_b) {
    return true;
  }
  return false;
}

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let b = wg_id.x;
  if (b >= params.batch) {
    return;
  }
  let tid = local_id.x;
  let base = b * params.candidates;

  var local_best_s: f32 = NEG_INF_FALLBACK;
  var local_best_i: u32 = INVALID_IDX;

  var c: u32 = tid;
  loop {
    if (c >= params.candidates) {
      break;
    }
    let m_val = mu[base + c];
    let s_val = se[base + c];
    var score: f32 = m_val;
    if (params.mode == 0u) {
      score = fma(params.beta, s_val, m_val);
    } else if (params.mode == 1u) {
      score = fma(z_noise[base + c], s_val, m_val);
    }
    out_scores[base + c] = score;
    if (is_not_nan(score) && better_score(score, c, local_best_s, local_best_i)) {
      local_best_s = score;
      local_best_i = c;
    }
    c = c + WG_SIZE;
  }

  sh_val[tid] = local_best_s;
  sh_idx[tid] = local_best_i;
  workgroupBarrier();

  var stride: u32 = WG_SIZE >> 1u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (tid < stride) {
      let os = sh_val[tid + stride];
      let oi = sh_idx[tid + stride];
      if (better_score(os, oi, sh_val[tid], sh_idx[tid])) {
        sh_val[tid] = os;
        sh_idx[tid] = oi;
      }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (tid == 0u) {
    out_best_idx[b] = sh_idx[0];
    out_best_score[b] = sh_val[0];
    sh_scalar = select(0.0, sh_val[0], sh_idx[0] != INVALID_IDX);
  }
  workgroupBarrier();

  let max_score = sh_scalar;
  let inv_temp = 1.0 / max(params.temperature, 1e-6);
  let do_softmax = params.emit_softmax != 0u;

  var local_exp_sum: f32 = 0.0;
  c = tid;
  loop {
    if (c >= params.candidates) {
      break;
    }
    var w: f32 = 0.0;
    if (do_softmax) {
      let sc = out_scores[base + c];
      if (is_not_nan(sc)) {
        let shifted = clamp((sc - max_score) * inv_temp, -80.0, 0.0);
        w = exp(shifted);
      }
    } else if (c == sh_idx[0]) {
      w = 1.0;
    }
    out_probs[base + c] = w;
    local_exp_sum = local_exp_sum + w;
    c = c + WG_SIZE;
  }

  sh_val[tid] = local_exp_sum;
  workgroupBarrier();

  stride = WG_SIZE >> 1u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (tid < stride) {
      sh_val[tid] = sh_val[tid] + sh_val[tid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (tid == 0u) {
    sh_scalar = max(sh_val[0], 1e-20);
  }
  workgroupBarrier();

  let norm = 1.0 / sh_scalar;
  c = tid;
  loop {
    if (c >= params.candidates) {
      break;
    }
    if (do_softmax) {
      out_probs[base + c] = max(out_probs[base + c] * norm, params.min_prob);
    }
    c = c + WG_SIZE;
  }
}
`;

  // ---------------------------------------------------------------------------
  // Bridge class managing device, handle Map, compiled pipelines, and batches
  // ---------------------------------------------------------------------------

  class BauxWebGpuBridge {
    constructor(adapter, device, adapterInfo) {
      this.adapter = adapter;
      this.device = device;
      this.adapterInfo = adapterInfo || { vendor: "", architecture: "", description: "" };
      this.limits = {
        maxStorageBufferBindingSize:
          (device.limits && device.limits.maxStorageBufferBindingSize) || 134217728,
        maxBufferSize: (device.limits && device.limits.maxBufferSize) || 268435456,
        maxComputeWorkgroupsPerDimension:
          (device.limits && device.limits.maxComputeWorkgroupsPerDimension) || 65535,
      };
      this.buffers = new Map();
      this.pipelines = new Map();
      this.nextHandle = 1;
      this.activeEncoder = null;
      this.pendingCommands = 0;
      this.pendingUniforms = [];
      this.dispatchCount = 0;
      this.lostReason = null;

      // Preallocated 16-byte neutral F32 buffer so optional kernel bindings never
      // require runtime dummy buffer allocations.
      this.dummyF32Handle = this.uploadF32(new Float32Array([1.0, 0.0, 0.0, 0.0]), "dummy_f32");

      if (device && device.lost && typeof device.lost.then === "function") {
        device.lost.then((info) => {
          this.lostReason = (info && info.message) || "WebGPU device lost";
        });
      }
    }

    _ensureAlive() {
      if (this.lostReason !== null) {
        throw new Error(`webgpu_bridge: device unavailable (${this.lostReason})`);
      }
    }

    _getBufferEntry(handle, label) {
      const entry = this.buffers.get(handle);
      if (!entry) {
        throw new Error(`webgpu_bridge: invalid buffer handle ${String(handle)} for ${label}`);
      }
      return entry;
    }

    _getPipeline(name) {
      let pipeline = this.pipelines.get(name);
      if (pipeline) {
        return pipeline;
      }
      let code;
      switch (name) {
        case "exact_dist":
          code = WGSL_EXACT_DIST;
          break;
        case "knn_topk":
          code = WGSL_KNN_TOPK;
          break;
        case "device_posterior":
          code = WGSL_DEVICE_POSTERIOR;
          break;
        case "dense_apply":
          code = WGSL_DENSE_APPLY;
          break;
        case "weight_select":
          code = WGSL_WEIGHT_SELECT;
          break;
        default:
          throw new Error(`webgpu_bridge: unknown compute kernel "${name}"`);
      }
      const module = this.device.createShaderModule({
        label: `baux_${name}_shader`,
        code,
      });
      pipeline = this.device.createComputePipeline({
        label: `baux_${name}_pipeline`,
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      this.pipelines.set(name, pipeline);
      return pipeline;
    }

    precompileAll() {
      this._getPipeline("exact_dist");
      this._getPipeline("knn_topk");
      this._getPipeline("device_posterior");
      this._getPipeline("dense_apply");
      this._getPipeline("weight_select");
    }

    allocBuffer(byteSize, label, extraUsage) {
      this._ensureAlive();
      const aligned = align4(
        assertPositiveInt("byteSize", byteSize, this.limits.maxStorageBufferBindingSize)
      );
      if (aligned > this.limits.maxBufferSize) {
        throw new RangeError(
          `webgpu_bridge: requested ${aligned} bytes exceeds maxBufferSize (${this.limits.maxBufferSize})`
        );
      }
      const usage =
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST |
        (extraUsage || 0);
      const buf = this.device.createBuffer({
        label: label || "baux_storage",
        size: aligned,
        usage,
      });
      const handle = this.nextHandle;
      this.nextHandle += 1;
      this.buffers.set(handle, {
        buf,
        size: aligned,
        usage,
        label: label || "baux_storage",
      });
      return handle;
    }

    uploadF32(data, label) {
      const arr = data instanceof Float32Array ? data : Float32Array.from(data);
      const byteLen = checkedElementByteSize(
        arr.length,
        1,
        4,
        this.limits.maxStorageBufferBindingSize,
        label || "uploadF32"
      );
      const handle = this.allocBuffer(byteLen, label || "baux_f32");
      const entry = this._getBufferEntry(handle, "uploadF32");
      this.device.queue.writeBuffer(entry.buf, 0, arr.buffer, arr.byteOffset, byteLen);
      return handle;
    }

    uploadU32(data, label) {
      const arr = data instanceof Uint32Array ? data : Uint32Array.from(data);
      const byteLen = checkedElementByteSize(
        arr.length,
        1,
        4,
        this.limits.maxStorageBufferBindingSize,
        label || "uploadU32"
      );
      const handle = this.allocBuffer(byteLen, label || "baux_u32");
      const entry = this._getBufferEntry(handle, "uploadU32");
      this.device.queue.writeBuffer(entry.buf, 0, arr.buffer, arr.byteOffset, byteLen);
      return handle;
    }

    writeF32(handle, data, byteOffset) {
      this._ensureAlive();
      const offset = assertNonNegativeInt("byteOffset", byteOffset || 0, MAX_I32_BYTES);
      const entry = this._getBufferEntry(handle, "writeF32");
      const arr = data instanceof Float32Array ? data : Float32Array.from(data);
      if (offset + arr.byteLength > entry.size) {
        throw new RangeError(
          `webgpu_bridge: writeF32 exceeds buffer size (${offset + arr.byteLength} > ${entry.size})`
        );
      }
      this.device.queue.writeBuffer(entry.buf, offset, arr.buffer, arr.byteOffset, arr.byteLength);
    }

    freeBuffer(handle) {
      if (handle === this.dummyF32Handle) {
        return false;
      }
      const entry = this.buffers.get(handle);
      if (!entry) {
        return false;
      }
      this.buffers.delete(handle);
      entry.buf.destroy();
      return true;
    }

    destroyAll() {
      this.flush();
      for (const [handle, entry] of this.buffers.entries()) {
        if (handle !== this.dummyF32Handle) {
          entry.buf.destroy();
          this.buffers.delete(handle);
        }
      }
    }

    getStats() {
      let totalBytes = 0;
      for (const entry of this.buffers.values()) {
        totalBytes += entry.size;
      }
      return {
        bufferCount: this.buffers.size,
        totalBytes,
        dispatchCount: this.dispatchCount,
        pendingCommands: this.pendingCommands,
        adapterInfo: this.adapterInfo,
        limits: this.limits,
      };
    }

    beginBatch() {
      this._ensureAlive();
      if (!this.activeEncoder) {
        this.activeEncoder = this.device.createCommandEncoder({
          label: "baux_compute_batch",
        });
      }
    }

    _retireUniforms(uniforms) {
      if (!uniforms || uniforms.length === 0) {
        return;
      }
      this.device.queue.onSubmittedWorkDone().then(
        () => {
          for (let i = 0; i < uniforms.length; i += 1) {
            uniforms[i].destroy();
          }
        },
        () => {
          for (let i = 0; i < uniforms.length; i += 1) {
            uniforms[i].destroy();
          }
        }
      );
    }

    flush() {
      if (this.activeEncoder) {
        const cmd = this.activeEncoder.finish();
        const uniforms = this.pendingUniforms;
        this.activeEncoder = null;
        this.pendingCommands = 0;
        this.pendingUniforms = [];
        this.device.queue.submit([cmd]);
        this._retireUniforms(uniforms);
      }
    }

    _makeUniformBuffer(u32View, f32OverlayEntries, label) {
      const buf = this.device.createBuffer({
        label: label || "baux_uniform",
        size: u32View.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      if (f32OverlayEntries && f32OverlayEntries.length > 0) {
        const f32View = new Float32Array(u32View.buffer);
        for (let i = 0; i < f32OverlayEntries.length; i += 2) {
          f32View[f32OverlayEntries[i]] = f32OverlayEntries[i + 1];
        }
      }
      this.device.queue.writeBuffer(buf, 0, u32View.buffer, u32View.byteOffset, u32View.byteLength);
      return buf;
    }

    _encodeCompute(pipeline, uniformBuf, storageEntries, wgX, wgY, wgZ, label) {
      this._ensureAlive();
      const maxWg = this.limits.maxComputeWorkgroupsPerDimension;
      assertPositiveInt("workgroups.x", wgX, maxWg);
      assertPositiveInt("workgroups.y", wgY, maxWg);
      assertPositiveInt("workgroups.z", wgZ, maxWg);

      const entries = [{ binding: 0, resource: { buffer: uniformBuf } }];
      for (let i = 0; i < storageEntries.length; i += 1) {
        entries.push({
          binding: i + 1,
          resource: { buffer: storageEntries[i].buf },
        });
      }

      const bindGroup = this.device.createBindGroup({
        label: `${label}_bg`,
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });

      const immediate = this.activeEncoder === null;
      const encoder = immediate
        ? this.device.createCommandEncoder({ label: `${label}_cmd` })
        : this.activeEncoder;

      const pass = encoder.beginComputePass({ label: `${label}_pass` });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(wgX, wgY, wgZ);
      pass.end();

      this.dispatchCount += 1;
      if (immediate) {
        this.device.queue.submit([encoder.finish()]);
        this._retireUniforms([uniformBuf]);
      } else {
        this.pendingCommands += 1;
        this.pendingUniforms.push(uniformBuf);
      }
    }

    _assertMinBytes(entry, requiredBytes, label) {
      if (entry.size < requiredBytes) {
        throw new RangeError(
          `webgpu_bridge: buffer "${entry.label}" too small for ${label}: ${entry.size} < ${requiredBytes}`
        );
      }
    }

    // -------------------------------------------------------------------------
    // Kernel Dispatch Operations (Synchronous Queue Submission)
    // -------------------------------------------------------------------------

    exactDist(desc) {
      const q = assertPositiveInt("q", desc.q, 65536);
      const n = assertPositiveInt("n", desc.n, 1048576);
      const d = assertPositiveInt("d", desc.d, 65536);
      const maxBytes = this.limits.maxStorageBufferBindingSize;

      const qBytes = checkedElementByteSize(q, d, 4, maxBytes, "exact_dist.queries");
      const rBytes = checkedElementByteSize(n, d, 4, maxBytes, "exact_dist.rows");
      const outBytes = checkedElementByteSize(q, n, 4, maxBytes, "exact_dist.outDist2");

      const qEntry = this._getBufferEntry(desc.queries, "exact_dist.queries");
      const rEntry = this._getBufferEntry(desc.rows, "exact_dist.rows");
      const outEntry = this._getBufferEntry(desc.outDist2, "exact_dist.outDist2");
      this._assertMinBytes(qEntry, qBytes, "exact_dist.queries");
      this._assertMinBytes(rEntry, rBytes, "exact_dist.rows");
      this._assertMinBytes(outEntry, outBytes, "exact_dist.outDist2");

      const hasScale = desc.invScale2 && desc.invScale2 !== 0 ? 1 : 0;
      let scaleEntry;
      if (hasScale) {
        const sBytes = checkedElementByteSize(d, 1, 4, maxBytes, "exact_dist.invScale2");
        scaleEntry = this._getBufferEntry(desc.invScale2, "exact_dist.invScale2");
        this._assertMinBytes(scaleEntry, sBytes, "exact_dist.invScale2");
      } else {
        scaleEntry = this._getBufferEntry(this.dummyF32Handle, "dummy");
      }

      const uniform = this._makeUniformBuffer(
        new Uint32Array([q, n, d, hasScale]),
        null,
        "exact_dist_params"
      );
      const pipeline = this._getPipeline("exact_dist");
      this._encodeCompute(
        pipeline,
        uniform,
        [qEntry, rEntry, scaleEntry, outEntry],
        divCeil(n, 16),
        divCeil(q, 16),
        1,
        "exact_dist"
      );
    }

    denseDist2(desc) {
      return this.exactDist({
        q: desc.q,
        n: desc.n,
        d: desc.d,
        queries: desc.a,
        rows: desc.b,
        invScale2: 0,
        outDist2: desc.outDist2,
      });
    }

    knnTopk(desc) {
      const q = assertPositiveInt("q", desc.q, 65536);
      const n = assertPositiveInt("n", desc.n, 1048576);
      const skip = assertNonNegativeInt("skip", desc.skip || 0, n);
      const k = assertPositiveInt("k", desc.k, n);
      if (skip > n) {
        throw new RangeError(`webgpu_bridge: skip (${skip}) exceeds n (${n})`);
      }

      const maxBytes = this.limits.maxStorageBufferBindingSize;
      const inBytes = checkedElementByteSize(q, n, 4, maxBytes, "knn_topk.dist2");
      const outBytes = checkedElementByteSize(q, k, 4, maxBytes, "knn_topk.out");

      const dEntry = this._getBufferEntry(desc.dist2, "knn_topk.dist2");
      const odEntry = this._getBufferEntry(desc.outDist, "knn_topk.outDist");
      const oiEntry = this._getBufferEntry(desc.outIdx, "knn_topk.outIdx");
      this._assertMinBytes(dEntry, inBytes, "knn_topk.dist2");
      this._assertMinBytes(odEntry, outBytes, "knn_topk.outDist");
      this._assertMinBytes(oiEntry, outBytes, "knn_topk.outIdx");

      const uniform = this._makeUniformBuffer(
        new Uint32Array([q, n, k, skip]),
        null,
        "knn_topk_params"
      );
      const pipeline = this._getPipeline("knn_topk");
      this._encodeCompute(
        pipeline,
        uniform,
        [dEntry, odEntry, oiEntry],
        q,
        1,
        1,
        "knn_topk"
      );
    }

    devicePosterior(desc) {
      const queries = assertPositiveInt("queries", desc.queries, 65536);
      const n = assertPositiveInt("n", desc.n, 1048576);
      const metrics = assertPositiveInt("metrics", desc.metrics, 4096);
      const inputK = assertPositiveInt("inputK", desc.inputK, 65536);
      const k = assertPositiveInt("k", desc.k, inputK);
      const skip = assertNonNegativeInt("skip", desc.skip || 0, inputK);
      if (skip + k > inputK) {
        throw new RangeError(
          `webgpu_bridge: skip + k (${skip + k}) exceeds inputK (${inputK})`
        );
      }
      const epistemicScale = assertFiniteNumber("epistemicScale", desc.epistemicScale);
      const aleatoricScale = assertFiniteNumber("aleatoricScale", desc.aleatoricScale);
      const epsilon = assertFiniteNumber(
        "epsilon",
        desc.epsilon !== undefined ? desc.epsilon : DEFAULT_EPSILON
      );
      if (epsilon <= 0) {
        throw new RangeError(`webgpu_bridge: epsilon (${epsilon}) must be > 0`);
      }

      const maxBytes = this.limits.maxStorageBufferBindingSize;
      const distBytes = checkedElementByteSize(queries, inputK, 4, maxBytes, "posterior.distances");
      const outBytes = checkedElementByteSize(n, metrics, 4, maxBytes, "posterior.outcomes");
      const scaleBytes = checkedElementByteSize(metrics, 1, 4, maxBytes, "posterior.scales");
      const qmCount = assertPositiveInt("queries*metrics", queries * metrics, 16777216);
      const summaryBytes = checkedElementByteSize(qmCount, 4, 4, maxBytes, "posterior.outSummary");
      const selBytes = checkedElementByteSize(queries, k, 4, maxBytes, "posterior.outSelected");
      const wBytes = checkedElementByteSize(queries * k, metrics, 4, maxBytes, "posterior.outWeights");

      const distEntry = this._getBufferEntry(desc.distances, "posterior.distances");
      const idsEntry = this._getBufferEntry(desc.ids, "posterior.ids");
      const outcEntry = this._getBufferEntry(desc.outcomes, "posterior.outcomes");
      const scaleEntry = this._getBufferEntry(desc.scales, "posterior.scales");
      const sumEntry = this._getBufferEntry(desc.outSummary, "posterior.outSummary");
      const selEntry = this._getBufferEntry(desc.outSelected, "posterior.outSelected");
      const wEntry = this._getBufferEntry(desc.outWeights, "posterior.outWeights");

      this._assertMinBytes(distEntry, distBytes, "posterior.distances");
      this._assertMinBytes(idsEntry, distBytes, "posterior.ids");
      this._assertMinBytes(outcEntry, outBytes, "posterior.outcomes");
      this._assertMinBytes(scaleEntry, scaleBytes, "posterior.scales");
      this._assertMinBytes(sumEntry, summaryBytes, "posterior.outSummary");
      this._assertMinBytes(selEntry, selBytes, "posterior.outSelected");
      this._assertMinBytes(wEntry, wBytes, "posterior.outWeights");

      const hasVar = desc.variances && desc.variances !== 0 ? 1 : 0;
      const varEntry = hasVar
        ? this._getBufferEntry(desc.variances, "posterior.variances")
        : this._getBufferEntry(this.dummyF32Handle, "dummy");
      if (hasVar) {
        this._assertMinBytes(varEntry, outBytes, "posterior.variances");
      }

      const obsNoise = desc.observationNoise ? 1 : 0;
      const u32 = new Uint32Array([
        queries,
        metrics,
        inputK,
        k,
        skip,
        hasVar,
        obsNoise,
        0,
        0,
        0,
        0,
        0,
      ]);
      const uniform = this._makeUniformBuffer(
        u32,
        [8, epistemicScale, 9, aleatoricScale, 10, epsilon],
        "device_posterior_params"
      );

      const pipeline = this._getPipeline("device_posterior");
      this._encodeCompute(
        pipeline,
        uniform,
        [
          distEntry,
          idsEntry,
          outcEntry,
          varEntry,
          scaleEntry,
          sumEntry,
          selEntry,
          wEntry,
        ],
        divCeil(qmCount, 64),
        1,
        1,
        "device_posterior"
      );
    }

    denseApply(desc) {
      const m = assertPositiveInt("m", desc.m, 65536);
      const n = assertPositiveInt("n", desc.n, 65536);
      const k = assertPositiveInt("k", desc.k, 65536);
      const maxBytes = this.limits.maxStorageBufferBindingSize;

      const xBytes = checkedElementByteSize(m, k, 4, maxBytes, "dense_apply.x");
      const wBytes = checkedElementByteSize(n, k, 4, maxBytes, "dense_apply.w");
      const yBytes = checkedElementByteSize(m, n, 4, maxBytes, "dense_apply.y");

      const xEntry = this._getBufferEntry(desc.x, "dense_apply.x");
      const wEntry = this._getBufferEntry(desc.w, "dense_apply.w");
      const yEntry = this._getBufferEntry(desc.y, "dense_apply.y");
      this._assertMinBytes(xEntry, xBytes, "dense_apply.x");
      this._assertMinBytes(wEntry, wBytes, "dense_apply.w");
      this._assertMinBytes(yEntry, yBytes, "dense_apply.y");

      const hasBias = desc.bias && desc.bias !== 0 ? 1 : 0;
      const bEntry = hasBias
        ? this._getBufferEntry(desc.bias, "dense_apply.bias")
        : this._getBufferEntry(this.dummyF32Handle, "dummy");
      if (hasBias) {
        const bBytes = checkedElementByteSize(n, 1, 4, maxBytes, "dense_apply.bias");
        this._assertMinBytes(bEntry, bBytes, "dense_apply.bias");
      }

      const uniform = this._makeUniformBuffer(
        new Uint32Array([m, n, k, hasBias]),
        null,
        "dense_apply_params"
      );
      const pipeline = this._getPipeline("dense_apply");
      this._encodeCompute(
        pipeline,
        uniform,
        [xEntry, wEntry, bEntry, yEntry],
        divCeil(n, 16),
        divCeil(m, 16),
        1,
        "dense_apply"
      );
    }

    weightSelect(desc) {
      const batch = assertPositiveInt("batch", desc.batch !== undefined ? desc.batch : 1, 65536);
      const candidates = assertPositiveInt("candidates", desc.candidates, 1048576);
      const mode = assertNonNegativeInt("mode", desc.mode || 0, 2);
      const emitSoftmax = desc.emitSoftmax ? 1 : 0;
      const beta = assertFiniteNumber("beta", desc.beta !== undefined ? desc.beta : 1.0);
      const temperature = assertFiniteNumber(
        "temperature",
        desc.temperature !== undefined ? desc.temperature : 1.0
      );
      if (temperature <= 0) {
        throw new RangeError(`webgpu_bridge: temperature (${temperature}) must be > 0`);
      }
      const minProb = assertFiniteNumber(
        "minProb",
        desc.minProb !== undefined ? desc.minProb : 0.0
      );

      const maxBytes = this.limits.maxStorageBufferBindingSize;
      const bcBytes = checkedElementByteSize(batch, candidates, 4, maxBytes, "weight_select.bc");
      const bBytes = checkedElementByteSize(batch, 1, 4, maxBytes, "weight_select.b");

      const muEntry = this._getBufferEntry(desc.mu, "weight_select.mu");
      const seEntry = this._getBufferEntry(desc.se, "weight_select.se");
      const scoresEntry = this._getBufferEntry(desc.outScores, "weight_select.outScores");
      const probsEntry = this._getBufferEntry(desc.outProbs, "weight_select.outProbs");
      const idxEntry = this._getBufferEntry(desc.outBestIdx, "weight_select.outBestIdx");
      const bestEntry = this._getBufferEntry(desc.outBestScore, "weight_select.outBestScore");

      this._assertMinBytes(muEntry, bcBytes, "weight_select.mu");
      this._assertMinBytes(seEntry, bcBytes, "weight_select.se");
      this._assertMinBytes(scoresEntry, bcBytes, "weight_select.outScores");
      this._assertMinBytes(probsEntry, bcBytes, "weight_select.outProbs");
      this._assertMinBytes(idxEntry, bBytes, "weight_select.outBestIdx");
      this._assertMinBytes(bestEntry, bBytes, "weight_select.outBestScore");

      const zEntry =
        mode === 1
          ? this._getBufferEntry(desc.z, "weight_select.z")
          : this._getBufferEntry(this.dummyF32Handle, "dummy");
      if (mode === 1) {
        this._assertMinBytes(zEntry, bcBytes, "weight_select.z");
      }

      const uniform = this._makeUniformBuffer(
        new Uint32Array([batch, candidates, mode, emitSoftmax, 0, 0, 0, 0]),
        [4, beta, 5, temperature, 6, minProb],
        "weight_select_params"
      );
      const pipeline = this._getPipeline("weight_select");
      this._encodeCompute(
        pipeline,
        uniform,
        [muEntry, seEntry, zEntry, scoresEntry, probsEntry, idxEntry, bestEntry],
        batch,
        1,
        1,
        "weight_select"
      );
    }

    dispatchOp(opName, desc) {
      switch (opName) {
        case "exact_dist":
          return this.exactDist(desc);
        case "dense_dist2":
          return this.denseDist2(desc);
        case "knn_topk":
          return this.knnTopk(desc);
        case "device_posterior":
          return this.devicePosterior(desc);
        case "dense_apply":
          return this.denseApply(desc);
        case "weight_select":
          return this.weightSelect(desc);
        default:
          throw new Error(`webgpu_bridge: unsupported op "${String(opName)}"`);
      }
    }

    // -------------------------------------------------------------------------
    // Async Staging Readback
    // -------------------------------------------------------------------------

    async _readBufferBytes(handle, elemCount, elemBytes) {
      this._ensureAlive();
      this.flush();
      const entry = this._getBufferEntry(handle, "readBuffer");
      const byteSize =
        elemCount !== undefined
          ? align4(assertPositiveInt("count", elemCount * elemBytes, entry.size))
          : entry.size;
      if (byteSize > entry.size) {
        throw new RangeError(`webgpu_bridge: readback (${byteSize}) > buffer (${entry.size})`);
      }

      const staging = this.device.createBuffer({
        label: "baux_staging_read",
        size: byteSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const encoder = this.device.createCommandEncoder({ label: "baux_copy_readback" });
      encoder.copyBufferToBuffer(entry.buf, 0, staging, 0, byteSize);
      this.device.queue.submit([encoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ, 0, byteSize);
      const mapped = staging.getMappedRange(0, byteSize);
      const copy = new Uint8Array(byteSize);
      copy.set(new Uint8Array(mapped));
      staging.unmap();
      staging.destroy();
      return copy.buffer;
    }

    async readBufferF32(handle, elemCount) {
      const buf = await this._readBufferBytes(handle, elemCount, 4);
      return new Float32Array(buf);
    }

    async readBufferU32(handle, elemCount) {
      const buf = await this._readBufferBytes(handle, elemCount, 4);
      return new Uint32Array(buf);
    }

    async runExactDist(queriesF32, rowsF32, q, n, d, invScale2F32) {
      const handles = [];
      try {
        const hQ = this.uploadF32(queriesF32, "run_exact_dist_q");
        handles.push(hQ);
        const hR = this.uploadF32(rowsF32, "run_exact_dist_r");
        handles.push(hR);
        let hS = 0;
        if (invScale2F32) {
          hS = this.uploadF32(invScale2F32, "run_exact_dist_s");
          handles.push(hS);
        }
        const hOut = this.allocBuffer(q * n * 4, "run_exact_dist_out");
        handles.push(hOut);
        this.exactDist({ q, n, d, queries: hQ, rows: hR, invScale2: hS, outDist2: hOut });
        return await this.readBufferF32(hOut, q * n);
      } finally {
        for (let i = 0; i < handles.length; i += 1) {
          this.freeBuffer(handles[i]);
        }
      }
    }

    async runKnnTopk(dist2F32, q, n, k, skip) {
      const handles = [];
      try {
        const hD = this.uploadF32(dist2F32, "run_knn_dist2");
        handles.push(hD);
        const hOd = this.allocBuffer(q * k * 4, "run_knn_out_d");
        handles.push(hOd);
        const hOi = this.allocBuffer(q * k * 4, "run_knn_out_i");
        handles.push(hOi);
        this.knnTopk({ q, n, k, skip: skip || 0, dist2: hD, outDist: hOd, outIdx: hOi });
        const dist = await this.readBufferF32(hOd, q * k);
        const idx = await this.readBufferU32(hOi, q * k);
        return { dist, idx };
      } finally {
        for (let i = 0; i < handles.length; i += 1) {
          this.freeBuffer(handles[i]);
        }
      }
    }

    async runDevicePosterior(opts) {
      const handles = [];
      try {
        const hDist = this.uploadF32(opts.distances, "run_post_dist");
        handles.push(hDist);
        const hIds = this.uploadU32(opts.ids, "run_post_ids");
        handles.push(hIds);
        const hOutc = this.uploadF32(opts.outcomes, "run_post_outcomes");
        handles.push(hOutc);
        const hScales = this.uploadF32(opts.scales, "run_post_scales");
        handles.push(hScales);
        let hVar = 0;
        if (opts.variances) {
          hVar = this.uploadF32(opts.variances, "run_post_var");
          handles.push(hVar);
        }
        const hSum = this.allocBuffer(opts.queries * opts.metrics * 16, "run_post_sum");
        handles.push(hSum);
        const hSel = this.allocBuffer(opts.queries * opts.k * 4, "run_post_sel");
        handles.push(hSel);
        const hW = this.allocBuffer(opts.queries * opts.k * opts.metrics * 4, "run_post_w");
        handles.push(hW);

        this.devicePosterior({
          queries: opts.queries,
          n: opts.n,
          metrics: opts.metrics,
          inputK: opts.inputK,
          k: opts.k,
          skip: opts.skip || 0,
          epistemicScale: opts.epistemicScale,
          aleatoricScale: opts.aleatoricScale,
          epsilon: opts.epsilon,
          observationNoise: Boolean(opts.observationNoise),
          distances: hDist,
          ids: hIds,
          outcomes: hOutc,
          variances: hVar,
          scales: hScales,
          outSummary: hSum,
          outSelected: hSel,
          outWeights: hW,
        });

        const summary = await this.readBufferF32(hSum, opts.queries * opts.metrics * 4);
        const selected = await this.readBufferU32(hSel, opts.queries * opts.k);
        const weights = await this.readBufferF32(hW, opts.queries * opts.k * opts.metrics);
        return { summary, selected, weights };
      } finally {
        for (let i = 0; i < handles.length; i += 1) {
          this.freeBuffer(handles[i]);
        }
      }
    }

    async runDenseDist2(aF32, bF32, q, n, d) {
      return await this.runExactDist(aF32, bF32, q, n, d, null);
    }

    async runDenseApply(xF32, wF32, m, n, k, biasF32) {
      const handles = [];
      try {
        const hX = this.uploadF32(xF32, "run_dense_apply_x");
        handles.push(hX);
        const hW = this.uploadF32(wF32, "run_dense_apply_w");
        handles.push(hW);
        let hB = 0;
        if (biasF32) {
          hB = this.uploadF32(biasF32, "run_dense_apply_b");
          handles.push(hB);
        }
        const hY = this.allocBuffer(m * n * 4, "run_dense_apply_y");
        handles.push(hY);
        this.denseApply({ m, n, k, x: hX, w: hW, bias: hB, y: hY });
        return await this.readBufferF32(hY, m * n);
      } finally {
        for (let i = 0; i < handles.length; i += 1) {
          this.freeBuffer(handles[i]);
        }
      }
    }

    async runWeightSelect(opts) {
      const batch = opts.batch !== undefined ? opts.batch : 1;
      const candidates = opts.candidates;
      const mode = opts.mode || 0;
      const handles = [];
      try {
        const hMu = this.uploadF32(opts.mu, "run_ws_mu");
        handles.push(hMu);
        const hSe = this.uploadF32(opts.se, "run_ws_se");
        handles.push(hSe);
        let hZ = 0;
        if (mode === 1 && opts.z) {
          hZ = this.uploadF32(opts.z, "run_ws_z");
          handles.push(hZ);
        }
        const bcBytes = batch * candidates * 4;
        const bBytes = batch * 4;
        const hScores = this.allocBuffer(bcBytes, "run_ws_scores");
        handles.push(hScores);
        const hProbs = this.allocBuffer(bcBytes, "run_ws_probs");
        handles.push(hProbs);
        const hBestIdx = this.allocBuffer(bBytes, "run_ws_best_idx");
        handles.push(hBestIdx);
        const hBestScore = this.allocBuffer(bBytes, "run_ws_best_score");
        handles.push(hBestScore);

        this.weightSelect({
          batch,
          candidates,
          mode,
          emitSoftmax: Boolean(opts.emitSoftmax),
          beta: opts.beta,
          temperature: opts.temperature,
          minProb: opts.minProb,
          mu: hMu,
          se: hSe,
          z: hZ,
          outScores: hScores,
          outProbs: hProbs,
          outBestIdx: hBestIdx,
          outBestScore: hBestScore,
        });

        const [scores, probs, bestIdx, bestScore] = await Promise.all([
          this.readBufferF32(hScores, batch * candidates),
          this.readBufferF32(hProbs, batch * candidates),
          this.readBufferU32(hBestIdx, batch),
          this.readBufferF32(hBestScore, batch),
        ]);
        return { scores, probs, bestIdx, bestScore };
      } finally {
        for (let i = 0; i < handles.length; i += 1) {
          this.freeBuffer(handles[i]);
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // High-level End-to-End Convenience & Emscripten Module.bauxWebGpu Wrapper
  // ---------------------------------------------------------------------------

  function createCModuleWrapper(bridge, moduleObj) {
    function viewF32(ptr, len) {
      if (!moduleObj || !moduleObj.HEAPF32) {
        throw new Error("webgpu_bridge: Emscripten Module.HEAPF32 unavailable");
      }
      const byteOffset = ptr >>> 0;
      if ((byteOffset & 3) !== 0) {
        throw new RangeError(`webgpu_bridge: unaligned f32 pointer ${byteOffset}`);
      }
      return new Float32Array(moduleObj.HEAPF32.buffer, byteOffset, len);
    }
    function viewU32(ptr, len) {
      if (!moduleObj || !moduleObj.HEAPU32) {
        throw new Error("webgpu_bridge: Emscripten Module.HEAPU32 unavailable");
      }
      const byteOffset = ptr >>> 0;
      if ((byteOffset & 3) !== 0) {
        throw new RangeError(`webgpu_bridge: unaligned u32 pointer ${byteOffset}`);
      }
      return new Uint32Array(moduleObj.HEAPU32.buffer, byteOffset, len);
    }

    return {
      ready: true,
      sentinelIdx: SENTINEL_IDX,
      caps: () => bridge.getStats(),
      beginBatch: () => bridge.beginBatch(),
      flush: () => bridge.flush(),
      allocBytes: (bytes) => bridge.allocBuffer(bytes, "wasm_alloc"),
      free: (handle) => bridge.freeBuffer(handle),
      destroyAll: () => bridge.destroyAll(),

      uploadF32Ptr: (ptr, len) => bridge.uploadF32(viewF32(ptr, len), "wasm_f32"),
      uploadU32Ptr: (ptr, len) => bridge.uploadU32(viewU32(ptr, len), "wasm_u32"),

      async readF32Ptr(handle, ptr, len) {
        const out = await bridge.readBufferF32(handle, len);
        viewF32(ptr, len).set(out);
        return out.length;
      },

      async readU32Ptr(handle, ptr, len) {
        const out = await bridge.readBufferU32(handle, len);
        viewU32(ptr, len).set(out);
        return out.length;
      },

      exactDist: (q, n, d, queries, rows, invScale2, outDist2) =>
        bridge.exactDist({ q, n, d, queries, rows, invScale2, outDist2 }),

      denseDist2: (q, n, d, a, b, outDist2) =>
        bridge.denseDist2({ q, n, d, a, b, outDist2 }),

      knnTopk: (q, n, k, skip, dist2, outDist, outIdx) =>
        bridge.knnTopk({ q, n, k, skip, dist2, outDist, outIdx }),

      devicePosterior: (
        queries,
        n,
        metrics,
        inputK,
        k,
        skip,
        epistemicScale,
        aleatoricScale,
        epsilon,
        observationNoise,
        distances,
        ids,
        outcomes,
        variances,
        scales,
        outSummary,
        outSelected,
        outWeights
      ) =>
        bridge.devicePosterior({
          queries,
          n,
          metrics,
          inputK,
          k,
          skip,
          epistemicScale,
          aleatoricScale,
          epsilon,
          observationNoise: Boolean(observationNoise),
          distances,
          ids,
          outcomes,
          variances,
          scales,
          outSummary,
          outSelected,
          outWeights,
        }),

      denseApply: (m, n, k, x, w, bias, y) =>
        bridge.denseApply({ m, n, k, x, w, bias, y }),

      weightSelect: (
        batch,
        candidates,
        mode,
        emitSoftmax,
        beta,
        temperature,
        minProb,
        mu,
        se,
        z,
        outScores,
        outProbs,
        outBestIdx,
        outBestScore
      ) =>
        bridge.weightSelect({
          batch,
          candidates,
          mode,
          emitSoftmax: Boolean(emitSoftmax),
          beta,
          temperature,
          minProb,
          mu,
          se,
          z,
          outScores,
          outProbs,
          outBestIdx,
          outBestScore,
        }),
    };
  }

  async function init(options) {
    const opts = options || {};
    const gpu =
      typeof navigator !== "undefined" && navigator && navigator.gpu
        ? navigator.gpu
        : null;
    if (!gpu || typeof gpu.requestAdapter !== "function") {
      return {
        available: false,
        reason: "navigator.gpu is unavailable in this browser/worker context",
        bridge: null,
        adapterInfo: null,
      };
    }

    let adapter = null;
    try {
      adapter = await gpu.requestAdapter({
        powerPreference: opts.powerPreference || "high-performance",
      });
    } catch (err) {
      return {
        available: false,
        reason: `requestAdapter failed: ${String(err && err.message ? err.message : err)}`,
        bridge: null,
        adapterInfo: null,
      };
    }
    if (!adapter) {
      return {
        available: false,
        reason: "no WebGPU GPUAdapter available",
        bridge: null,
        adapterInfo: null,
      };
    }

    let adapterInfo = { vendor: "", architecture: "", description: "", device: "" };
    try {
      if (adapter.info) {
        adapterInfo = {
          vendor: String(adapter.info.vendor || ""),
          architecture: String(adapter.info.architecture || ""),
          device: String(adapter.info.device || ""),
          description: String(adapter.info.description || ""),
        };
      } else if (typeof adapter.requestAdapterInfo === "function") {
        const info = await adapter.requestAdapterInfo();
        adapterInfo = {
          vendor: String((info && info.vendor) || ""),
          architecture: String((info && info.architecture) || ""),
          device: String((info && info.device) || ""),
          description: String((info && info.description) || ""),
        };
      }
    } catch (_) {
      // Adapter info is informational; continue initialization cleanly.
    }

    let device = null;
    try {
      device = await adapter.requestDevice();
    } catch (err) {
      return {
        available: false,
        reason: `requestDevice failed: ${String(err && err.message ? err.message : err)}`,
        bridge: null,
        adapterInfo,
      };
    }

    const bridge = new BauxWebGpuBridge(adapter, device, adapterInfo);
    if (opts.eagerCompile !== false) {
      try {
        bridge.precompileAll();
      } catch (err) {
        return {
          available: false,
          reason: `WGSL compilation failed: ${String(err && err.message ? err.message : err)}`,
          bridge: null,
          adapterInfo,
        };
      }
    }

    return {
      available: true,
      reason: null,
      bridge,
      adapterInfo,
      limits: bridge.limits,
    };
  }

  function formatAdapterLabel(adapterInfo) {
    if (!adapterInfo || typeof adapterInfo !== "object") {
      return "";
    }
    const candidates = [
      adapterInfo.description,
      adapterInfo.device,
      adapterInfo.vendor,
      adapterInfo.architecture,
    ];
    const parts = [];
    for (let i = 0; i < candidates.length; i += 1) {
      const s = String(candidates[i] || "").trim();
      if (s && !parts.includes(s)) {
        parts.push(s);
      }
      if (parts.length >= 2) {
        break;
      }
    }
    return parts.join(" · ");
  }

  // Synchronous SharedArrayBuffer + Atomics RPC control header layout shared
  // between repl_worker.js (WASM C-ABI caller) and webgpu_worker.js (GPU host).
  const SAB_RPC = Object.freeze({
    HEADER_BYTES: 256,
    HEADER_I32_LEN: 64,
    DEFAULT_STAGING_BYTES: 16 * 1024 * 1024,
    MAX_STAGING_BYTES: 128 * 1024 * 1024,
    // Control header word indices (0..63, 4 bytes each)
    IDX_STATE: 0,
    IDX_SEQ: 1,
    IDX_OP: 2,
    IDX_RESULT_I32: 3,
    IDX_RESULT_U32: 4,
    IDX_P0: 5,
    IDX_P1: 6,
    IDX_P2: 7,
    IDX_P3: 8,
    IDX_P4: 9,
    IDX_P5: 10,
    IDX_P6: 11,
    IDX_P7: 12,
    IDX_F0: 16,
    IDX_F1: 17,
    IDX_F2: 18,
    IDX_F3: 19,
    IDX_OFF0: 24,
    IDX_OFF1: 25,
    IDX_OFF2: 26,
    IDX_OFF3: 27,
    IDX_OFF4: 28,
    IDX_OFF5: 29,
    IDX_OFF6: 30,
    IDX_OFF7: 31,
    IDX_LEN0: 32,
    IDX_LEN1: 33,
    IDX_LEN2: 34,
    IDX_LEN3: 35,
    IDX_LEN4: 36,
    IDX_LEN5: 37,
    IDX_LEN6: 38,
    IDX_LEN7: 39,
    IDX_STAGING_BYTES: 40,
    IDX_ERR_CODE: 41,
    // Request state machine
    STATE_IDLE: 0,
    STATE_PENDING: 1,
    STATE_DONE: 2,
    STATE_ERROR: 3,
    // Opcodes mapping 1:1 to rust/crates/baux/src/webgpu.rs C ABI
    OP_BUFFER_ALLOC: 1,
    OP_BUFFER_FREE: 2,
    OP_BUFFER_READ_F32: 3,
    OP_EXACT_DIST: 4,
    OP_KNN_TOPK: 5,
    OP_DEVICE_POSTERIOR: 6,
    OP_DENSE_APPLY: 7,
    OP_DENSE_DIST2: 8,
    OP_WEIGHT_SELECT: 9,
    // Return codes
    ERR_OK: 0,
    ERR_INVALID_ARGS: -1,
    ERR_STAGING_OVERFLOW: -2,
    ERR_TIMEOUT: -3,
    ERR_UNAVAILABLE: -4,
    ERR_GPU_EXEC: -5,
  });

  scope.BauxWebGpu = {
    SENTINEL_IDX,
    DEFAULT_EPSILON,
    SAB_RPC,
    formatAdapterLabel,
    BauxWebGpuBridge,
    createCModuleWrapper,
    init,
  };
})(typeof self !== "undefined" ? self : globalThis);
