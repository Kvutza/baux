#include <metal_stdlib>
using namespace metal;

struct Shape { uint rows, cols, batch; };
struct Step { float rate, momentum, decay, epsilon, beta2, bias1, bias2, scale; };
struct Polynomial { float a, b, c; };

kernel void muon_init(device const half* weights [[buffer(0)]],
    device float* master [[buffer(1)]], device float* momentum [[buffer(2)]],
    constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    master[i] = float(weights[i]);
    momentum[i] = 0.0f;
}

// Consume the gradient, replacing it with the Nesterov lookahead. One partial
// norm per 8192 elements; normalization never reads a scalar back to the host.
kernel void muon_momentum(device float* gradient [[buffer(0)]],
    device float* momentum [[buffer(1)]], device float* partials [[buffer(2)]],
    constant Shape& p [[buffer(3)]], constant Step& s [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint size = p.rows * p.cols;
    const uint chunks = (size + 8191) / 8192;
    const uint matrix = group / chunks;
    const uint first = (group % chunks) * 8192;
    threadgroup float sums[8];
    float sum = 0.0f;
    for (uint j = first + tid; j < min(first + 8192, size); j += 256) {
        const uint i = matrix * size + j;
        const float g = gradient[i];
        const float m = s.momentum * momentum[i] + (1.0f - s.momentum) * g;
        momentum[i] = m;
        const float v = (1.0f - s.momentum) * g + s.momentum * m;
        gradient[i] = v;
        sum = fma(v, v, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) sums[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        sum = simd_sum(lane < 8 ? sums[lane] : 0.0f);
        if (lane == 0) partials[group] = sum;
    }
}

kernel void muon_norm(device const float* partials [[buffer(0)]],
    device float* norm [[buffer(1)]], constant Shape& p [[buffer(2)]],
    constant Step& s [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint chunks = (p.rows * p.cols + 8191) / 8192;
    threadgroup float sums[8];
    float sum = 0.0f;
    for (uint i = tid; i < chunks; i += 256) sum += partials[group * chunks + i];
    sum = simd_sum(sum);
    if (lane == 0) sums[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        sum = simd_sum(lane < 8 ? sums[lane] : 0.0f);
        if (lane == 0) norm[group] = 1.0f / max(1.01f * sqrt(sum), s.epsilon);
    }
}

kernel void muon_normalize(device const float* gradient [[buffer(0)]],
    device const float* norm [[buffer(1)]], device half* x [[buffer(2)]],
    constant Shape& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    const uint size = p.rows * p.cols;
    if (i >= size * p.batch) return;
    const uint j = i % size;
    const uint source = p.rows > p.cols ? (j % p.rows) * p.cols + j / p.rows : j;
    x[i] = half(gradient[(i / size) * size + source] * norm[i / size]);
}

kernel void muon_gram(device const float* product [[buffer(0)]],
    device half* gram [[buffer(1)]], constant uint& count [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) gram[i] = half(product[i]);
}

kernel void muon_polynomial(device const half* gram [[buffer(0)]],
    device const float* square [[buffer(1)]], device half* polynomial [[buffer(2)]],
    constant uint& count [[buffer(3)]], constant Polynomial& c [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) polynomial[i] = half(c.b * float(gram[i]) + c.c * square[i]);
}

kernel void muon_axpy(device const half* x [[buffer(0)]],
    device const float* product [[buffer(1)]], device half* next [[buffer(2)]],
    constant uint& count [[buffer(3)]], constant Polynomial& c [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) next[i] = half(c.a * float(x[i]) + product[i]);
}

kernel void muon_update(device const half* direction [[buffer(0)]],
    device float* master [[buffer(1)]], device half* weights [[buffer(2)]],
    constant Shape& p [[buffer(3)]], constant Step& s [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    const uint size = p.rows * p.cols;
    if (i >= size * p.batch) return;
    const uint j = i % size;
    const uint source = p.rows > p.cols ? (j % p.cols) * p.rows + j / p.cols : j;
    const float value = (1.0f - s.rate * s.decay) * master[i]
        - s.rate * s.scale * float(direction[(i / size) * size + source]);
    master[i] = value;
    weights[i] = half(value);
}

kernel void adam_update(device const float* gradient [[buffer(0)]],
    device float* first [[buffer(1)]], device float* second [[buffer(2)]],
    device float* master [[buffer(3)]], device half* weights [[buffer(4)]],
    constant uint& count [[buffer(5)]], constant Step& s [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float g = gradient[i];
    const float m = s.momentum * first[i] + (1.0f - s.momentum) * g;
    const float v = s.beta2 * second[i] + (1.0f - s.beta2) * g * g;
    first[i] = m;
    second[i] = v;
    const float update = (m / s.bias1) / (sqrt(v / s.bias2) + s.epsilon);
    const float value = (1.0f - s.rate * s.decay) * master[i] - s.rate * update;
    master[i] = value;
    weights[i] = half(value);
}
