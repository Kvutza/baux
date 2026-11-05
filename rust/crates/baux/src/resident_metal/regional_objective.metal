#include <metal_stdlib>
using namespace metal;

// Multimodal separable objective over BF16 parameters. Evaluation consumes the
// resident proposal directly; only the scalar reward returns to the host.
kernel void regional_objective(device const ushort *x [[buffer(0)]],
                               device float *reward [[buffer(1)]],
                               constant uint &count [[buffer(2)]],
                               uint lane [[thread_position_in_grid]]) {
    if (lane != 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < count; ++i) {
        const float value = as_type<float>(uint(x[i]) << 16);
        const float residual = value * value - 1.0f;
        sum += residual * residual + 0.05f * value;
    }
    reward[0] = -sum / float(count);
}
