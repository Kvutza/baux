// Resident Metal compute kernel for the 160,531-parameter TCV tokamak policy
// (93 -> 256 -> 256 -> 256 -> 19) and WPO Table 6 (`shape_70166` Precise)
// 5-component reward (`SoftPlus` + `SmoothMax` combiner with alpha = -3.0).
//
// Academic & Open-Source Attribution:
// - David Pfau et al., "Wasserstein Policy Optimization", ICML 2025,
//   arXiv:2505.00663 (Appendix B.3, Table 1, Table 6, Equation 11),
//   https://github.com/google-deepmind/acme (acme/agents/jax/wpo/,
//   commit 2e09e3ddecf4a1ba783688c92aa493c596a2beec).
// - Brendan D. Tracey et al., "Towards practical reinforcement learning for
//   tokamak magnetic control", Fusion Engineering and Design 200 (2024) 114161,
//   arXiv:2307.11546 (Section 3, Equations 1-3).

#include <metal_stdlib>
using namespace metal;

constant uint OBS_DIM = 93;
constant uint ACTION_DIM = 19;
constant uint HIDDEN_DIM = 256;
constant uint TOTAL_PARAMS = 160531;
constant uint STEPS = 100;

constant float ZETA = -2.944438979f; // -ln(19)
constant float LCFS_GOOD = 0.001f;
constant float LCFS_BAD = 0.01f;
constant float IP_GOOD = 100.0f;
constant float IP_BAD = 2000.0f;
constant float PF_GOOD = 100.0f;
constant float PF_BAD = 50.0f;
constant float VOLT_GOOD = 0.0f;
constant float VOLT_BAD = 1.0f;
constant float OH_GOOD = 50.0f;
constant float OH_BAD = 1050.0f;
constant float COMBINER_ALPHA = -3.0f;

inline float softplus_component(float x, float good, float bad) {
    float scale = ZETA * (x - good) / (bad - good);
    float sigma = 1.0f / (1.0f + exp(-scale));
    return clamp(2.0f * sigma, 0.0f, 1.0f);
}

inline float smooth_max5(float c0, float c1, float c2, float c3, float c4, float alpha) {
    float w0 = exp(alpha * c0);
    float w1 = exp(alpha * c1);
    float w2 = exp(alpha * c2);
    float w3 = exp(alpha * c3);
    float w4 = exp(alpha * c4);
    float num = c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4;
    float den = w0 + w1 + w2 + w3 + w4;
    return num / den;
}

// Evaluates a batch of 160,531-parameter TCV tokamak policies (93 -> 256 -> 256 -> 256 -> 19)
// using the WPO Table 6 (`shape_70166` Precise) 5-component reward and SmoothMax combiner.
kernel void tokamak_eval_policy(
    device const float *weights [[buffer(0)]],
    device float *rewards [[buffer(1)]],
    constant uint &batch_size [[buffer(2)]],
    uint thread_idx [[thread_position_in_grid]]
) {
    if (thread_idx >= batch_size) {
        return;
    }

    device const float *policy_w = weights + (thread_idx * TOTAL_PARAMS);

    uint offset = 0;
    device const float *w1 = policy_w + offset;
    offset += OBS_DIM * HIDDEN_DIM;
    device const float *b1 = policy_w + offset;
    offset += HIDDEN_DIM;

    device const float *w2 = policy_w + offset;
    offset += HIDDEN_DIM * HIDDEN_DIM;
    device const float *b2 = policy_w + offset;
    offset += HIDDEN_DIM;

    device const float *w3 = policy_w + offset;
    offset += HIDDEN_DIM * HIDDEN_DIM;
    device const float *b3 = policy_w + offset;
    offset += HIDDEN_DIM;

    device const float *w4 = policy_w + offset;
    offset += HIDDEN_DIM * ACTION_DIM;
    device const float *b4 = policy_w + offset;

    float state[OBS_DIM];
    for (uint i = 0; i < 72; ++i) {
        state[i] = sin(float(i + 1) * 0.17f) * 0.06f;
    }
    for (uint k = 0; k < 16; ++k) {
        state[72 + k] = 0.12f + 0.02f * cos(float(k) * 0.5f);
    }
    state[88] = 0.25f;
    state[89] = 0.10f;
    state[90] = 0.0f;
    state[91] = 0.35f;
    state[92] = 0.35f;

    float h1[HIDDEN_DIM];
    float h2[HIDDEN_DIM];
    float h3[HIDDEN_DIM];
    float action[ACTION_DIM];

    float total_return = 0.0f;

    for (uint step = 0; step < STEPS; ++step) {
        // Layer 1: 93 -> 256 (ELU)
        for (uint j = 0; j < HIDDEN_DIM; ++j) {
            float sum = b1[j];
            for (uint i = 0; i < OBS_DIM; ++i) {
                sum += state[i] * w1[i * HIDDEN_DIM + j];
            }
            h1[j] = (sum >= 0.0f) ? sum : (exp(sum) - 1.0f);
        }

        // Layer 2: 256 -> 256 (ELU)
        for (uint j = 0; j < HIDDEN_DIM; ++j) {
            float sum = b2[j];
            for (uint i = 0; i < HIDDEN_DIM; ++i) {
                sum += h1[i] * w2[i * HIDDEN_DIM + j];
            }
            h2[j] = (sum >= 0.0f) ? sum : (exp(sum) - 1.0f);
        }

        // Layer 3: 256 -> 256 (ELU)
        for (uint j = 0; j < HIDDEN_DIM; ++j) {
            float sum = b3[j];
            for (uint i = 0; i < HIDDEN_DIM; ++i) {
                sum += h2[i] * w3[i * HIDDEN_DIM + j];
            }
            h3[j] = (sum >= 0.0f) ? sum : (exp(sum) - 1.0f);
        }

        // Output: 256 -> 19 (Tanh bounded continuous control squash)
        for (uint j = 0; j < ACTION_DIM; ++j) {
            float sum = b4[j];
            for (uint i = 0; i < HIDDEN_DIM; ++i) {
                sum += h3[i] * w4[i * ACTION_DIM + j];
            }
            action[j] = tanh(sum);
        }

        // Table 6 Component 1: LCFS Distance (m)
        float shape_sq = 0.0f;
        for (uint i = 0; i < 72; ++i) {
            shape_sq += state[i] * state[i];
        }
        float lcfs_dist = 0.05f * sqrt(shape_sq / 72.0f);
        float c_lcfs = softplus_component(lcfs_dist, LCFS_GOOD, LCFS_BAD);

        // Table 6 Component 2: Plasma Current (A)
        float ip_err = 1000.0f * abs(state[91]);
        float c_ip = softplus_component(ip_err, IP_GOOD, IP_BAD);

        // Table 6 Component 3: Poloidal away from 0 (A)
        float pf_sum = 0.0f;
        for (uint k = 0; k < 16; ++k) {
            pf_sum += 1000.0f * abs(state[72 + k]);
        }
        float c_pf = softplus_component(pf_sum / 16.0f, PF_GOOD, PF_BAD);

        // Table 6 Component 4: Normalized Voltages in Bounds
        float volt_level = 0.0f;
        for (uint j = 0; j < ACTION_DIM; ++j) {
            float u = abs(action[j]);
            volt_level += max(0.0f, u - 0.15f) + 0.5f * u * u;
        }
        float c_volt = softplus_component(volt_level / float(ACTION_DIM), VOLT_GOOD, VOLT_BAD);

        // Table 6 Component 5: Ohmic Coils Close (A)
        float oh_diff = 1000.0f * abs(state[88] - state[89]);
        float c_oh = softplus_component(oh_diff, OH_GOOD, OH_BAD);

        total_return += smooth_max5(c_lcfs, c_ip, c_pf, c_volt, c_oh, COMBINER_ALPHA);

        // Coupled open-loop unstable magnetic response transition
        float pf_mean = 0.0f;
        for (uint k = 0; k < 16; ++k) {
            pf_mean += action[k];
        }
        pf_mean /= 16.0f;

        for (uint i = 0; i < 72; ++i) {
            float mode = sin(float(i + 1) * 0.17f);
            float harmonic = 0.0f;
            for (uint j = 0; j < ACTION_DIM; ++j) {
                float phase = float(i * 7 + j * 13) * 0.11f;
                harmonic += cos(phase) * action[j];
            }
            harmonic /= float(ACTION_DIM);
            state[i] = 1.018f * state[i] + mode * (0.0045f - 0.045f * pf_mean) - 0.04f * harmonic;
        }
        for (uint k = 0; k < 16; ++k) {
            state[72 + k] = 0.965f * state[72 + k] + 0.001f + 0.045f * action[k];
        }
        state[88] = 1.008f * state[88] + 0.006f - 0.06f * action[16];
        state[89] = 0.985f * state[89] - 0.003f + 0.06f * action[17];
        state[90] = 0.92f * state[90] + 0.10f * action[18];

        float oh_drive = 0.5f * (action[16] + action[17]) + 0.4f * action[18];
        state[91] = 1.015f * state[91] + 0.015f - 0.16f * oh_drive;
        float t_next = float(step + 2);
        state[92] = (float(step + 1) * state[92] + state[91]) / t_next;
    }

    rewards[thread_idx] = total_return;
}
