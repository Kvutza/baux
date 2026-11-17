// Parallel arbitrary-pool-size multi-objective selector (Pareto NDS + MORBO
// augmented Chebyshev + Stiefel James-Stein contraction + Affine + y_bounds).
struct ObjectiveParams {
    float outcomes[8][128], variances[8][128], draws[8][128];
    float scales[8], weights[8], minima[8], maxima[8];
    uint y_bounds_mode[8];
    float y_lower[8], y_upper[8];
    float affine_a[8], affine_b[8], affine_c[8];
    float stiefel_proj[8][8];
    uint stiefel_enabled, width, mode;
    float alpha;
    uint clip, pad;
};
struct ObjectiveSelectionReport {
    float values[4][8], weights[8];
    uint width, nondominated_mask, selected, pad;
};

inline float objective_chebyshev(thread const float *values, constant ObjectiveParams &o) {
    float minimum = INFINITY, sum = 0.0f;
    for (uint metric = 0; metric < o.width; metric++) {
        float range = o.maxima[metric] - o.minima[metric];
        float normalized = range <= 0.0f ? 0.5f : (values[metric] - o.minima[metric]) / range;
        if (o.clip != 0) normalized = clamp(normalized, 0.0f, 1.0f);
        float weighted = normalized * o.weights[metric];
        minimum = min(minimum, weighted);
        sum += weighted;
    }
    return minimum + o.alpha * sum;
}

inline uint objective_geometry(
    thread const float *distances,
    constant SelectionParams &q,
    uint excluded_index,
    thread float *scaled_distances,
    thread uint *nearest) {
    if (q.distance_metric != 0) {
        float query_neighbors[128];
        uint query_count = 0;
        for (uint i = 0; i < q.history; i++) {
            if (i == excluded_index || distances[i] <= 0.0f) continue;
            uint position = query_count;
            while (position > 0 && distances[i] < query_neighbors[position - 1]) {
                query_neighbors[position] = query_neighbors[position - 1];
                position--;
            }
            query_neighbors[position] = distances[i];
            query_count++;
        }
        uint local_rank = min(q.local_scale_neighbors, query_count);
        float query_scale = local_rank == 0
            ? 1.0f
            : max(query_neighbors[local_rank - 1], 1.0e-12f);
        for (uint i = 0; i < q.history; i++) {
            float s_q = sqrt(query_scale), s_h = sqrt(max(q.local_scales[i], 1.0e-12f));
            if (q.distance_metric == 3) {
                float diff = abs(s_q - s_h);
                scaled_distances[i] = distances[i] / sqrt(max(s_q * s_h, 1.0e-12f)) + diff * diff;
            } else {
                float hyp = float(q.distance_metric >> 1);
                scaled_distances[i] = (distances[i] + hyp * (s_q - s_h) * (s_q - s_h)) / ((1.0f + hyp) * s_q * s_h);
            }
        }
    } else {
        for (uint i = 0; i < q.history; i++) scaled_distances[i] = distances[i];
    }
    uint available = 0;
    for (uint i = 0; i < q.history; i++) {
        if (i == excluded_index) continue;
        uint position = available++;
        while (position > 0
            && scaled_distances[i] < scaled_distances[nearest[position - 1]]) {
            nearest[position] = nearest[position - 1];
            position--;
        }
        nearest[position] = i;
    }
    return available;
}

inline void metric_posterior(
    thread const float *scaled_distances,
    thread const uint *nearest,
    uint available,
    constant SelectionParams &q,
    constant float *outcomes,
    constant float *variances,
    constant float *draws,
    float scale,
    uint yb_mode,
    float yb_lo,
    float yb_hi,
    float aff_a,
    float aff_b,
    float aff_c,
    thread float &mean_out,
    thread float &standard_error_out,
    thread float &epistemic_se_out,
    thread float &thompson_mult_out,
    thread uint &neighbors_out) {
    uint count = min(q.neighbors, available);
    thompson_mult_out = 0.0f;
    if (count == 0) {
        float mean = outcomes[q.incumbent_index];
        float standard_error = sqrt(max(q.aleatoric_scale, 1.0e-12f)) * scale;
        float epistemic_se = standard_error;
        warp_output(yb_mode, yb_lo, yb_hi, aff_a, aff_b, aff_c, mean, standard_error, epistemic_se);
        mean_out = mean;
        standard_error_out = standard_error;
        epistemic_se_out = epistemic_se;
        neighbors_out = 0;
        return;
    }
    float y_scale_sq = max(scale * scale, 1.0e-12f);
    float var_gate = q.noise_aware != 0u ? 1.0f : 0.0f;
    if (q.adaptive_neighbors != 0) {
        float precision_sum = 0.0f, noise_sum = 0.0f, best_risk = INFINITY;
        uint chosen = 1;
        for (uint position = 0; position < count; position++) {
            uint index = nearest[position];
            float yvar = var_gate * variances[index] / y_scale_sq;
            float noise = q.aleatoric_scale + yvar;
            float precision = 1.0f / max(1.0e-9f + q.epistemic_scale * scaled_distances[index] + noise, 1.0e-12f);
            precision_sum += precision;
            noise_sum += precision * precision * noise;
            float variance_error = noise_sum / max(precision_sum * precision_sum, 1.0e-24f);
            float risk = q.epistemic_scale * scaled_distances[index] + variance_error;
            if (risk < best_risk) { best_risk = risk; chosen = position + 1; }
        }
        count = chosen;
    }
    neighbors_out = count;
    float sum = 0.0f;
    float value = 0.0f;
    float reference = 1.17549435e-38f;
    for (uint position = 0; position < count; position++) {
        uint index = nearest[position];
        float variance = 1.0e-9f
            + q.epistemic_scale * scaled_distances[index]
            + q.aleatoric_scale
            + var_gate * variances[index] / y_scale_sq;
        float weight = 1.0f / max(variance, 1.0e-12f);
        sum += weight;
        value += weight * outcomes[index];
        reference = max(reference, weight);
    }
    float mean = value / max(sum, 1.0e-12f);
    float aleatoric = 0.0f;
    for (uint position = 0; position < count; position++) {
        uint index = nearest[position];
        float yvar = var_gate * variances[index] / y_scale_sq;
        float variance = 1.0e-9f
            + q.epistemic_scale * scaled_distances[index]
            + q.aleatoric_scale
            + yvar;
        float weight = (1.0f / max(variance, 1.0e-12f)) / max(sum, 1.0e-12f);
        aleatoric += weight * (q.aleatoric_scale + yvar);
    }
    float epistemic_var = 1.0f / max(sum, 1.0e-12f);
    float standard_error = sqrt(epistemic_var + aleatoric) * scale;
    float epistemic_se = sqrt(epistemic_var) * scale;
    warp_output(yb_mode, yb_lo, yb_hi, aff_a, aff_b, aff_c, mean, standard_error, epistemic_se);
    mean_out = mean;
    standard_error_out = standard_error;
    epistemic_se_out = epistemic_se;
    if (q.acquisition == 1) {
        float noise = 0.0f;
        float squared = 0.0f;
        for (uint position = 0; position < count; position++) {
            uint index = nearest[position];
            float variance = 1.0e-9f
                + q.epistemic_scale * scaled_distances[index]
                + q.aleatoric_scale
                + var_gate * variances[index] / y_scale_sq;
            float weight = (1.0f / max(variance, 1.0e-12f)) / reference;
            noise += weight * draws[index];
            squared += weight * weight;
        }
        thompson_mult_out = noise / max(sqrt(squared), 1.0e-12f);
    }
}

inline float metric_acq(uint acq_kind, float beta, float mean, float epistemic_se, float thompson_mult) {
    if (acq_kind == 1u) return mean + epistemic_se * thompson_mult;
    if (acq_kind == 2u) return mean + epistemic_se;
    return mean + beta * epistemic_se;
}

// Stage 1 parallel multi-objective candidate evaluation: 1 GPU thread per candidate across 0..q.pool_count!
kernel void dense_score_objectives(
    device const Partial *aggregates [[buffer(0)]],
    device float *pool_distances [[buffer(1)]],
    device CandidateScore *candidate_scores [[buffer(2)]],
    constant SelectionParams &q [[buffer(3)]],
    constant ObjectiveParams &o [[buffer(4)]],
    device float *candidate_vectors [[buffer(5)]],
    device const float *axis_candidates [[buffer(6)]],
    uint candidate [[thread_position_in_grid]]) {
    if (candidate >= q.pool_count) return;
    float distances[128];
    float proposal_norm = aggregates[candidate].squared;
    bool changed = false;
    bool invalid = false;
    for (uint row = 0; row < q.history; row++) {
        distances[row] = q.implicit_history != 0
            ? q.base_distances[row] + proposal_norm
            : 0.0f;
    }
    for (uint row = 0; row < q.resident_history; row += 2) {
        Partial aggregate = aggregates[(row / 2) * q.pool_count + candidate];
        float first = aggregate.anchor;
        float second = aggregate.rejected;
        changed = changed || aggregate.changed != 0;
        invalid = invalid || aggregate.invalid != 0;
        uint first_index = q.implicit_history != 0 ? q.resident_indices[row] : row;
        if (first_index < q.history) {
            distances[first_index] = first;
        }
        if (row + 1 < q.resident_history) {
            uint second_index = q.implicit_history != 0 ? q.resident_indices[row + 1] : row + 1;
            if (second_index < q.history) {
                distances[second_index] = second;
            }
        }
        invalid = invalid || !isfinite(first)
            || (row + 1 < q.resident_history && !isfinite(second));
    }
    if (q.axis_enabled != 0) {
        float axis_cand = axis_candidates[candidate];
        changed = changed || (axis_cand != q.axis_base);
        for (uint row = 0; row < q.history; row++) {
            float delta = axis_cand - q.axis_history[row];
            distances[row] += q.axis_weight * delta * delta;
        }
    }
    for (uint row = 0; row < q.history; row++) {
        pool_distances[candidate * q.history + row] = distances[row];
    }
    for (uint m = 0; m < 8; m++) {
        candidate_vectors[candidate * 8 + m] = NAN;
    }
    if (candidate < q.candidate_floor || invalid) {
        candidate_scores[candidate] = {NAN, NAN, NAN, 0u, uint(changed), 1u};
        return;
    }

    float scaled_distances[128];
    uint nearest[128];
    uint available = objective_geometry(distances, q, 128u, scaled_distances, nearest);

    float ctrl_mean = NAN, ctrl_se = NAN, ctrl_ep = NAN, ctrl_tm = 0.0f;
    uint ctrl_nb = 0u;
    metric_posterior(
        scaled_distances, nearest, available, q,
        q.outcomes, q.variances, q.draws, q.y_scale,
        q.y_bounds_mode, q.y_lower, q.y_upper,
        q.affine_a, q.affine_b, q.affine_c,
        ctrl_mean, ctrl_se, ctrl_ep, ctrl_tm, ctrl_nb);
    float ctrl_score = metric_acq(q.acquisition, q.beta, ctrl_mean, ctrl_ep, ctrl_tm);
    bool eligible = isfinite(ctrl_score) && isfinite(ctrl_mean) && isfinite(ctrl_se);

    float means[8], ep_ses[8], t_mults[8];
    for (uint metric = 0; metric < o.width; metric++) {
        float m_mean = NAN, m_se = NAN, m_ep = NAN, m_tm = 0.0f;
        uint m_nb = 0u;
        metric_posterior(
            scaled_distances, nearest, available, q,
            o.outcomes[metric], o.variances[metric], o.draws[metric], o.scales[metric],
            o.y_bounds_mode[metric], o.y_lower[metric], o.y_upper[metric],
            o.affine_a[metric], o.affine_b[metric], o.affine_c[metric],
            m_mean, m_se, m_ep, m_tm, m_nb);
        means[metric] = m_mean;
        ep_ses[metric] = m_ep;
        t_mults[metric] = m_tm;
    }

    // Optional Stiefel Empirical-Bayes James-Stein subspace contraction
    if (o.stiefel_enabled != 0u && o.width >= 2u) {
        float p_r[8];
        float sig_sq = 0.0f, noise_var = 0.0f;
        for (uint i = 0; i < o.width; i++) {
            float pi = 0.0f;
            for (uint j = 0; j < o.width; j++) {
                pi += o.stiefel_proj[i][j] * means[j];
            }
            p_r[i] = pi;
            float diff = means[i] - pi;
            sig_sq += diff * diff;
            noise_var += max(1.0f - o.stiefel_proj[i][i], 0.0f) * ep_ses[i] * ep_ses[i];
        }
        float denom = max(sig_sq, 0.0f) + noise_var;
        float stein_a = (noise_var > 0.0f && denom > 0.0f) ? clamp(max(sig_sq, 0.0f) / denom, 0.0f, 1.0f) : 1.0f;
        float new_ep[8];
        for (uint i = 0; i < o.width; i++) {
            means[i] = p_r[i] + stein_a * (means[i] - p_r[i]);
            float acc = 0.0f;
            for (uint j = 0; j < o.width; j++) {
                float diag = (i == j) ? stein_a : 0.0f;
                float tij = (1.0f - stein_a) * o.stiefel_proj[i][j] + diag;
                float term = tij * ep_ses[j];
                acc += term * term;
            }
            new_ep[i] = sqrt(acc);
        }
        for (uint i = 0; i < o.width; i++) ep_ses[i] = new_ep[i];
    }

    float vec[8];
    for (uint metric = 0; metric < o.width; metric++) {
        vec[metric] = metric_acq(q.acquisition, q.beta, means[metric], ep_ses[metric], t_mults[metric]);
        candidate_vectors[candidate * 8 + metric] = vec[metric];
        eligible = eligible && isfinite(vec[metric]);
    }
    float final_score = ctrl_score;
    if (o.mode == 1u) {
        final_score = objective_chebyshev(vec, o);
        eligible = eligible && isfinite(final_score);
    }
    candidate_scores[candidate] = {
        ctrl_mean,
        ctrl_se,
        final_score,
        ctrl_nb,
        uint(changed),
        uint(!eligible)
    };
}

// Stage 2 parallel 256-thread Pareto NDS / MORBO Chebyshev reduction!
kernel void dense_select_objectives(
    device const CandidateScore *scores [[buffer(0)]],
    device Decision *decision [[buffer(1)]],
    device uint *selected_indices [[buffer(2)]],
    constant SelectionParams &q [[buffer(3)]],
    constant ObjectiveParams &o [[buffer(4)]],
    device ObjectiveSelectionReport *report [[buffer(5)]],
    device const float *candidate_vectors [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint local_changed = 0u;
    for (uint c = tid; c < q.pool_count; c += 256u) {
        if (scores[c].invalid == 0u && scores[c].changed != 0u) local_changed = 1u;
    }
    threadgroup uint shared_changed[256];
    shared_changed[tid] = local_changed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) shared_changed[tid] |= shared_changed[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    bool any_changed = shared_changed[0] != 0u;

    threadgroup float inc_mean_sh;
    threadgroup float inc_se_sh;
    threadgroup uint inc_nb_sh;
    if (tid == 0u) {
        float incumbent_distances[128];
        for (uint row = 0; row < q.history; row++) {
            incumbent_distances[row] = q.base_distances[row]
                + (q.axis_enabled != 0
                    ? q.axis_weight * (q.axis_base - q.axis_history[row])
                        * (q.axis_base - q.axis_history[row])
                    : 0.0f);
        }
        float im = 0.0f, ise = 0.0f;
        uint inb = 0u;
        acquisition_score(incumbent_distances, q, 128u, im, ise, inb);
        inc_mean_sh = im;
        inc_se_sh = ise;
        inc_nb_sh = inb;
    }

    // Parallel non-dominated frontier bitmask (low 32 candidates) + Pareto/MORBO reduction
    uint local_nd_mask = 0u;
    for (uint c = tid; c < q.pool_count; c += 256u) {
        CandidateScore cs = scores[c];
        if (c < q.candidate_floor || cs.invalid != 0u || (any_changed && cs.changed == 0u)) continue;
        bool dominated = false;
        for (uint other = 0u; other < q.pool_count; other++) {
            if (other == c) continue;
            CandidateScore os = scores[other];
            if (other < q.candidate_floor || os.invalid != 0u || (any_changed && os.changed == 0u)) continue;
            bool no_worse = true, strictly_better = false;
            for (uint m = 0u; m < o.width; m++) {
                float ov = candidate_vectors[other * 8u + m];
                float cv = candidate_vectors[c * 8u + m];
                no_worse = no_worse && (ov >= cv);
                strictly_better = strictly_better || (ov > cv);
            }
            if (no_worse && strictly_better) { dominated = true; break; }
        }
        if (!dominated && c < 32u) {
            local_nd_mask |= (1u << c);
        }
    }
    shared_changed[tid] = local_nd_mask;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) shared_changed[tid] |= shared_changed[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint nondominated_mask = shared_changed[0];

    uint arms_to_pick = max(min(q.num_arms, q.pool_count), 1u);
    threadgroup float red_score[256];
    threadgroup ulong red_prio[256];
    threadgroup uint red_idx[256];
    threadgroup uint picked_count;
    if (tid == 0u) picked_count = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (q.forced_candidate != 0xffffffffu) {
        if (tid == 0u) {
            uint fc = q.forced_candidate;
            bool ok = fc < q.pool_count
                && fc >= q.candidate_floor
                && scores[fc].invalid == 0u
                && (!any_changed || scores[fc].changed != 0u);
            if (ok) {
                selected_indices[0] = fc;
                picked_count = 1u;
            } else {
                selected_indices[0] = 0xffffffffu;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else if (o.mode == 0u) {
        // Multi-objective Pareto NDS + priority tie-break across 0..q.pool_count
        for (uint arm = 0u; arm < arms_to_pick; arm++) {
            ulong best_prio = 0ul;
            uint best_c = 0xffffffffu;
            for (uint c = tid; c < q.pool_count; c += 256u) {
                CandidateScore cs = scores[c];
                if (c < q.candidate_floor || cs.invalid != 0u || (any_changed && cs.changed == 0u)) continue;
                bool already = false;
                for (uint prev = 0u; prev < arm; prev++) {
                    if (selected_indices[prev] == c) { already = true; break; }
                }
                if (already) continue;
                bool dominated = false;
                for (uint other = 0u; other < q.pool_count; other++) {
                    if (other == c) continue;
                    CandidateScore os = scores[other];
                    if (other < q.candidate_floor || os.invalid != 0u || (any_changed && os.changed == 0u)) continue;
                    bool prev_picked = false;
                    for (uint prev = 0u; prev < arm; prev++) {
                        if (selected_indices[prev] == other) { prev_picked = true; break; }
                    }
                    if (prev_picked) continue;
                    bool no_worse = true, strictly_better = false;
                    for (uint m = 0u; m < o.width; m++) {
                        float ov = candidate_vectors[other * 8u + m];
                        float cv = candidate_vectors[c * 8u + m];
                        no_worse = no_worse && (ov >= cv);
                        strictly_better = strictly_better || (ov > cv);
                    }
                    if (no_worse && strictly_better) { dominated = true; break; }
                }
                if (!dominated) {
                    ulong prio = mix64(q.root_seed ^ (ulong(c) * 0x9e3779b97f4a7c15ul));
                    if (best_c == 0xffffffffu || prio > best_prio || (prio == best_prio && c < best_c)) {
                        best_prio = prio;
                        best_c = c;
                    }
                }
            }
            red_prio[tid] = best_prio;
            red_idx[tid] = best_c;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 128u; stride != 0u; stride >>= 1u) {
                if (tid < stride) {
                    uint rc = red_idx[tid + stride];
                    ulong rp = red_prio[tid + stride];
                    uint lc = red_idx[tid];
                    ulong lp = red_prio[tid];
                    if (rc != 0xffffffffu && (lc == 0xffffffffu || rp > lp || (rp == lp && rc < lc))) {
                        red_idx[tid] = rc;
                        red_prio[tid] = rp;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0u) {
                selected_indices[arm] = red_idx[0];
                if (red_idx[0] != 0xffffffffu) picked_count = arm + 1u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (selected_indices[arm] == 0xffffffffu) break;
        }
    } else {
        // MORBO Chebyshev parallel argmax across 0..q.pool_count
        for (uint arm = 0u; arm < arms_to_pick; arm++) {
            float best_s = -INFINITY;
            uint best_c = 0xffffffffu;
            for (uint c = tid; c < q.pool_count; c += 256u) {
                CandidateScore cs = scores[c];
                if (c < q.candidate_floor || cs.invalid != 0u || (any_changed && cs.changed == 0u)) continue;
                bool already = false;
                for (uint prev = 0u; prev < arm; prev++) {
                    if (selected_indices[prev] == c) { already = true; break; }
                }
                if (already) continue;
                if (best_c == 0xffffffffu || cs.score > best_s || (cs.score == best_s && c < best_c)) {
                    best_s = cs.score;
                    best_c = c;
                }
            }
            red_score[tid] = best_s;
            red_idx[tid] = best_c;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 128u; stride != 0u; stride >>= 1u) {
                if (tid < stride) {
                    uint rc = red_idx[tid + stride];
                    float rs = red_score[tid + stride];
                    uint lc = red_idx[tid];
                    float ls = red_score[tid];
                    if (rc != 0xffffffffu && (lc == 0xffffffffu || rs > ls || (rs == ls && rc < lc))) {
                        red_idx[tid] = rc;
                        red_score[tid] = rs;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0u) {
                selected_indices[arm] = red_idx[0];
                if (red_idx[0] != 0xffffffffu) picked_count = arm + 1u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (selected_indices[arm] == 0xffffffffu) break;
        }
    }

    if (tid == 0u) {
        uint selected = picked_count > 0u ? selected_indices[0] : 0xffffffffu;
        bool valid = selected < q.pool_count;
        CandidateScore win = valid ? scores[selected] : CandidateScore{0.0f, 0.0f, 0.0f, 0u, 0u, 1u};
        float reported_score = valid
            ? (o.mode == 1u ? win.score : 0.0f)
            : 0.0f;
        for (uint c = 0u; c < 4u; c++) {
            bool ok_c = c < q.pool_count && scores[c].invalid == 0u && (!any_changed || scores[c].changed != 0u) && c >= q.candidate_floor;
            for (uint m = 0u; m < 8u; m++) {
                report->values[c][m] = (ok_c && m < o.width) ? candidate_vectors[c * 8u + m] : NAN;
            }
        }
        for (uint m = 0u; m < 8u; m++) report->weights[m] = o.weights[m];
        report->width = o.width;
        report->nondominated_mask = nondominated_mask;
        report->selected = selected;
        report->pad = 0u;

        decision->index = selected;
        decision->valid = valid ? 1u : 0u;
        decision->root_seed = q.root_seed;
        decision->seed = valid ? candidate_seed(q.root_seed, selected, q.pool_slots) : 0ul;
        decision->radius = valid ? q.radius : 0.0f;
        decision->score = reported_score;
        decision->mode = q.mode;
        decision->num_selected = picked_count;
        decision->predicted_mean = valid ? win.mean : 0.0f;
        decision->predicted_standard_error = valid ? win.standard_error : 0.0f;
        decision->incumbent_mean = inc_mean_sh;
        decision->incumbent_standard_error = inc_se_sh;
        decision->neighbors = valid ? win.neighbors : 0u;
        decision->incumbent_neighbors = inc_nb_sh;
        for (uint c = 0u; c < 4u; c++) {
            bool ok_c = c < q.pool_count && scores[c].invalid == 0u && (!any_changed || scores[c].changed != 0u) && c >= q.candidate_floor;
            decision->candidate_means[c] = ok_c ? scores[c].mean : NAN;
            decision->candidate_standard_errors[c] = ok_c ? scores[c].standard_error : NAN;
            decision->candidate_scores[c] = ok_c ? scores[c].score : NAN;
        }
    }
}
