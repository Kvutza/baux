use super::generation::GenerationTask;
use super::*;

pub(super) struct TargetStats {
    sum: f64,
    squared_sum: f64,
    pub(super) count: usize,
    maximum: f32,
    window: Vec<f32>,
    window_sum: f64,
    window_count: usize,
    window_cursor: usize,
    worst_window_sum: f64,
    matches: usize,
    mismatch_sum: f64,
    mismatches: usize,
    first_mismatch: Option<usize>,
}

impl Default for TargetStats {
    fn default() -> Self {
        Self::new(1)
    }
}

impl TargetStats {
    pub(super) fn new(window_tokens: usize) -> Self {
        assert!(window_tokens > 0);
        Self {
            sum: 0.0,
            squared_sum: 0.0,
            count: 0,
            maximum: f32::NEG_INFINITY,
            window: vec![0.0; window_tokens],
            window_sum: 0.0,
            window_count: 0,
            window_cursor: 0,
            worst_window_sum: f64::NEG_INFINITY,
            matches: 0,
            mismatch_sum: 0.0,
            mismatches: 0,
            first_mismatch: None,
        }
    }

    pub(super) fn push(&mut self, position: usize, loss: f32, matched: bool) {
        self.sum += f64::from(loss);
        self.squared_sum += f64::from(loss).powi(2);
        self.count += 1;
        self.maximum = self.maximum.max(loss);
        if self.window_count == self.window.len() {
            self.window_sum -= f64::from(self.window[self.window_cursor]);
        } else {
            self.window_count += 1;
        }
        self.window[self.window_cursor] = loss;
        self.window_cursor = (self.window_cursor + 1) % self.window.len();
        self.window_sum += f64::from(loss);
        if self.window_count == self.window.len() {
            self.worst_window_sum = self.worst_window_sum.max(self.window_sum);
        }
        if matched {
            self.matches += 1;
        } else {
            self.mismatch_sum += f64::from(loss);
            self.mismatches += 1;
            self.first_mismatch.get_or_insert(position);
        }
    }

    pub(super) fn push_span(
        &mut self,
        position: usize,
        loss: f32,
        actual: &[u32],
        expected: &[u32],
    ) {
        debug_assert_eq!(actual.len(), expected.len());
        let count = actual.len();
        self.sum += f64::from(loss) * count as f64;
        self.squared_sum += f64::from(loss).powi(2) * count as f64;
        self.count += count;
        self.maximum = self.maximum.max(loss);
        if self.window_count + count <= self.window.len() {
            let first = count.min(self.window.len() - self.window_cursor);
            self.window[self.window_cursor..self.window_cursor + first].fill(loss);
            self.window[..count - first].fill(loss);
            self.window_cursor = (self.window_cursor + count) % self.window.len();
            self.window_count += count;
            self.window_sum += f64::from(loss) * count as f64;
            if self.window_count == self.window.len() {
                self.worst_window_sum = self.worst_window_sum.max(self.window_sum);
            }
        } else {
            for _ in 0..count {
                if self.window_count == self.window.len() {
                    self.window_sum -= f64::from(self.window[self.window_cursor]);
                } else {
                    self.window_count += 1;
                }
                self.window[self.window_cursor] = loss;
                self.window_cursor = (self.window_cursor + 1) % self.window.len();
                self.window_sum += f64::from(loss);
                if self.window_count == self.window.len() {
                    self.worst_window_sum = self.worst_window_sum.max(self.window_sum);
                }
            }
        }
        let matches = actual
            .iter()
            .zip(expected)
            .filter(|(left, right)| left == right)
            .count();
        let mismatches = count - matches;
        self.matches += matches;
        self.mismatches += mismatches;
        self.mismatch_sum += f64::from(loss) * mismatches as f64;
        if self.first_mismatch.is_none() {
            self.first_mismatch = actual
                .iter()
                .zip(expected)
                .position(|(left, right)| left != right)
                .map(|offset| position + offset);
        }
    }

    pub(super) fn finish(&self) -> Result<decode::TargetQuality, String> {
        if self.count == 0 {
            return Err("target quality requires at least one scored token".into());
        }
        let window_tokens = self.count.min(self.window.len());
        let worst_window_sum = if self.count < self.window.len() {
            self.sum
        } else {
            self.worst_window_sum
        };
        let count = self.count as f64;
        let mean = self.sum / count;
        let mean_variance = if self.count > 1 {
            ((self.squared_sum - self.sum * mean).max(0.0) / (count * (count - 1.0))) as f32
        } else {
            0.0
        };
        let codelength_bits = self.sum / std::f64::consts::LN_2;
        Ok(decode::TargetQuality {
            scored_tokens: self.count,
            codelength_bits,
            bits_per_token: (codelength_bits / count) as f32,
            mean_nll: mean as f32,
            mean_nll_variance: mean_variance,
            maximum_nll: self.maximum,
            worst_window_nll: (worst_window_sum / window_tokens as f64) as f32,
            window_tokens,
            positional_accuracy: self.matches as f32 / self.count as f32,
            positional_mismatches: self.mismatches,
            first_positional_mismatch: self.first_mismatch,
            mismatch_target_nll: (self.mismatches > 0)
                .then_some((self.mismatch_sum / self.mismatches as f64) as f32),
            paired_improvement: None,
            paired_improvement_variance: None,
            patch_nll_octiles: None,
        })
    }
}

pub(super) fn record(
    buffers: &Buffers,
    context: u32,
    task: &GenerationTask,
    tokens: &[u32],
    range: std::ops::Range<usize>,
    patch: usize,
    stats: &mut TargetStats,
) -> Result<(), String> {
    let losses = crate::apple_gpu::buffer_slice::<f32>(&buffers.losses, context as usize)?;
    let prompt_macros = task.prompt.len() / patch;
    if patch > 1 {
        for macro_position in range.start / patch..range.end.div_ceil(patch) {
            let start = range.start.max(macro_position * patch);
            let end = range.end.min((macro_position + 1) * patch);
            let index = prompt_macros.saturating_sub(1) + macro_position;
            let loss = losses[index.min(losses.len() - 1)];
            if !loss.is_finite() {
                return Err(format!(
                    "nonfinite conditional target loss at generated token {start}"
                ));
            }
            stats.push_span(start, loss, &tokens[start..end], &task.expected[start..end]);
        }
        return Ok(());
    }
    for position in range {
        let loss = losses[task.prompt.len() - 1 + position];
        if !loss.is_finite() {
            return Err(format!(
                "nonfinite conditional target loss at generated token {position}"
            ));
        }
        stats.push(position, loss, tokens[position] == task.expected[position]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_parity() {
        let actual = [1, 2, 4, 4, 5, 7];
        let expected = [1, 3, 4, 6, 5, 8];
        let mut scalar = TargetStats::new(4);
        for (position, (&left, &right)) in actual.iter().zip(&expected).enumerate() {
            scalar.push(position, 2.5, left == right);
        }
        let mut span = TargetStats::new(4);
        span.push_span(0, 2.5, &actual, &expected);
        assert_eq!(scalar.finish().unwrap(), span.finish().unwrap());
        let quality = span.finish().unwrap();
        assert_eq!(quality.scored_tokens, actual.len());
        assert!((quality.bits_per_token - 2.5 / std::f32::consts::LN_2).abs() < 1e-6);
        assert!(
            (quality.codelength_bits - 2.5 * actual.len() as f64 / std::f64::consts::LN_2).abs()
                < 1e-9
        );
    }
}
