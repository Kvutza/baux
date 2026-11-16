//! Full-coordinate reference mutation for the temporal mixture proposal.

/// `x' = x + drift * (anchor - x)
///       + diffusion * (|anchor - x| + floor) * Normal(0, 1)` coordinate-wise.
///
/// The drift and distance-dependent noise follow the CBO mechanisms cited in
/// `mixture`. Selecting one shared random anchor instead of using the weighted
/// mean changes the law. Published CBO convergence results do not establish
/// convergence of this temporal mixture or its finite-precision sampler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mutation {
    drift: f32,
    diffusion: f32,
    floor: f32,
}

impl Mutation {
    /// Coefficients are explicit; there are no fixed tensor-family scales or
    /// prescribed parameter count. A zero diffusion permits deterministic
    /// ablations. Positive diffusion and floor supply coordinate-wise noise.
    pub fn new(drift: f32, diffusion: f32, floor: f32) -> Result<Self, String> {
        if !drift.is_finite()
            || !(0.0..=1.0).contains(&drift)
            || !diffusion.is_finite()
            || diffusion < 0.0
            || !floor.is_finite()
            || floor < 0.0
        {
            return Err("invalid temporal mutation coefficients".into());
        }
        Ok(Self {
            drift,
            diffusion,
            floor,
        })
    }

    pub(super) fn apply(
        self,
        current: &[f32],
        anchor: &[f32],
        seed: u64,
    ) -> Result<Vec<f32>, String> {
        if current.is_empty() || current.len() != anchor.len() || current.len() > i64::MAX as usize
        {
            return Err("invalid temporal mutation shape".into());
        }
        let mut candidate = Vec::new();
        candidate
            .try_reserve_exact(current.len())
            .map_err(|error| error.to_string())?;
        for (index, (&state, &target)) in current.iter().zip(anchor).enumerate() {
            if !state.is_finite() || !target.is_finite() {
                return Err("temporal mutation input contains non-finite coordinates".into());
            }
            // Domain separation keeps anchor draws distinct from innovations.
            // These indices address the complete vector. Device shards must
            // preserve global indices, not restart this stream at zero.
            let noise = crate::semantics::random::normal_metric(
                seed ^ 0x84d6_912b_cfe2_03a7,
                index as i64,
                0,
            );
            let displacement = f64::from(target) - f64::from(state);
            let value = (f64::from(state)
                + f64::from(self.drift) * displacement
                + f64::from(self.diffusion) * (displacement.abs() + f64::from(self.floor)) * noise)
                as f32;
            if !value.is_finite() {
                return Err("temporal mutation produced a non-finite candidate".into());
            }
            candidate.push(value);
        }
        Ok(candidate)
    }
}
