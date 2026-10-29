//! Compile the proposal kernels for a validated logical pool.
//!
//! Configuration changes kernel specialization, not the incumbent parameters.
//! Seed layout is owned by `procedural_pool`; proposal formulas are implemented
//! by `resident_search.metal`. The historical default source omits pool macros,
//! whereas pair rotation always requires explicit program-aware specialization.

use super::*;
use crate::procedural_pool::{ProceduralPool, ProceduralPoolConfig};

#[cfg(test)]
#[path = "procedural_tests.rs"]
mod tests;

impl SearchState {
    /// Replace the pool's proposal, replay, selection, and materialization
    /// kernels. Requires an idle state before any published ask or vector
    /// selector. Only one or four total proposals are supported. A matching
    /// ordinary layout is a no-op; pair rotation recompiles even for the
    /// historical layout. Compilation may fail; callers must not treat
    /// configuration as an ask or an observation, and this method does not
    /// evaluate or accept a candidate.
    pub fn configure_pool(&mut self, pool: ProceduralPool) -> Result<(), String> {
        self.check_idle()?;
        ProceduralPoolConfig {
            arms: pool.arms(),
            candidates_per_arm: pool.slots(),
        }
        .resolve_resident()?;
        if self.started || self.objective_selector.is_some() {
            return Err(
                "Configure procedural layout before asking or configuring vector acquisition"
                    .into(),
            );
        }
        if pool == self.pool_layout
            && self.proposal_method != crate::procedural_pool::ProposalMethod::PairMix
            && self.proposal_method != crate::procedural_pool::ProposalMethod::QuadraticFlow
        {
            return Ok(());
        }
        let source = self.procedural_source(pool);
        let propose = self
            .runtime
            .pipeline(&source, "Procedural search", "bf16_propose")?;
        let initial =
            self.runtime
                .pipeline(&source, "Procedural search", "bf16_propose_initial")?;
        let pool_pipeline =
            self.runtime
                .pipeline(&source, "Procedural search", "bf16_propose_pool")?;
        let short = self
            .runtime
            .precise(&source, "Procedural search", "bf16_replay_short")?;
        let select = self
            .runtime
            .precise(&source, "Procedural search", "bf16_select")?;
        let materialize =
            self.runtime
                .pipeline(&source, "Procedural search", "bf16_materialize")?;
        self.propose_pipeline = propose;
        self.initial_pipeline = initial;
        self.pool_pipeline = pool_pipeline;
        self.replay_short_pipeline = short;
        self.selection_pipeline = select;
        self.materialize_pipeline = materialize;
        self.pool_layout = pool;
        Ok(())
    }

    /// Derive pool metadata's candidate seed. The index must be inside the
    /// configured pool; this internal helper panics if that invariant is
    /// broken. Program-specific kernel remapping and the noise-domain salt
    /// are separate.
    pub(super) fn procedural_seed(&self, root: u64, candidate: usize) -> u64 {
        self.pool_layout
            .seed(root, self.pool_layout.identity(candidate as u32).unwrap())
            .unwrap()
    }

    /// Assemble precision/noise source plus pool specialization macros.
    /// Returns source text without compiling it or allocating parameter rows.
    /// The unspecialized legacy source is retained only outside pair rotation
    /// and Lie flow.
    pub(super) fn procedural_source(&self, pool: ProceduralPool) -> String {
        let source = initialization::search_source(self.independent_fp16, self.perturbation);
        if pool == ProceduralPool::legacy()
            && self.proposal_method != crate::procedural_pool::ProposalMethod::PairMix
            && self.proposal_method != crate::procedural_pool::ProposalMethod::QuadraticFlow
        {
            source
        } else {
            format!(
                "#define PROCEDURAL_SLOTS {}\n#define PROCEDURAL_COUNT {}\n{source}",
                pool.slots(),
                pool.count()
            )
        }
    }
}
