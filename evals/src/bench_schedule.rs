//! Rollout scheduling and deterministic PRNG seed derivation for control
//! evaluation.

/// Strategy for active sampling candidate rollouts across rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveSampling {
    /// Reuses the same seeds across rounds (Frozen).
    Frozen { episodes: u32 },
    /// Uses fresh Common Random Numbers per round (Natural).
    Natural { episodes: u32 },
}

impl ActiveSampling {
    /// Returns the number of episodes evaluated per round.
    pub fn episodes(&self) -> u32 {
        match *self {
            Self::Frozen { episodes } | Self::Natural { episodes } => episodes,
        }
    }
}

/// Configuration for evaluation rollout schedules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RolloutSchedule {
    pub rep: u32,
    pub active: ActiveSampling,
    pub heldout_episodes: u32,
}

impl RolloutSchedule {
    /// Creates a new rollout schedule.
    pub fn new(rep: u32, active: ActiveSampling, heldout_episodes: u32) -> Self {
        Self {
            rep,
            active,
            heldout_episodes,
        }
    }

    /// Derives an active rollout seed for the specified round and episode.
    pub fn active_seed(&self, round: u32, episode: u32) -> u64 {
        let index = match self.active {
            ActiveSampling::Frozen { .. } => u64::from(episode),
            ActiveSampling::Natural { episodes } => {
                u64::from(round) * u64::from(episodes) + u64::from(episode)
            }
        };
        control_seed(self.rep, "mujoco-active", index)
    }

    /// Derives a heldout rollout seed for the specified episode.
    pub fn heldout_seed(&self, episode: u32) -> u64 {
        control_seed(self.rep, "mujoco-heldout", u64::from(episode))
    }
}

/// Derives a 64-bit deterministic seed from repetition, domain label, and
/// index.
pub fn control_seed(rep: u32, domain: &str, index: u64) -> u64 {
    let mut domain_hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in domain.as_bytes() {
        domain_hash ^= u64::from(*byte);
        domain_hash = domain_hash.wrapping_mul(0x0100_0000_01b3u64);
    }
    let state = (u64::from(rep) << 32) ^ domain_hash ^ index;
    baux::hash::splitmix64(state)
}
