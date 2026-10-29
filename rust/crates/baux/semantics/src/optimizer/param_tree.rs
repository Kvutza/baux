//! 17-family PyTree parameter algebra for the native `fbt-pisa1-mhc4-v1`
//! architecture.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParamFamily {
    TokenEmbed,
    MhcAlpha,
    MhcPre,
    MhcPost,
    MhcMix,
    AttnNorm,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    FfnNorm,
    Router,
    SharedGate,
    SharedUp,
    SharedDown,
    RoutedGateUp,
    RoutedDown,
}

impl ParamFamily {
    pub const ALL: [ParamFamily; 17] = [
        ParamFamily::TokenEmbed,
        ParamFamily::MhcAlpha,
        ParamFamily::MhcPre,
        ParamFamily::MhcPost,
        ParamFamily::MhcMix,
        ParamFamily::AttnNorm,
        ParamFamily::AttnQ,
        ParamFamily::AttnK,
        ParamFamily::AttnV,
        ParamFamily::AttnO,
        ParamFamily::FfnNorm,
        ParamFamily::Router,
        ParamFamily::SharedGate,
        ParamFamily::SharedUp,
        ParamFamily::SharedDown,
        ParamFamily::RoutedGateUp,
        ParamFamily::RoutedDown,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamTree<T> {
    pub values: [T; 17],
}

impl<T> ParamTree<T> {
    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(ParamFamily) -> T,
    {
        Self {
            values: std::array::from_fn(|i| f(ParamFamily::ALL[i])),
        }
    }

    pub fn get(&self, family: ParamFamily) -> &T {
        &self.values[family.index()]
    }

    pub fn zip_with<U, V, F>(&self, other: &ParamTree<U>, mut f: F) -> ParamTree<V>
    where
        F: FnMut(&T, &U) -> V,
    {
        ParamTree::from_fn(|family| f(self.get(family), other.get(family)))
    }

    pub fn fold<Acc, F>(&self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, &T) -> Acc,
    {
        let mut acc = init;
        for val in &self.values {
            acc = f(acc, val);
        }
        acc
    }
}

const MHC4_COUNTS: [usize; 17] = [
    262_144 * 512,
    5 * 2 * 4,
    5 * 2 * (4 * 512 * 4 + 4),
    5 * 2 * (4 * 512 * 4 + 4),
    5 * 2 * (4 * 512 * 16 + 16),
    5 * 512,
    5 * 8 * 512 * 64,
    5 * 512 * 64,
    5 * 512 * 64,
    5 * 8 * 64 * 512,
    5 * 512 + 512,
    5 * 512 * 625,
    5 * 512 * 216,
    5 * 512 * 216,
    5 * 216 * 512,
    5 * 625 * 2 * 512 * 216,
    5 * 625 * 216 * 512,
];

/// Aeneas-extracted 5-layer `fbt-pisa1-mhc4-v1` routed MoE parameter count
/// (`ActiveEnergy.lean` P4).
pub const fn routed_count() -> u64 {
    1_036_800_000
}

/// Aeneas-extracted native `fbt-pisa1-mhc4-v1` total parameter count at
/// `patch_size = 1` (`ActiveEnergy.lean` P5).
pub const fn total_count() -> u64 {
    1_177_723_160
}

impl ParamFamily {
    fn expert_family(family: &str, expert: Option<usize>) -> Option<Self> {
        match (family, expert) {
            ("feedback_state", _) | ("expert_gate_up", Some(0)) => Some(Self::SharedGate),
            ("feedback_gate", _) => Some(Self::SharedUp),
            ("expert_down", Some(0)) => Some(Self::SharedDown),
            ("expert_gate_up", _) => Some(Self::RoutedGateUp),
            ("expert_down", _) => Some(Self::RoutedDown),
            _ => None,
        }
    }

    /// Map runtime Metal/CUDA tensor family names (and optional MoE expert
    /// index, where `expert == Some(0)` is the always-active shared expert)
    /// to the 17-family `ParamTree` algebra.
    pub fn from_expert(family: &str, expert: Option<usize>) -> Self {
        if let Some(mapped) = Self::expert_family(family, expert) {
            return mapped;
        }
        match family {
            "embedding_readout" | "mask_embed" => Self::TokenEmbed,
            "mhc_attention_control" | "mhc_moe_control" => Self::MhcAlpha,
            "mhc_attention_predictor" => Self::MhcPre,
            "mhc_moe_predictor" => Self::MhcPost,
            "mhc_attention_bias" | "mhc_moe_bias" => Self::MhcMix,
            "attention_norm" => Self::AttnNorm,
            "qkv" => Self::AttnQ,
            "attention_output" => Self::AttnO,
            "ffn_norm" | "final_norm" => Self::FfnNorm,
            "router" | "index_query" => Self::Router,
            _ => Self::SharedDown,
        }
    }

    /// Map runtime Metal/CUDA tensor family names to the 17-family `ParamTree`
    /// algebra.
    pub fn from_tensor(family: &str) -> Self {
        Self::from_expert(family, None)
    }
}

impl ParamTree<usize> {
    pub fn mhc4_counts(patch_size: usize) -> Self {
        let extra = if patch_size <= 1 {
            0
        } else {
            (patch_size + 1) * 512
        };
        Self::from_fn(|family| {
            let idx = family.index();
            MHC4_COUNTS[idx] + if idx == 0 { extra } else { 0 }
        })
    }

    pub fn total_parameters(patch_size: usize) -> usize {
        Self::mhc4_counts(patch_size).fold(0, |acc, &x| acc + x)
    }

    pub fn mhc4_shares(patch_size: usize) -> ParamTree<f64> {
        let total = Self::total_parameters(patch_size) as f64;
        let counts = Self::mhc4_counts(patch_size);
        counts.zip_with(&counts, |&c, _| c as f64 / total)
    }

    /// Compute the 17-family active-energy multipliers (`ActiveEnergy.lean`
    /// P4/P5). Balances fan-in-normalized perturbation scales across
    /// always-active dense/shared projections (`1.0`), routed experts
    /// (`1.0`), routers/predictors (`0.5`), and sensitive RMSNorms and mHC
    /// mixing controls (`0.25`).
    pub fn active_multipliers() -> ParamTree<f32> {
        let shares = Self::mhc4_shares(1);
        let routed_share =
            *shares.get(ParamFamily::RoutedGateUp) + *shares.get(ParamFamily::RoutedDown);
        let routed_scale = if routed_share > 0.0 { 1.0_f32 } else { 0.0_f32 };
        ParamTree::from_fn(|family| match family {
            ParamFamily::RoutedGateUp | ParamFamily::RoutedDown => routed_scale,
            ParamFamily::TokenEmbed
            | ParamFamily::AttnQ
            | ParamFamily::AttnK
            | ParamFamily::AttnV
            | ParamFamily::AttnO
            | ParamFamily::SharedGate
            | ParamFamily::SharedUp
            | ParamFamily::SharedDown => 1.0,
            ParamFamily::Router | ParamFamily::MhcPre | ParamFamily::MhcPost => 0.5,
            ParamFamily::MhcAlpha
            | ParamFamily::MhcMix
            | ParamFamily::AttnNorm
            | ParamFamily::FfnNorm => 0.25,
        })
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn prove_counts() {
        assert_eq!(routed_count(), 1_036_800_000);
        assert_eq!(total_count(), 1_177_723_160);
        assert_eq!(
            ParamTree::<usize>::total_parameters(1) as u64,
            total_count()
        );
    }
}
