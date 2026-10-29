use deser::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum ModelInitialization {
    /// Historical systems-probe pattern. Retained only as an explicit control.
    #[default]
    Patterned,
    /// GPT-2 Conv1D weights and token embeddings use a zero-mean normal with
    /// standard deviation 0.02; its learned position embeddings use 0.01.
    /// BAUX uses RoPE rather than a learned position-embedding matrix.
    /// https://github.com/openai/gpt-2/blob/master/src/model.py
    Gpt2,
    /// Megatron: 0.02 normal weights; attention and MLP output projections use
    /// 0.02 / sqrt(2 * num_layers). BAUX feedback matrices, which have no
    /// Megatron analogue, retain the base 0.02 distribution.
    /// https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py
    Megatron,
    /// Glorot & Bengio (2010) uniform initialization, applied per matrix.
    /// https://proceedings.mlr.press/v9/glorot10a.html
    XavierUniform,
    /// Initialization derived for `fbt-pisa1-mhc4-v1`.
    ///
    /// Projection columns have fixed second moments, residual writers are
    /// scaled by the squared physical-layer visit multiplicities, router
    /// columns keep the expected maximum initial logit near one, and mHC
    /// begins inside the Birkhoff polytope with every predictor path active.
    FbtPisa1Mhc4,
}
