use deser::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum ModelInitialization {
    /// Historical systems-probe pattern. Retained only as an explicit control.
    #[default]
    Patterned,
    /// GPT-2 Conv1D weights and token embeddings use a zero-mean normal with
    /// standard deviation 0.02; its learned position embeddings use 0.01
    /// (Radford et al., 2019, "Language Models are Unsupervised Multitask
    /// Learners"). BAUX uses RoPE (Su et al., 2021, "RoFormer: Enhanced
    /// Transformer with Rotary Position Embedding", arXiv:2104.09864)
    /// rather than a learned position-embedding matrix.
    /// https://github.com/openai/gpt-2/blob/master/src/model.py
    Gpt2,
    /// Megatron: 0.02 normal weights; attention and MLP output projections use
    /// 0.02 / sqrt(2 * num_layers) (Shoeybi et al., 2019, "Megatron-LM:
    /// Training Multi-Billion Parameter Language Models Using Model
    /// Parallelism", arXiv:1909.08053). BAUX feedback matrices, which have
    /// no Megatron analogue, retain the base 0.02 distribution.
    /// https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py
    Megatron,
    /// Glorot & Bengio (2010) uniform initialization, applied per matrix.
    /// https://proceedings.mlr.press/v9/glorot10a.html
    XavierUniform,
    /// Initialization derived for `pisa-mhc4-v1`.
    ///
    /// Projection columns have fixed second moments, residual writers are
    /// scaled by the squared physical-layer visit multiplicities, router
    /// columns keep the expected maximum initial logit near one, and mHC
    /// begins inside the Birkhoff polytope with every predictor path active
    /// (Su et al., 2021, arXiv:2104.09864; Radford et al., 2019; Shoeybi et
    /// al., 2019, arXiv:1909.08053; Xie et al., 2025, "mHC:
    /// Manifold-Constrained Hyper-Connections", arXiv:2512.24880).
    #[deser(alias = "fbt-pisa1-mhc4")]
    PisaMhc4,
}
