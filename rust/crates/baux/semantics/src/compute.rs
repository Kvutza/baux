//! Typed signatures for portable numerical programs.
//!
//! StableHLO owns the instruction set and its verifier:
//! <https://openxla.org/stablehlo/spec>. This module owns the call contract;
//! it does not infer a signature by parsing arbitrary MLIR text.

/// A tensor argument or result. Shape and dtype are part of specialization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: String,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, dtype: &str) -> Result<Self, String> {
        if !matches!(
            dtype,
            "bool"
                | "int32"
                | "int64"
                | "uint32"
                | "uint64"
                | "float16"
                | "bfloat16"
                | "float32"
                | "float64"
        ) {
            return Err(format!("unsupported tensor dtype {dtype:?}"));
        }
        shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or("tensor element count overflows usize")?;
        Ok(Self {
            shape,
            dtype: dtype.into(),
        })
    }
}

/// Owned StableHLO source and its flattened, ordered call signature.
///
/// Trees stay in the language adapter. A compiler must verify that the entry
/// function agrees with this signature before an executable can be called.
/// State and RNG tensors are ordinary arguments/results; no hidden mutation
/// or host-array conversion is implied by this contract.
#[derive(Clone, Debug)]
pub struct Program {
    source: String,
    entry: String,
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
}

impl Program {
    pub fn new(
        source: String,
        entry: String,
        inputs: Vec<Tensor>,
        outputs: Vec<Tensor>,
    ) -> Result<Self, String> {
        if source.trim().is_empty() {
            return Err("StableHLO source is empty".into());
        }
        if entry.is_empty()
            || !entry
                .bytes()
                .enumerate()
                .all(|(i, b)| b.is_ascii_alphabetic() || b == b'_' || (i > 0 && b.is_ascii_digit()))
        {
            return Err("entry must be an unquoted MLIR identifier".into());
        }
        // Revalidate public Tensor fields, including values constructed by Rust
        // callers.
        for tensor in inputs.iter().chain(&outputs) {
            Tensor::new(tensor.shape.clone(), &tensor.dtype)?;
        }
        Ok(Self {
            source,
            entry,
            inputs,
            outputs,
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn entry(&self) -> &str {
        &self.entry
    }
    pub fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[Tensor] {
        &self.outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature() {
        let scalar = Tensor::new(vec![], "float32").unwrap();
        assert!(
            Program::new(
                "module {}".into(),
                "invalid entry".into(),
                vec![scalar.clone()],
                vec![scalar],
            )
            .is_err()
        );
        assert!(Tensor::new(vec![usize::MAX, 2], "float32").is_err());
        assert!(Tensor::new(vec![0, 2], "float32").is_ok());
        assert!(Tensor::new(vec![2], "object").is_err());
    }
}
