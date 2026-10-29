//! Device-resident Muon/AdamW primitives. These consume analytic gradients;
//! they do not implement a model backward pass or replace the BO optimizer.
pub use baux::semantics::muon::{POLAR, Polynomial, Shape, Step, reference};

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod probe;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod bank {
    use super::{Shape, Step, metal};
    use ::metal::{BufferRef, CommandBufferRef};

    #[derive(Clone, Copy)]
    pub struct MatrixSlice {
        offset: u64,
        shape: Shape,
    }

    #[derive(Clone, Copy)]
    pub struct VectorSlice {
        offset: u64,
        count: u32,
    }

    pub struct Optimizer {
        matrices: Vec<(metal::Muon, u64)>,
        matrix_adams: Vec<(metal::Adam, u64)>,
        vectors: Vec<(metal::Adam, u64)>,
        workspace: Option<metal::Workspace>,
        matrix_step: Step,
        vector_step: Step,
        is_adamw: bool,
    }

    impl MatrixSlice {
        pub fn new(offset: u64, rows: u32, cols: u32, batch: u32) -> Result<Self, String> {
            Ok(Self {
                offset,
                shape: Shape::new(rows, cols, batch)?,
            })
        }
    }

    impl VectorSlice {
        pub const fn new(offset: u64, count: u32) -> Self {
            Self { offset, count }
        }
    }

    impl Optimizer {
        pub fn new(
            weights: &BufferRef,
            matrices: &[MatrixSlice],
            vectors: &[VectorSlice],
            workspace: (u32, u32, u32),
        ) -> Result<Self, String> {
            let matrices = matrices
                .iter()
                .map(|bank| {
                    metal::Muon::new(bank.shape, weights, bank.offset).map(|v| (v, bank.offset))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let vectors = vectors
                .iter()
                .map(|bank| {
                    metal::Adam::new(weights, bank.offset, bank.count).map(|v| (v, bank.offset))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Self {
                matrices,
                matrix_adams: Vec::new(),
                vectors,
                workspace: Some(metal::Workspace::new(Shape::new(
                    workspace.0,
                    workspace.1,
                    workspace.2,
                )?)?),
                matrix_step: Step::muon(),
                vector_step: Step::adamw(),
                is_adamw: false,
            })
        }

        pub fn adamw(
            weights: &BufferRef,
            matrices: &[MatrixSlice],
            vectors: &[VectorSlice],
        ) -> Result<Self, String> {
            let matrix_adams = matrices
                .iter()
                .map(|bank| {
                    metal::Adam::new(weights, bank.offset, bank.shape.elements() as u32)
                        .map(|v| (v, bank.offset))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let vectors = vectors
                .iter()
                .map(|bank| {
                    metal::Adam::new(weights, bank.offset, bank.count).map(|v| (v, bank.offset))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Self {
                matrices: Vec::new(),
                matrix_adams,
                vectors,
                workspace: None,
                matrix_step: Step::adamw(),
                vector_step: Step::adamw(),
                is_adamw: true,
            })
        }

        pub fn encode(
            &mut self,
            command: &CommandBufferRef,
            weights: &BufferRef,
            gradients: &BufferRef,
        ) -> Result<(), String> {
            if self.is_adamw {
                for (optimizer, offset) in &mut self.matrix_adams {
                    optimizer.encode(
                        command,
                        weights,
                        *offset,
                        gradients,
                        *offset * 2,
                        self.matrix_step,
                    )?;
                }
            } else if let Some(workspace) = &mut self.workspace {
                for (optimizer, offset) in &mut self.matrices {
                    optimizer.encode(
                        workspace,
                        command,
                        weights,
                        *offset,
                        gradients,
                        *offset * 2,
                        self.matrix_step,
                    )?;
                }
            }
            for (optimizer, offset) in &mut self.vectors {
                optimizer.encode(
                    command,
                    weights,
                    *offset,
                    gradients,
                    *offset * 2,
                    self.vector_step,
                )?;
            }
            Ok(())
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use bank::{MatrixSlice, Optimizer, VectorSlice};
