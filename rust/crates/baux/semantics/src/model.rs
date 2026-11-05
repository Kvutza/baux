//! Typed model dataflow shared by accelerator implementations.
use super::graph::{BackendCapability, SemanticNode};
use super::recurrence::{LayerVisit, RecurrentCore};
use super::tensor::{Extent, TensorDType, TensorSpec, TensorStorage};
use std::collections::{HashMap, HashSet};

#[path = "model/programs.rs"]
mod programs;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProgramKind {
    Prefill,
    Decode,
    Draft,
    Verify,
    Score,
    Objective,
    ExperimentRound,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramFunction {
    pub name: &'static str,
    pub kind: ProgramKind,
    pub nodes: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightArenaSpec {
    pub name: &'static str,
    pub overlay: &'static str,
    pub dtype: TensorDType,
    pub exact: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProgram {
    pub name: &'static str,
    pub version: u32,
    pub tensors: Vec<TensorSpec>,
    pub nodes: Vec<SemanticNode>,
    pub functions: Vec<ProgramFunction>,
    pub weight_arenas: Vec<WeightArenaSpec>,
    pub requirements: Vec<BackendCapability>,
    pub physical_layers: usize,
    pub recurrent_core: Option<RecurrentCore>,
}

impl ModelProgram {
    pub const VERSION: u32 = 3;

    pub fn weight_tensors(&self) -> Vec<&TensorSpec> {
        self.tensors
            .iter()
            .filter(|tensor| tensor.storage == TensorStorage::WeightArena)
            .collect()
    }

    pub fn parameter_count(&self) -> u64 {
        self.weight_tensors()
            .into_iter()
            .map(|tensor| {
                tensor
                    .shape
                    .iter()
                    .map(|extent| match *extent {
                        Extent::Fixed(count) => count as u64,
                        Extent::Symbol(_) => 0,
                    })
                    .product::<u64>()
            })
            .sum()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() || self.version != Self::VERSION {
            return Err("model program identity is invalid".into());
        }
        let tensors = self
            .tensors
            .iter()
            .map(|tensor| (tensor.name, tensor))
            .collect::<HashMap<_, _>>();
        if tensors.len() != self.tensors.len()
            || self.tensors.iter().any(|tensor| {
                tensor.name.is_empty()
                    || tensor.shape.is_empty()
                    || tensor.shape.contains(&Extent::Fixed(0))
            })
        {
            return Err("model program tensors must have unique names and nonzero shapes".into());
        }
        let nodes = self
            .nodes
            .iter()
            .map(|node| (node.name, node))
            .collect::<HashMap<_, _>>();
        if nodes.len() != self.nodes.len() || self.nodes.iter().any(|node| node.name.is_empty()) {
            return Err("model program nodes must have unique names".into());
        }
        for node in &self.nodes {
            if node
                .inputs
                .iter()
                .chain(&node.outputs)
                .any(|name| !tensors.contains_key(name))
            {
                return Err(format!("node {} references an unknown tensor", node.name));
            }
        }
        let mut kinds = HashSet::new();
        for function in &self.functions {
            if function.name.is_empty()
                || function.nodes.is_empty()
                || !kinds.insert(function.kind)
                || function.nodes.iter().any(|name| !nodes.contains_key(name))
            {
                return Err(format!("function {} is invalid", function.name));
            }
        }
        let requirements = self.requirements.iter().copied().collect::<HashSet<_>>();
        if requirements.len() != self.requirements.len() {
            return Err("model program backend requirements must be unique".into());
        }
        if self.physical_layers == 0 {
            return Err("model program must contain physical layers".into());
        }
        if let Some(core) = self.recurrent_core {
            core.validate(self.physical_layers)?;
        }
        Ok(())
    }

    pub fn layer_visits(&self, visits: usize) -> Result<Vec<LayerVisit>, String> {
        match self.recurrent_core {
            Some(core) => core.layer_visits(self.physical_layers, visits),
            None if visits == 1 => Ok((0..self.physical_layers)
                .map(|layer| LayerVisit {
                    layer,
                    visit: 0,
                    execution: layer,
                    recurrent: false,
                })
                .collect()),
            None => Err("model program has no recurrent core".into()),
        }
    }

    pub fn function(&self, kind: ProgramKind) -> Option<&ProgramFunction> {
        self.functions.iter().find(|function| function.kind == kind)
    }

    pub fn node(&self, name: &str) -> Option<&SemanticNode> {
        self.nodes.iter().find(|node| node.name == name)
    }
}
