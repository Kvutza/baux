//! Owned compiler interchange, shared only with the native MLIR importer.
use deser::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub(super) struct Node {
    pub name: String,
    pub inputs: Vec<usize>,
    pub shape: Vec<i64>,
    pub dimensions: Vec<i64>,
    pub dot: String,
    pub constant: Vec<u32>,
}
#[derive(Serialize, Deserialize)]
pub(super) struct Graph {
    pub inputs: Vec<Vec<i64>>,
    pub nodes: Vec<Node>,
    pub outputs: Vec<usize>,
}
