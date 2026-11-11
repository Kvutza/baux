//! Owned compiler interchange, shared only with the native MLIR importer.
use deser::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize)]
pub(super) enum Attribute {
    Integer(i64),
    Integers(Vec<i64>),
    Text(String),
}

#[derive(Serialize, Deserialize)]
pub(super) struct Node {
    pub name: String,
    pub inputs: Vec<usize>,
    pub shape: Vec<i64>,
    pub dtype: String,
    pub dimensions: Vec<i64>,
    pub reducer: String,
    pub dot: String,
    pub constant: Vec<u8>,
    pub attributes: BTreeMap<String, Attribute>,
}
#[derive(Serialize, Deserialize)]
pub(super) struct Graph {
    pub inputs: Vec<Vec<i64>>,
    pub input_dtypes: Vec<String>,
    pub nodes: Vec<Node>,
    pub outputs: Vec<usize>,
}

impl Node {
    pub fn integer(&self, key: &str) -> Result<i64, String> {
        match self.attributes.get(key) {
            Some(Attribute::Integer(n)) => Ok(*n),
            _ => Err(format!("{} requires integer attribute {key}", self.name)),
        }
    }
    pub fn integers(&self, key: &str) -> Result<&[i64], String> {
        match self.attributes.get(key) {
            Some(Attribute::Integers(values)) => Ok(values),
            _ => Err(format!("{} requires integer array {key}", self.name)),
        }
    }
    pub fn text(&self, key: &str) -> Result<&str, String> {
        match self.attributes.get(key) {
            Some(Attribute::Text(value)) => Ok(value),
            _ => Err(format!("{} requires attribute {key}", self.name)),
        }
    }
}
