//! General StableHLO gather, including batching, collapsed windows and
//! arbitrary index-vector position. Clamp starts before adding window offsets.
use super::RawAny as Value;
use super::compiler::binary;
use super::contraction::dimensions;
use super::element::{clamp, compute, index, load};
use super::graph::Node;
use crate::portable::Result;

pub(super) fn lower(node: &Node, args: &[Value], shapes: &[Vec<i64>]) -> Result<Value> {
    let data_shape = &shapes[node.inputs[0]];
    let index_shape = &shapes[node.inputs[1]];
    let numbers = node.text("dimension_numbers")?;
    let offsets = dimensions(numbers, "offset_dims")?;
    let collapsed = dimensions(numbers, "collapsed_slice_dims")?;
    let mapping = dimensions(numbers, "start_index_map")?;
    let data_batch = dimensions(numbers, "operand_batching_dims")?;
    let index_batch = dimensions(numbers, "start_indices_batching_dims")?;
    // MLIR's canonical printer omits the default value, index_vector_dim=0.
    let vector = numbers
        .split_once("index_vector_dim = ")
        .map(|(_, tail)| {
            tail.split(|c: char| !c.is_ascii_digit())
                .next()
                .unwrap_or("")
                .parse::<usize>()
                .map_err(|e| e.to_string())
        })
        .transpose()?
        .unwrap_or(0);
    let windows = node.integers("slice_sizes")?;
    let batch_output = (0..node.shape.len())
        .filter(|i| !offsets.contains(i))
        .collect::<Vec<_>>();
    let window_input = (0..data_shape.len())
        .filter(|i| !collapsed.contains(i) && !data_batch.contains(i))
        .collect::<Vec<_>>();
    compute(&node.shape, "gather", |output| {
        let mut indices = vec![];
        let mut batch = 0;
        for axis in 0..index_shape.len() {
            if axis == vector {
                indices.push(index(0)?);
            } else {
                indices.push(output[batch_output[batch]].clone());
                batch += 1;
            }
        }
        let mut source = vec![];
        for (axis, &extent) in data_shape.iter().enumerate() {
            let mut start = if let Some(position) = mapping.iter().position(|&i| i == axis) {
                let mut location = indices.clone();
                if vector < index_shape.len() {
                    location[vector] = index(position as i64)?;
                }
                clamp(load(&args[1], &location)?, extent - windows[axis])?
            } else {
                index(0)?
            };
            if let Some(position) = data_batch.iter().position(|&i| i == axis) {
                start = indices[index_batch[position]].clone();
            }
            if let Some(position) = window_input.iter().position(|&i| i == axis) {
                start = binary("prim._OpAdd", start, output[offsets[position]].clone())?;
            }
            source.push(start);
        }
        load(&args[0], &source)
    })
}
