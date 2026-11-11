//! StableHLO index transformations lowered to native TE, without host
//! execution. Semantics: https://openxla.org/stablehlo/spec.
use super::RawAny as Value;
use super::compiler::{array, binary, call, integer};
use super::element::{cast, clamp, compute, index, load, select};
use super::graph::Node;
use crate::portable::Result;

fn pad(node: &Node, args: &[Value], shape: &[i64]) -> Result<Value> {
    let low = node.integers("edge_padding_low")?;
    let interior = node.integers("interior_padding")?;
    compute(&node.shape, "pad", |output| {
        let mut valid = integer("bool", 1)?;
        let mut indices = vec![];
        for (axis, o) in output.iter().enumerate() {
            let offset = binary("prim._OpSub", o.clone(), index(low[axis])?)?;
            let stride = index(interior[axis] + 1)?;
            let position = binary("prim._OpFloorDiv", offset.clone(), stride.clone())?;
            let aligned = binary(
                "prim._OpEQ",
                binary("prim._OpFloorMod", offset, stride)?,
                index(0)?,
            )?;
            let lower = binary("prim._OpGE", position.clone(), index(0)?)?;
            let upper = binary("prim._OpLT", position.clone(), index(shape[axis])?)?;
            valid = binary(
                "prim._OpAnd",
                valid,
                binary("prim._OpAnd", aligned, binary("prim._OpAnd", lower, upper)?)?,
            )?;
            indices.push(position);
        }
        select(valid, load(&args[0], &indices)?, load(&args[1], &[])?)
    })
}

fn slice(node: &Node, args: &[Value], shape: &[i64], dynamic: bool) -> Result<Value> {
    compute(&node.shape, "slice", |output| {
        let mut indices = vec![];
        for (axis, value) in output.iter().enumerate() {
            let (start, offset) = if dynamic {
                (
                    clamp(load(&args[axis + 1], &[])?, shape[axis] - node.shape[axis])?,
                    value.clone(),
                )
            } else {
                (
                    index(node.integers("start_indices")?[axis])?,
                    binary(
                        "prim._OpMul",
                        value.clone(),
                        index(node.integers("strides")?[axis])?,
                    )?,
                )
            };
            indices.push(binary("prim._OpAdd", start, offset)?);
        }
        load(&args[0], &indices)
    })
}

fn update(node: &Node, args: &[Value], shape: &[i64]) -> Result<Value> {
    compute(&node.shape, "update", |output| {
        let mut inside = integer("bool", 1)?;
        let mut indices = vec![];
        for (axis, value) in output.iter().enumerate() {
            let start = clamp(load(&args[axis + 2], &[])?, node.shape[axis] - shape[axis])?;
            let offset = binary("prim._OpSub", value.clone(), start)?;
            let lower = binary("prim._OpGE", offset.clone(), index(0)?)?;
            let upper = binary("prim._OpLT", offset.clone(), index(shape[axis])?)?;
            inside = binary("prim._OpAnd", inside, binary("prim._OpAnd", lower, upper)?)?;
            indices.push(offset);
        }
        select(inside, load(&args[1], &indices)?, load(&args[0], output)?)
    })
}

pub(super) fn lower(node: &Node, args: &[Value], shapes: &[Vec<i64>]) -> Result<Value> {
    match node.name.as_str() {
        "stablehlo.gather" => super::gather::lower(node, args, shapes),
        "stablehlo.iota" => compute(&node.shape, "iota", |indices| {
            cast(
                indices[node.integer("iota_dimension")? as usize].clone(),
                &node.dtype,
            )
        }),
        "stablehlo.convert" => call(
            "topi.cast",
            &mut [args[0].clone(), Value::string(&node.dtype)?],
        ),
        "stablehlo.concatenate" => call(
            "topi.concatenate",
            &mut [array(args)?, Value::integer(node.integer("dimension")?)],
        ),
        "stablehlo.select" => call("topi.where", &mut args.to_vec()),
        "stablehlo.slice" => slice(node, args, &shapes[node.inputs[0]], false),
        "stablehlo.dynamic_slice" => slice(node, args, &shapes[node.inputs[0]], true),
        "stablehlo.dynamic_update_slice" => update(node, args, &shapes[node.inputs[1]]),
        "stablehlo.pad" => pad(node, args, &shapes[node.inputs[0]]),
        "stablehlo.reverse" => {
            let mut result = args[0].clone();
            for &axis in &node.dimensions {
                result = call("topi.flip", &mut [result, Value::integer(axis)])?;
            }
            Ok(result)
        }
        _ => Err(format!("unsupported StableHLO operation {}", node.name)),
    }
}
