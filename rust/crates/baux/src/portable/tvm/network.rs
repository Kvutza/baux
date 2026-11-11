//! Bounded TE kernels joined by Relax SSA bindings. Boundaries materialize only
//! live values; Relax plans their storage and releases intermediates after use.
//! See TVM's relax/vm_build.py and s_tir/pipeline.py (pinned 0.27).
use super::compiler::{array, call, finish, library, lowered, placeholder, tensor_type};
use super::graph::Graph;
use super::lower::{lower, spec};
use super::{RawAny as Value, TvmModule, TvmTensor};
use crate::portable::Result;
use std::collections::BTreeSet;

struct Kernel {
    name: String,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

pub(super) fn compile(graph: &Graph) -> Result<(TvmModule, Vec<TvmTensor>)> {
    let base = graph.inputs.len();
    let shapes = graph
        .inputs
        .iter()
        .cloned()
        .chain(graph.nodes.iter().map(|node| node.shape.clone()))
        .collect::<Vec<_>>();
    let dtypes = graph
        .input_dtypes
        .iter()
        .cloned()
        .chain(graph.nodes.iter().map(|node| node.dtype.clone()))
        .collect::<Vec<_>>();
    let mut parameters = (0..base).collect::<Vec<_>>();
    let mut constants = Vec::new();
    let mut last = vec![0; shapes.len()];
    for (offset, node) in graph.nodes.iter().enumerate() {
        if node.name == "stablehlo.constant" {
            parameters.push(base + offset);
            constants.push(TvmTensor::put(
                spec(&node.shape, &node.dtype)?,
                &node.constant,
            )?);
        }
        for &input in &node.inputs {
            last[input] = base + offset;
        }
    }
    for &output in &graph.outputs {
        last[output] = shapes.len();
    }
    let constant_ids = parameters.iter().copied().collect::<BTreeSet<_>>();
    let mut module = None;
    let mut kernels = Vec::new();
    // This bounds compiler pass complexity, not tensor dimensions or semantics.
    for start in (base..shapes.len()).step_by(128) {
        let end = (start + 128).min(shapes.len());
        let outputs = (start..end)
            .filter(|id| !constant_ids.contains(id) && last[*id] >= end)
            .collect::<Vec<_>>();
        if outputs.is_empty() {
            continue;
        }
        let inputs = graph.nodes[start - base..end - base]
            .iter()
            .flat_map(|node| node.inputs.iter().copied())
            .filter(|id| *id < start || constant_ids.contains(id))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut values = (0..shapes.len()).map(|_| Value::none()).collect::<Vec<_>>();
        let mut tensors = Vec::new();
        for &id in &inputs {
            values[id] = placeholder(&shapes[id], &dtypes[id], &format!("value{id}"))?;
            tensors.push(values[id].clone());
        }
        for id in start..end {
            if !constant_ids.contains(&id) {
                values[id] = lower(&graph.nodes[id - base], &values, &shapes, &dtypes)?;
            }
        }
        for &id in &outputs {
            tensors.push(call("topi.identity", &mut [values[id].clone()])?);
        }
        let name = format!("kernel{}", kernels.len());
        let part = lowered(&tensors, &name)?;
        if let Some(module) = &module {
            call("ir.Module_Update", &mut [Value::clone(module), part])?;
        } else {
            module = Some(part);
        }
        kernels.push(Kernel {
            name,
            inputs,
            outputs,
        });
    }
    let kernel = module.map(library).transpose()?;
    let executable = connect(
        kernel.as_ref(),
        &kernels,
        &parameters,
        &graph.outputs,
        &shapes,
        &dtypes,
    )?;
    Ok((executable, constants))
}

fn connect(
    library: Option<&TvmModule>,
    kernels: &[Kernel],
    parameters: &[usize],
    outputs: &[usize],
    shapes: &[Vec<i64>],
    dtypes: &[String],
) -> Result<TvmModule> {
    let mut values = (0..shapes.len()).map(|_| Value::none()).collect::<Vec<_>>();
    let mut params = Vec::new();
    for &id in parameters {
        values[id] = call(
            "ir.Var",
            &mut [
                Value::string(&format!("arg{id}"))?,
                tensor_type(&shapes[id], &dtypes[id])?,
                Value::none(),
            ],
        )?;
        params.push(values[id].clone());
    }
    let builder = call("relax.BlockBuilderCreate", &mut [Value::none()])?;
    call(
        "relax.BlockBuilderBeginScope",
        &mut [builder.clone(), array(&params)?],
    )?;
    call(
        "relax.BlockBuilderBeginBindingBlock",
        &mut [builder.clone()],
    )?;
    for kernel in kernels {
        let args = kernel
            .inputs
            .iter()
            .map(|&id| values[id].clone())
            .collect::<Vec<_>>();
        let types = kernel
            .outputs
            .iter()
            .map(|&id| tensor_type(&shapes[id], &dtypes[id]))
            .collect::<Result<Vec<_>>>()?;
        let function = call(
            "relax.ExternFunc",
            &mut [Value::string(&kernel.name)?, Value::none(), Value::none()],
        )?;
        let args = call("ir.Tuple", &mut [array(&args)?, Value::none()])?;
        let body = call(
            "relax.op.call_dps_packed",
            &mut [function, args, array(&types)?],
        )?;
        let result = call(
            "relax.BlockBuilderEmit",
            &mut [builder.clone(), body, Value::string(&kernel.name)?],
        )?;
        for (index, &id) in kernel.outputs.iter().enumerate() {
            values[id] = if kernel.outputs.len() == 1 {
                result.clone()
            } else {
                let field = call(
                    "ir.TupleGetItem",
                    &mut [result.clone(), Value::integer(index as i64), Value::none()],
                )?;
                call(
                    "relax.BlockBuilderEmit",
                    &mut [
                        builder.clone(),
                        field,
                        Value::string(&format!("value{id}"))?,
                    ],
                )?
            };
        }
    }
    let results = outputs
        .iter()
        .map(|&id| values[id].clone())
        .collect::<Vec<_>>();
    let body = call("ir.Tuple", &mut [array(&results)?, Value::none()])?;
    let block = call("relax.BlockBuilderEndBlock", &mut [builder.clone()])?;
    let body = call(
        "relax.SeqExpr",
        &mut [array(&[block])?, body, Value::none()],
    )?;
    let body = call("relax.BlockBuilderNormalize", &mut [builder.clone(), body])?;
    call("relax.BlockBuilderEndScope", &mut [builder])?;
    finish(library, params, body, 1)
}
