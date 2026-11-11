//! Native TVM compilation. Pass ordering follows TVM 0.27 s_tir/pipeline.py.
use super::{RawAny as Value, TvmFunction, TvmModule};
use crate::portable::Result;

pub(super) fn call(name: &str, args: &mut [Value]) -> Result<Value> {
    TvmFunction::global(name)?
        .ok_or_else(|| format!("missing native compiler function {name}"))?
        .call(args)
        .map_err(|e| format!("{name}: {e}"))
}
fn text(s: &str) -> Result<Value> {
    Value::string(s)
}

pub(super) fn primitive(dtype: &str) -> Result<Value> {
    call("ir.PrimType", &mut [text(dtype)?])
}

pub(super) fn integer(dtype: &str, value: i64) -> Result<Value> {
    call(
        "ir.IntImm",
        &mut [text(dtype)?, Value::integer(value), Value::none()],
    )
}

pub(super) fn floating(dtype: &str, value: f64) -> Result<Value> {
    call(
        "ir.FloatImm",
        &mut [text(dtype)?, Value::float(value), Value::none()],
    )
}

fn variable(dtype: &str, name: &str) -> Result<Value> {
    call(
        "ir.Var",
        &mut [text(name)?, primitive(dtype)?, Value::none()],
    )
}

pub(super) fn binary(name: &str, lhs: Value, rhs: Value) -> Result<Value> {
    call(name, &mut [lhs, rhs, Value::none()])
}

fn extreme(dtype: &str, name: &str, lhs: Value, rhs: Value) -> Result<Value> {
    let value = binary(
        if name == "minimum" {
            "prim._OpMin"
        } else {
            "prim._OpMax"
        },
        lhs.clone(),
        rhs.clone(),
    )?;
    if !dtype.starts_with("float") && dtype != "bfloat16" {
        return Ok(value);
    }
    let zero = floating(dtype, 0.0)?;
    let lhs_zero = binary("prim._OpEQ", lhs.clone(), zero.clone())?;
    let rhs_zero = binary("prim._OpEQ", rhs.clone(), zero)?;
    let both_zero = binary("prim._OpAnd", lhs_zero, rhs_zero)?;
    let integer = match dtype {
        "float16" | "bfloat16" => "uint16",
        "float32" => "uint32",
        "float64" => "uint64",
        _ => return Err(format!("unsupported floating extrema dtype {dtype}")),
    };
    let lhs_bits = call(
        "tirx.reinterpret",
        &mut [primitive(integer)?, lhs.clone(), Value::none()],
    )?;
    let rhs_bits = call(
        "tirx.reinterpret",
        &mut [primitive(integer)?, rhs.clone(), Value::none()],
    )?;
    let zero_bits = binary(
        if name == "minimum" {
            "prim.bitwise_or"
        } else {
            "prim.bitwise_and"
        },
        lhs_bits,
        rhs_bits,
    )?;
    let zero_value = call(
        "tirx.reinterpret",
        &mut [primitive(dtype)?, zero_bits, Value::none()],
    )?;
    let value = call(
        "prim._OpIfThenElse",
        &mut [both_zero, zero_value, value, Value::none()],
    )?;
    let rhs_nan = call("tirx.isnan", &mut [rhs.clone(), Value::none()])?;
    let value = call(
        "prim._OpIfThenElse",
        &mut [rhs_nan, rhs, value, Value::none()],
    )?;
    let lhs_nan = call("tirx.isnan", &mut [lhs.clone(), Value::none()])?;
    call(
        "prim._OpIfThenElse",
        &mut [lhs_nan, lhs, value, Value::none()],
    )
}

pub(super) fn combine(dtype: &str, reducer: &str, lhs: Value, rhs: Value) -> Result<Value> {
    match reducer {
        "add" => binary("prim._OpAdd", lhs, rhs),
        "multiply" => binary("prim._OpMul", lhs, rhs),
        "minimum" | "maximum" => extreme(dtype, reducer, lhs, rhs),
        "and" => binary(
            if dtype == "bool" {
                "prim._OpAnd"
            } else {
                "prim.bitwise_and"
            },
            lhs,
            rhs,
        ),
        "or" => binary(
            if dtype == "bool" {
                "prim._OpOr"
            } else {
                "prim.bitwise_or"
            },
            lhs,
            rhs,
        ),
        "xor" => binary("prim.bitwise_xor", lhs, rhs),
        _ => Err(format!("unsupported StableHLO reducer {reducer}")),
    }
}

fn identity(dtype: &str, reducer: &str) -> Result<Value> {
    match reducer {
        "add" | "or" | "xor" => {
            if dtype.starts_with("float") || dtype == "bfloat16" {
                floating(dtype, 0.0)
            } else {
                integer(dtype, 0)
            }
        }
        "multiply" => {
            if dtype.starts_with("float") || dtype == "bfloat16" {
                floating(dtype, 1.0)
            } else {
                integer(dtype, 1)
            }
        }
        "and" if dtype == "bool" => integer(dtype, 1),
        "and" => call("prim.max_value", &mut [text(dtype)?, Value::none()]),
        "minimum" if dtype.starts_with("float") || dtype == "bfloat16" => {
            floating(dtype, f64::INFINITY)
        }
        "maximum" if dtype.starts_with("float") || dtype == "bfloat16" => {
            floating(dtype, f64::NEG_INFINITY)
        }
        "minimum" => call("prim.max_value", &mut [text(dtype)?, Value::none()]),
        "maximum" => call("prim.min_value", &mut [text(dtype)?, Value::none()]),
        _ => Err(format!("unsupported StableHLO reducer {reducer}")),
    }
}

pub(super) fn iteration(extent: i64, name: &str, kind: i64) -> Result<(Value, Value)> {
    let variable = variable("int64", name)?;
    let range = call(
        "ir.Range_from_min_extent",
        &mut [
            integer("int64", 0)?,
            integer("int64", extent)?,
            Value::none(),
        ],
    )?;
    let iteration = call(
        "tirx.IterVar",
        &mut [
            range,
            variable.clone(),
            Value::integer(kind),
            text("")?,
            Value::none(),
        ],
    )?;
    Ok((iteration, variable))
}

/// Build a bounded-size TE commutative reduction with StableHLO semantics.
pub(super) fn reduction(
    data: Value,
    initial: Value,
    shape: &[i64],
    axes: &[i64],
    dtype: &str,
    reducer: &str,
    name: &str,
) -> Result<Value> {
    let mut output_axes = Vec::new();
    let mut reduction_axes = Vec::new();
    let mut indices = Vec::with_capacity(shape.len());
    for (axis, &extent) in shape.iter().enumerate() {
        let reduced = axes.contains(&(axis as i64));
        let (iteration, variable) = iteration(
            extent,
            &format!("{}{}", if reduced { "r" } else { "i" }, axis),
            if reduced { 2 } else { 0 },
        )?;
        if reduced {
            reduction_axes.push(iteration);
        } else {
            output_axes.push(iteration);
        }
        indices.push(variable);
    }
    let source = call("te.TensorLoad", &mut [data, array(&indices)?])?;
    let initial = call("te.TensorLoad", &mut [initial, array(&[])?])?;
    if axes.is_empty() {
        let body = combine(dtype, reducer, initial, source)?;
        let operation = call(
            "te.ComputeOp",
            &mut [
                text(name)?,
                text("")?,
                Value::none(),
                array(&output_axes)?,
                array(&[body])?,
            ],
        )?;
        return call("te.OpGetOutput", &mut [operation, Value::integer(0)]);
    }
    let lhs = variable(dtype, "lhs")?;
    let rhs = variable(dtype, "rhs")?;
    let result = combine(dtype, reducer, lhs.clone(), rhs.clone())?;
    let combiner = call(
        "te.CommReducer",
        &mut [
            array(&[lhs])?,
            array(&[rhs])?,
            array(&[result])?,
            array(&[identity(dtype, reducer)?])?,
            Value::none(),
        ],
    )?;
    let condition = integer("bool", 1)?;
    let reduced = call(
        "te.Reduce",
        &mut [
            combiner,
            array(&[source])?,
            array(&reduction_axes)?,
            condition,
            Value::integer(0),
            array(&[])?,
            Value::none(),
        ],
    )?;
    let reduction_operation = call(
        "te.ComputeOp",
        &mut [
            text(&format!("{name}_raw"))?,
            text("")?,
            Value::none(),
            array(&output_axes)?,
            array(&[reduced])?,
        ],
    )?;
    let reduced_tensor = call(
        "te.OpGetOutput",
        &mut [reduction_operation, Value::integer(0)],
    )?;

    let mut final_axes = Vec::with_capacity(output_axes.len());
    let mut final_indices = Vec::with_capacity(output_axes.len());
    for (axis, &extent) in shape.iter().enumerate() {
        if !axes.contains(&(axis as i64)) {
            let (iteration, variable) = iteration(extent, &format!("o{axis}"), 0)?;
            final_axes.push(iteration);
            final_indices.push(variable);
        }
    }
    let reduced = call(
        "te.TensorLoad",
        &mut [reduced_tensor, array(&final_indices)?],
    )?;
    let body = combine(dtype, reducer, initial, reduced)?;
    let final_operation = call(
        "te.ComputeOp",
        &mut [
            text(name)?,
            text("")?,
            Value::none(),
            array(&final_axes)?,
            array(&[body])?,
        ],
    )?;
    call("te.OpGetOutput", &mut [final_operation, Value::integer(0)])
}
pub(super) fn array(values: &[Value]) -> Result<Value> {
    call("ffi.Array", &mut values.to_vec())
}
fn attr(function: Value, name: &str, value: Value) -> Result<Value> {
    call("ir.BaseFuncWithAttr", &mut [function, text(name)?, value])
}
fn pass(module: Value, name: &str, args: &mut [Value]) -> Result<Value> {
    let transform = call(name, args)?;
    call("transform.RunPass", &mut [transform, module])
}
pub(super) fn placeholder(shape: &[i64], dtype: &str, name: &str) -> Result<Value> {
    let dims = shape
        .iter()
        .map(|&n| {
            call(
                "ir.IntImm",
                &mut [text("int64")?, Value::integer(n), Value::none()],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    call(
        "te.Placeholder",
        &mut [array(&dims)?, text(dtype)?, text(name)?],
    )
}

pub(super) fn build(args: &[Value]) -> Result<TvmModule> {
    let function = call("te.CreatePrimFunc", &mut [array(args)?, Value::none()])?;
    let function = attr(function, "global_symbol", text("kernel")?)?;
    let target = call("target.Target", &mut [text("llvm")?])?;
    let target = call("target.WithHost", &mut [target.clone(), target])?;
    let function = attr(function, "target", target.clone())?;
    let mut module = call(
        "ir.Module_FromExpr",
        &mut [function, call("ffi.Map", &mut [])?],
    )?;
    for name in [
        "s_tir.transform.CanonicalizeLoop",
        "s_tir.transform.LowerInitBlock",
        "s_tir.transform.PlanAndUpdateBufferAllocationLocation",
        "s_tir.transform.ConvertBlocksToOpaque",
        "s_tir.transform.CompactBufferAllocation",
        "s_tir.transform.LowerMatchBuffer",
        "s_tir.transform.StmtSimplify",
        "s_tir.transform.LowerOpaqueBlock",
        "tirx.transform.FlattenBuffer",
        "tirx.transform.BF16ComputeLegalize",
        "tirx.transform.StorageRewrite",
        "tirx.transform.UnrollLoop",
        "tirx.transform.StmtSimplify",
        "tirx.transform.RemoveNoOp",
        "tirx.transform.VerifyMemory",
        "tirx.transform.AnnotateEntryFunc",
        "tirx.transform.SplitHostDevice",
        "tirx.transform.MakePackedAPI",
        "tirx.transform.BF16StorageLegalize",
        "tirx.transform.LowerTVMBuiltin",
        "tirx.transform.LowerIntrin",
    ] {
        module = if name.ends_with("CompactBufferAllocation") {
            pass(module, name, &mut [Value::boolean(true)])?
        } else {
            pass(module, name, &mut [])?
        };
    }
    let mut result = call("target.build.llvm", &mut [module, target])?;
    if result.type_index != 73 {
        return Err("TVM codegen returned a non-module".into());
    }
    Ok(TvmModule(
        std::ptr::NonNull::new(result.take()).ok_or("null compiler output")?,
    ))
}

fn tensor_type(shape: &[i64], dtype: &str) -> Result<Value> {
    let dims = shape
        .iter()
        .map(|&n| {
            call(
                "ir.IntImm",
                &mut [text("int64")?, Value::integer(n), Value::none()],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let shape = call("relax.ShapeExpr", &mut [array(&dims)?, Value::none()])?;
    call(
        "relax.TensorType",
        &mut [
            shape,
            text(dtype)?,
            Value::integer(-1),
            Value::none(),
            Value::none(),
        ],
    )
}

pub(super) fn vm(
    kernel: &TvmModule,
    inputs: &[Vec<i64>],
    input_dtypes: &[String],
    outputs: &[Vec<i64>],
    output_dtypes: &[String],
) -> Result<TvmModule> {
    let params = inputs
        .iter()
        .zip(input_dtypes)
        .enumerate()
        .map(|(i, (shape, dtype))| {
            call(
                "ir.Var",
                &mut [
                    text(&format!("arg{i}"))?,
                    tensor_type(shape, dtype)?,
                    Value::none(),
                ],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let types = outputs
        .iter()
        .zip(output_dtypes)
        .map(|(shape, dtype)| tensor_type(shape, dtype))
        .collect::<Result<Vec<_>>>()?;
    let arguments = call("ir.Tuple", &mut [array(&params)?, Value::none()])?;
    let external = call(
        "relax.ExternFunc",
        &mut [text("kernel")?, Value::none(), Value::none()],
    )?;
    let body = call(
        "relax.op.call_dps_packed",
        &mut [external, arguments, array(&types)?],
    )?;
    finish(Some(kernel), params, body, 1)
}

pub(super) fn external(
    name: &str,
    inputs: &[super::super::Tensor],
    outputs: &[super::super::Tensor],
    device: i64,
) -> Result<TvmModule> {
    let signature = |spec: &super::super::Tensor| {
        let shape = spec
            .shape
            .iter()
            .map(|&n| i64::try_from(n).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>>>()?;
        tensor_type(&shape, &spec.dtype)
    };
    let params = inputs
        .iter()
        .enumerate()
        .map(|(i, spec)| {
            call(
                "ir.Var",
                &mut [text(&format!("arg{i}"))?, signature(spec)?, Value::none()],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let types = outputs.iter().map(signature).collect::<Result<Vec<_>>>()?;
    let result = call("ir.TupleType", &mut [array(&types)?, Value::none()])?;
    let function = call(
        "relax.ExternFunc",
        &mut [text(name)?, Value::none(), Value::none()],
    )?;
    let body = call(
        "ir.Call",
        &mut [
            result.clone(),
            function,
            array(&params)?,
            Value::none(),
            array(&[result])?,
            Value::none(),
        ],
    )?;
    finish(None, params, body, device)
}

fn finish(
    kernel: Option<&TvmModule>,
    params: Vec<Value>,
    body: Value,
    device: i64,
) -> Result<TvmModule> {
    let builder = call("relax.BlockBuilderCreate", &mut [Value::none()])?;
    call(
        "relax.BlockBuilderBeginScope",
        &mut [builder.clone(), array(&params)?],
    )?;
    call(
        "relax.BlockBuilderBeginBindingBlock",
        &mut [builder.clone()],
    )?;
    let body = call(
        "relax.BlockBuilderEmit",
        &mut [builder.clone(), body, text("result")?],
    )?;
    let block = call("relax.BlockBuilderEndBlock", &mut [builder.clone()])?;
    let body = call(
        "relax.SeqExpr",
        &mut [array(&[block])?, body, Value::none()],
    )?;
    let body = call("relax.BlockBuilderNormalize", &mut [builder.clone(), body])?;
    call("relax.BlockBuilderEndScope", &mut [builder.clone()])?;
    let attrs = call(
        "ffi.MakeObjectFromPackedArgs",
        &mut [
            text("ir.DictAttrs")?,
            text("__dict__")?,
            call("ffi.Map", &mut [])?,
        ],
    )?;
    let function = call(
        "relax.Function",
        &mut [
            array(&params)?,
            body,
            Value::none(),
            Value::boolean(kernel.is_some()),
            attrs,
            Value::none(),
        ],
    )?;
    let function = call("relax.BlockBuilderNormalize", &mut [builder, function])?;
    let function = attr(function, "global_symbol", text("main")?)?;
    let mut module = call(
        "ir.Module_FromExpr",
        &mut [function, call("ffi.Map", &mut [])?],
    )?;
    for name in [
        "ToNonDataflow",
        "RemovePurityChecking",
        "CallTIRRewrite",
        "StaticPlanBlockMemory",
        "LowerAllocTensor",
        "KillAfterLastUse",
        "LowerRuntimeBuiltin",
        "ComputePrimValue",
        "VMShapeLower",
        "AttachGlobalSymbol",
    ] {
        let mut args = if name == "VMShapeLower" {
            vec![Value::boolean(true)]
        } else {
            vec![]
        };
        module = pass(module, &format!("relax.transform.{name}"), &mut args)?;
    }
    let builder = call("relax.ExecBuilderCreate", &mut [])?;
    call("relax.VMCodeGen", &mut [builder.clone(), module])?;
    let target = call("target.Target", &mut [text("llvm")?])?;
    let library = if let Some(kernel) = kernel {
        super::status(
            unsafe { super::ffi_increment(kernel.0.as_ptr()) },
            "retain kernel",
        )?;
        Value {
            type_index: 73,
            padding: 0,
            value: super::RawValue {
                pointer: kernel.0.as_ptr(),
            },
        }
    } else {
        Value::none()
    };
    let mut result = call(
        "relax.VMLink",
        &mut [
            builder,
            target,
            library,
            array(&[])?,
            call("ffi.Map", &mut [])?,
        ],
    )?;
    if result.type_index != 73 {
        return Err("TVM linker returned a non-module".into());
    }
    let executable = TvmModule(std::ptr::NonNull::new(result.take()).ok_or("null VM executable")?);
    let mut result = executable
        .function("vm_load_executable")?
        .ok_or("missing VM loader")?
        .call(&mut [])?;
    if result.type_index != 73 {
        return Err("TVM loader returned a non-module".into());
    }
    let vm = TvmModule(std::ptr::NonNull::new(result.take()).ok_or("null VM")?);
    let mut devices = vec![Value::integer(device), Value::integer(0), Value::integer(2)];
    if device != 1 {
        devices.extend([Value::integer(1), Value::integer(0), Value::integer(2)]);
    }
    vm.function("vm_initialization")?
        .ok_or("missing VM initialization")?
        .call(&mut devices)?;
    Ok(vm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portable::{Tensor, TvmTensor};
    #[test]
    fn llvm_kernel() {
        let a = placeholder(&[4], "float32", "a").unwrap();
        let b = placeholder(&[4], "float32", "b").unwrap();
        let c = call("topi.add", &mut [a.clone(), b.clone()]).unwrap();
        let module = build(&[a, b, c]).unwrap();
        let kernel = module.function("kernel").unwrap().unwrap();
        let spec = Tensor::new(vec![4], "float32").unwrap();
        let bytes = [1f32, 2., 3., 4.]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        let a = TvmTensor::put(spec.clone(), &bytes).unwrap();
        let b = TvmTensor::put(spec.clone(), &bytes).unwrap();
        let c = TvmTensor::put(spec, &[0; 16]).unwrap();
        kernel
            .call(&mut [a.value.clone(), b.value.clone(), c.value.clone()])
            .unwrap();
        let expected = [2f32, 4., 6., 8.]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        assert_eq!(c.read().unwrap(), expected);
        let vm = vm(
            &module,
            &[vec![4], vec![4]],
            &["float32".into(), "float32".into()],
            &[vec![4]],
            &["float32".into()],
        )
        .unwrap();
        let result = vm
            .function("main")
            .unwrap()
            .unwrap()
            .run(&[&a, &b])
            .unwrap();
        assert_eq!(result[0].read().unwrap(), expected);
    }
}
