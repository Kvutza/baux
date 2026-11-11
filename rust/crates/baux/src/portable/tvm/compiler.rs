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
pub(super) fn placeholder(shape: &[i64], name: &str) -> Result<Value> {
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
        &mut [array(&dims)?, text("float32")?, text(name)?],
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
        "tirx.transform.StorageRewrite",
        "tirx.transform.UnrollLoop",
        "tirx.transform.StmtSimplify",
        "tirx.transform.RemoveNoOp",
        "tirx.transform.VerifyMemory",
        "tirx.transform.AnnotateEntryFunc",
        "tirx.transform.SplitHostDevice",
        "tirx.transform.MakePackedAPI",
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

fn tensor_type(shape: &[i64]) -> Result<Value> {
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
            text("float32")?,
            Value::integer(-1),
            Value::none(),
            Value::none(),
        ],
    )
}

pub(super) fn vm(
    kernel: &TvmModule,
    inputs: &[Vec<i64>],
    outputs: &[Vec<i64>],
) -> Result<TvmModule> {
    let params = inputs
        .iter()
        .enumerate()
        .map(|(i, shape)| {
            call(
                "ir.Var",
                &mut [
                    text(&format!("arg{i}"))?,
                    tensor_type(shape)?,
                    Value::none(),
                ],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let types = outputs
        .iter()
        .map(|shape| tensor_type(shape))
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
            Value::boolean(true),
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
    super::status(
        unsafe { super::ffi_increment(kernel.0.as_ptr()) },
        "retain kernel",
    )?;
    let library = Value {
        type_index: 73,
        padding: 0,
        value: super::RawValue {
            pointer: kernel.0.as_ptr(),
        },
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
    vm.function("vm_initialization")?
        .ok_or("missing VM initialization")?
        .call(&mut [Value::integer(1), Value::integer(0), Value::integer(2)])?;
    Ok(vm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portable::{Tensor, TvmTensor};
    #[test]
    fn llvm_kernel() {
        let a = placeholder(&[4], "a").unwrap();
        let b = placeholder(&[4], "b").unwrap();
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
        let vm = vm(&module, &[vec![4], vec![4]], &[vec![4]]).unwrap();
        let result = vm
            .function("main")
            .unwrap()
            .unwrap()
            .run(&[&a, &b])
            .unwrap();
        assert_eq!(result[0].read().unwrap(), expected);
    }
}
