//! Native TE indexing expressions shared by tensor transformations.
use super::RawAny as Value;
use super::compiler::{array, binary, call, floating, integer, iteration};
use crate::portable::Result;

pub(super) fn compute(
    shape: &[i64],
    name: &str,
    body: impl FnOnce(&[Value]) -> Result<Value>,
) -> Result<Value> {
    let mut axes = vec![];
    let mut indices = vec![];
    for (axis, &extent) in shape.iter().enumerate() {
        let (iter, index) = iteration(extent, &format!("i{axis}"), 0)?;
        axes.push(iter);
        indices.push(index);
    }
    let body = body(&indices)?;
    let operation = call(
        "te.ComputeOp",
        &mut [
            Value::string(name)?,
            Value::string("")?,
            Value::none(),
            array(&axes)?,
            array(&[body])?,
        ],
    )?;
    call("te.OpGetOutput", &mut [operation, Value::integer(0)])
}
pub(super) fn load(value: &Value, indices: &[Value]) -> Result<Value> {
    call("te.TensorLoad", &mut [value.clone(), array(indices)?])
}
pub(super) fn index(n: i64) -> Result<Value> {
    integer("int64", n)
}
pub(super) fn cast(value: Value, dtype: &str) -> Result<Value> {
    call(
        "prim.Cast",
        &mut [Value::string(dtype)?, value, Value::none()],
    )
}
pub(super) fn select(condition: Value, yes: Value, no: Value) -> Result<Value> {
    call(
        "prim._OpIfThenElse",
        &mut [condition, yes, no, Value::none()],
    )
}
pub(super) fn clamp(value: Value, high: i64) -> Result<Value> {
    let value = binary("prim._OpMax", cast(value, "int64")?, index(0)?)?;
    binary("prim._OpMin", value, index(high)?)
}

/// Round ties away from zero without x +/- 0.5, which can change large
/// already-integral floating values. Truncation retains signed zero.
pub(super) fn round(value: Value, dtype: &str) -> Result<Value> {
    let base = call("tirx.trunc", &mut [value.clone(), Value::none()])?;
    let fraction = binary("prim._OpSub", value.clone(), base.clone())?;
    let magnitude = call("tirx.abs", &mut [fraction, Value::none()])?;
    let adjust = binary("prim._OpGE", magnitude, floating(dtype, 0.5)?)?;
    let negative = binary("prim._OpLT", value, floating(dtype, 0.0)?)?;
    let step = select(negative, floating(dtype, -1.0)?, floating(dtype, 1.0)?)?;
    let rounded = binary("prim._OpAdd", base.clone(), step)?;
    select(adjust, rounded, base)
}
