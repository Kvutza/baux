//! IEEE total ordering preserves signed zeros and NaN payload ordering.
//! Reference: https://openxla.org/stablehlo/spec#compare.
use super::RawAny as Value;
use super::compiler::{binary, call, integer, primitive};
use super::element::{compute, load, select};
use super::graph::Node;
use crate::portable::Result;

fn ordered(value: Value, dtype: &str) -> Result<Value> {
    let (kind, mask) = match dtype {
        "float16" | "bfloat16" => ("int16", i16::MAX as i64),
        "float32" => ("int32", i32::MAX as i64),
        "float64" => ("int64", i64::MAX),
        _ => {
            return Err(format!(
                "total order requires floating operands, got {dtype}"
            ));
        }
    };
    let bits = call(
        "tirx.reinterpret",
        &mut [primitive(kind)?, value, Value::none()],
    )?;
    let negative = binary("prim._OpLT", bits.clone(), integer(kind, 0)?)?;
    let reflected = binary("prim.bitwise_xor", bits.clone(), integer(kind, mask)?)?;
    select(negative, reflected, bits)
}

pub(super) fn lower(node: &Node, args: &[Value], dtype: &str) -> Result<Value> {
    let direction = node.text("comparison_direction")?;
    let direction = direction
        .rsplit_once(' ')
        .map(|(_, v)| v.trim_end_matches('>'));
    let operation = match direction {
        Some("EQ") => "prim._OpEQ",
        Some("NE") => "prim._OpNE",
        Some("LT") => "prim._OpLT",
        Some("LE") => "prim._OpLE",
        Some("GT") => "prim._OpGT",
        Some("GE") => "prim._OpGE",
        _ => return Err("invalid comparison direction".into()),
    };
    let total = node
        .text("compare_type")
        .is_ok_and(|s| s.contains("TOTALORDER"));
    compute(&node.shape, "compare", |indices| {
        let mut lhs = load(&args[0], indices)?;
        let mut rhs = load(&args[1], indices)?;
        if total {
            lhs = ordered(lhs, dtype)?;
            rhs = ordered(rhs, dtype)?;
        }
        binary(operation, lhs, rhs)
    })
}
