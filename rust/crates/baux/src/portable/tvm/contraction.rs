//! StableHLO dot_general dimension semantics expressed as one native TE einsum.
//! Output order is batch, lhs free, rhs free; contracted dimensions are paired
//! in attribute order. Reference: https://openxla.org/stablehlo/spec#dot_general.
use super::RawAny as Value;
use super::compiler::{array, call};
use super::graph::Node;
use crate::portable::Result;

pub(super) fn dimensions(text: &str, key: &str) -> Result<Vec<usize>> {
    let Some((_, tail)) = text.split_once(&format!("{key} = [")) else {
        return Ok(vec![]);
    };
    let (values, _) = tail.split_once(']').ok_or("invalid dot dimensions")?;
    if values.is_empty() {
        return Ok(vec![]);
    }
    values
        .split(',')
        .map(|n| n.trim().parse::<usize>().map_err(|e| e.to_string()))
        .collect()
}

pub(super) fn lower(node: &Node, args: &[Value], shapes: &[Vec<i64>]) -> Result<Value> {
    let lhs = &shapes[node.inputs[0]];
    let rhs = &shapes[node.inputs[1]];
    let lc = dimensions(&node.dot, "lhs_contracting_dimensions")?;
    let rc = dimensions(&node.dot, "rhs_contracting_dimensions")?;
    let lb = dimensions(&node.dot, "lhs_batching_dimensions")?;
    let rb = dimensions(&node.dot, "rhs_batching_dimensions")?;
    let letters = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let mut count = 0;
    let mut label = || {
        let value = letters
            .get(count)
            .copied()
            .ok_or("TVM einsum supports at most 52 distinct dimensions")?;
        count += 1;
        Ok::<_, String>(value as char)
    };
    let mut a = vec![' '; lhs.len()];
    let mut b = vec![' '; rhs.len()];
    let mut output = String::new();
    for (&l, &r) in lb.iter().zip(&rb) {
        let c = label()?;
        a[l] = c;
        b[r] = c;
        output.push(c);
    }
    for (&l, &r) in lc.iter().zip(&rc) {
        let c = label()?;
        a[l] = c;
        b[r] = c;
    }
    for entry in a.iter_mut().chain(b.iter_mut()) {
        if *entry == ' ' {
            *entry = label()?;
            output.push(*entry);
        }
    }
    let expression = format!(
        "{},{}->{output}",
        a.iter().collect::<String>(),
        b.iter().collect::<String>()
    );
    // StableHLO permits a wider result/accumulation type than operand storage.
    // Cast before contraction so rounding does not happen in the input dtype.
    let dtype = if node.dtype == "bool" {
        "int64"
    } else {
        &node.dtype
    };
    let operands = args
        .iter()
        .map(|value| call("topi.cast", &mut [value.clone(), Value::string(dtype)?]))
        .collect::<Result<Vec<_>>>()?;
    let result = call(
        "topi.einsum",
        &mut [Value::string(&expression)?, array(&operands)?],
    )?;
    call("topi.cast", &mut [result, Value::string(&node.dtype)?])
}
