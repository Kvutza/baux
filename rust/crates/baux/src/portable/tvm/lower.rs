//! StableHLO -> native TOPI/TIR kernels -> Relax destination-passing calls ->
//! VM. All compiler operations use the pinned TVM registry; no Python
//! registrations.
use super::compiler::{array, build, call, placeholder, reduction, vm};
use super::graph::{Graph, Node};
use super::{RawAny as Value, TvmModule, TvmTensor};
use crate::portable::{Result, Tensor};
use std::io::Write;
use std::sync::{Arc, Mutex};

fn import(source: &str, entry: &str, importer: &std::path::Path) -> Result<Graph> {
    use std::process::{Command, Stdio};
    let mut child = Command::new(importer)
        .arg(entry)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("native MLIR importer: {e}"))?;
    let mut input = child.stdin.take().ok_or("missing importer stdin")?;
    // Drain output concurrently with writing: a large graph or parse diagnostic
    // must not deadlock the bounded OS pipes.
    let source = source.as_bytes().to_vec();
    let writer = std::thread::spawn(move || input.write_all(&source));
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    let written = writer.join().map_err(|_| "import writer panicked")?;
    if !output.status.success() {
        return Err(format!(
            "StableHLO import failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    written.map_err(|e| e.to_string())?;
    deser_json::from_str(std::str::from_utf8(&output.stdout).map_err(|e| e.to_string())?)
        .map_err(|e| format!("invalid native importer graph: {e}"))
}

fn dims(values: &[i64]) -> Result<Value> {
    let values = values
        .iter()
        .map(|&n| {
            call(
                "ir.IntImm",
                &mut [Value::string("int64")?, Value::integer(n), Value::none()],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    array(&values)
}

fn axes(values: &[i64]) -> Result<Value> {
    array(
        &values
            .iter()
            .map(|&v| Value::integer(v))
            .collect::<Vec<_>>(),
    )
}

fn dot_dims(text: &str, key: &str) -> Result<Vec<usize>> {
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

fn dot(node: &Node, args: &mut [Value], shapes: &[Vec<i64>]) -> Result<Value> {
    if args.len() != 2 {
        return Err("dot requires two operands".into());
    }
    let lhs = &shapes[node.inputs[0]];
    let rhs = &shapes[node.inputs[1]];
    let lc = dot_dims(&node.dot, "lhs_contracting_dimensions")?;
    let rc = dot_dims(&node.dot, "rhs_contracting_dimensions")?;
    let lb = dot_dims(&node.dot, "lhs_batching_dimensions")?;
    let rb = dot_dims(&node.dot, "rhs_batching_dimensions")?;
    if !lb.is_empty() || !rb.is_empty() {
        return Err("native TVM dot_general does not yet support batch dimensions".into());
    }
    let lf = (0..lhs.len())
        .filter(|i| !lc.contains(i))
        .collect::<Vec<_>>();
    let rf = (0..rhs.len())
        .filter(|i| !rc.contains(i))
        .collect::<Vec<_>>();
    let lp = lf.iter().chain(&lc).map(|&i| i as i64).collect::<Vec<_>>();
    let rp = rc.iter().chain(&rf).map(|&i| i as i64).collect::<Vec<_>>();
    let m = lf.iter().map(|&i| lhs[i]).product();
    let n = rf.iter().map(|&i| rhs[i]).product();
    let k = lc.iter().map(|&i| lhs[i]).product();
    let a = call("topi.transpose", &mut [args[0].clone(), axes(&lp)?])?;
    let b = call("topi.transpose", &mut [args[1].clone(), axes(&rp)?])?;
    let a = call("topi.reshape", &mut [a, dims(&[m, k])?])?;
    let b = call("topi.reshape", &mut [b, dims(&[k, n])?])?;
    let result = call(
        "topi.matmul",
        &mut [a, b, Value::boolean(false), Value::boolean(false)],
    )?;
    call("topi.reshape", &mut [result, dims(&node.shape)?])
}

fn lower(node: &Node, values: &[Value], shapes: &[Vec<i64>]) -> Result<Value> {
    let mut args = node
        .inputs
        .iter()
        .map(|&i| values[i].clone())
        .collect::<Vec<_>>();
    let op = node
        .name
        .strip_prefix("stablehlo.")
        .ok_or_else(|| format!("unsupported operation {}", node.name))?;
    let binary = match op {
        "add" => Some("add"),
        "subtract" => Some("subtract"),
        "multiply" => Some("multiply"),
        "divide" => Some("divide"),
        "maximum" => Some("maximum"),
        "minimum" => Some("minimum"),
        _ => None,
    };
    if let Some(name) = binary {
        return call(&format!("topi.{name}"), &mut args);
    }
    match op {
        "reduce" => {
            if node.dimensions.iter().any(|&axis| {
                usize::try_from(axis)
                    .ok()
                    .and_then(|axis| shapes[node.inputs[0]].get(axis))
                    == Some(&0)
            }) {
                return call(
                    "topi.broadcast_to",
                    &mut [args[1].clone(), dims(&node.shape)?],
                );
            }
            let reducer = if node.dtype == "bool" {
                match node.reducer.as_str() {
                    "add" | "maximum" => "or",
                    "multiply" | "minimum" => "and",
                    reducer => reducer,
                }
            } else {
                &node.reducer
            };
            reduction(
                args[0].clone(),
                args[1].clone(),
                &shapes[node.inputs[0]],
                &node.dimensions,
                &node.dtype,
                reducer,
                "stablehlo_reduce",
            )
        }
        "tanh" | "exp" | "log" | "sqrt" | "rsqrt" | "negative" | "abs" | "cos" | "sin" => {
            call(&format!("topi.{op}"), &mut args)
        }
        "exponential" => call("topi.exp", &mut args),
        "negate" => call("topi.negative", &mut args),
        "cosine" => call("topi.cos", &mut args),
        "sine" => call("topi.sin", &mut args),
        "dot_general" => dot(node, &mut args, shapes),
        "reshape" => call("topi.reshape", &mut [args[0].clone(), dims(&node.shape)?]),
        "transpose" => call(
            "topi.transpose",
            &mut [args[0].clone(), axes(&node.dimensions)?],
        ),
        "broadcast_in_dim" => {
            let mut shape = vec![1; node.shape.len()];
            for (&axis, &size) in node.dimensions.iter().zip(&shapes[node.inputs[0]]) {
                shape[axis as usize] = size;
            }
            let value = call("topi.reshape", &mut [args[0].clone(), dims(&shape)?])?;
            call("topi.broadcast_to", &mut [value, dims(&node.shape)?])
        }
        _ => Err(format!("unsupported StableHLO operation {}", node.name)),
    }
}

fn spec(shape: &[i64], dtype: &str) -> Result<Tensor> {
    Tensor::new(
        shape
            .iter()
            .map(|&n| usize::try_from(n).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>>>()?,
        dtype,
    )
}

struct Program {
    module: Option<TvmModule>,
    constants: Vec<TvmTensor>,
    aliases: Option<Vec<usize>>,
}
// CPU LLVM modules and VM state can move between threads. All VM access,
// including destruction, is exclusive; TvmExecutable serializes calls.
unsafe impl Send for Program {}

/// A compiled static float32 StableHLO function running in the native Relax VM.
/// Inputs and results are owned CPU allocations. Unsupported operations fail
/// at compile time, before any model invocation.
#[derive(Clone)]
pub struct TvmExecutable {
    program: Arc<Mutex<Program>>,
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
    queue: crate::portable::pending::Queue,
}

impl TvmExecutable {
    pub fn compile(source: &str, entry: &str, importer: &std::path::Path) -> Result<Self> {
        let graph = import(source, entry, importer)?;
        if graph.outputs.is_empty() {
            return Err("TVM function must return a tensor".into());
        }
        let aliases = graph.nodes.is_empty().then(|| graph.outputs.clone());
        let mut shapes = graph.inputs.clone();
        let mut values = graph
            .inputs
            .iter()
            .zip(&graph.input_dtypes)
            .enumerate()
            .map(|(i, (shape, dtype))| placeholder(shape, dtype, &format!("arg{i}")))
            .collect::<Result<Vec<_>>>()?;
        let mut parameters = values.clone();
        let mut parameter_shapes = graph.inputs.clone();
        let mut parameter_dtypes = graph.input_dtypes.clone();
        let mut constants = vec![];
        for (i, node) in graph.nodes.iter().enumerate() {
            let value = if node.name == "stablehlo.constant" {
                let value = placeholder(&node.shape, &node.dtype, &format!("constant{i}"))?;
                parameters.push(value.clone());
                parameter_shapes.push(node.shape.clone());
                parameter_dtypes.push(node.dtype.clone());
                let bytes = node
                    .constant
                    .iter()
                    .flat_map(|n| n.to_ne_bytes())
                    .collect::<Vec<_>>();
                constants.push(TvmTensor::put(spec(&node.shape, &node.dtype)?, &bytes)?);
                value
            } else {
                lower(node, &values, &shapes)?
            };
            values.push(value);
            shapes.push(node.shape.clone());
        }
        let output_shapes = graph
            .outputs
            .iter()
            .map(|&i| shapes[i].clone())
            .collect::<Vec<_>>();
        let mut dtypes = graph.input_dtypes.clone();
        dtypes.extend(graph.nodes.iter().map(|node| node.dtype.clone()));
        let output_dtypes = graph
            .outputs
            .iter()
            .map(|&i| dtypes[i].clone())
            .collect::<Vec<_>>();
        for &i in &graph.outputs {
            // Distinct outputs also handle returning an argument or alias
            // twice.
            parameters.push(call("topi.identity", &mut [values[i].clone()])?);
        }
        let module = if aliases.is_some() {
            None
        } else {
            let kernel = build(&parameters)?;
            Some(vm(
                &kernel,
                &parameter_shapes,
                &parameter_dtypes,
                &output_shapes,
                &output_dtypes,
            )?)
        };
        Ok(Self {
            program: Arc::new(Mutex::new(Program {
                module,
                constants,
                aliases,
            })),
            queue: crate::portable::pending::Queue::new()?,
            inputs: graph
                .inputs
                .iter()
                .zip(&graph.input_dtypes)
                .map(|(shape, dtype)| spec(shape, dtype))
                .collect::<Result<_>>()?,
            outputs: output_shapes
                .iter()
                .zip(&output_dtypes)
                .map(|(shape, dtype)| spec(shape, dtype))
                .collect::<Result<_>>()?,
        })
    }

    pub fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }

    pub fn submit(&self, inputs: Vec<TvmTensor>) -> Result<crate::portable::Pending<TvmTensor>> {
        if inputs.len() != self.inputs.len()
            || inputs.iter().zip(&self.inputs).any(|(v, s)| v.spec() != s)
        {
            return Err("TVM input tensors differ from compiled signature".into());
        }
        let model = self.clone();
        self.queue
            .submit(move || model.run(&inputs.iter().collect::<Vec<_>>()))
    }
    pub fn outputs(&self) -> &[Tensor] {
        &self.outputs
    }

    pub fn run(&self, inputs: &[&TvmTensor]) -> Result<Vec<TvmTensor>> {
        if inputs.len() != self.inputs.len()
            || inputs.iter().zip(&self.inputs).any(|(v, s)| v.spec() != s)
        {
            return Err("TVM input tensors differ from compiled signature".into());
        }
        let program = self
            .program
            .lock()
            .map_err(|_| "TVM session mutex poisoned")?;
        if let Some(aliases) = &program.aliases {
            return aliases
                .iter()
                .map(|&index| {
                    inputs
                        .get(index)
                        .map(|tensor| (*tensor).clone())
                        .ok_or_else(|| "TVM alias output index is invalid".into())
                })
                .collect();
        }
        let mut args = inputs.to_vec();
        args.extend(program.constants.iter());
        let result = program
            .module
            .as_ref()
            .ok_or("missing compiled TVM module")?
            .function("main")?
            .ok_or("missing Relax entry")?
            .run(&args)?;
        if result.len() != self.outputs.len()
            || result.iter().zip(&self.outputs).any(|(v, s)| v.spec() != s)
        {
            return Err("TVM outputs differ from compiled signature".into());
        }
        Ok(result)
    }
}
