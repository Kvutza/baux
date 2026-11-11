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

fn lower(node: &Node, values: &[Value], shapes: &[Vec<i64>], dtypes: &[String]) -> Result<Value> {
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
        _ => None,
    };
    if let Some(name) = binary {
        if node.dtype == "bool" && matches!(op, "add" | "multiply") {
            return call(
                if op == "add" {
                    "topi.logical_or"
                } else {
                    "topi.logical_and"
                },
                &mut args,
            );
        }
        return call(&format!("topi.{name}"), &mut args);
    }
    match op {
        "round_nearest_afz" => super::element::compute(&node.shape, op, |indices| {
            super::element::round(super::element::load(&args[0], indices)?, &node.dtype)
        }),
        "floor" | "ceil" | "round_nearest_even" | "is_finite" => {
            let intrinsic = match op {
                "floor" => "tirx.floor",
                "ceil" => "prim.ceil",
                "round_nearest_even" => "tirx.nearbyint",
                _ => "tirx.isfinite",
            };
            super::element::compute(&node.shape, op, |indices| {
                call(
                    intrinsic,
                    &mut [super::element::load(&args[0], indices)?, Value::none()],
                )
            })
        }
        "maximum" | "minimum" | "and" | "or" | "xor" => {
            super::element::compute(&node.shape, op, |indices| {
                super::compiler::combine(
                    &node.dtype,
                    op,
                    super::element::load(&args[0], indices)?,
                    super::element::load(&args[1], indices)?,
                )
            })
        }
        "compare" => super::comparison::lower(node, &args, &dtypes[node.inputs[0]]),
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
        "dot_general" => super::contraction::lower(node, &args, shapes),
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
        _ => super::transform::lower(node, &args, shapes),
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
    registration: Option<super::callback::Registration>,
    constants: Vec<TvmTensor>,
    aliases: Option<Vec<usize>>,
}
// CPU LLVM modules and VM state can move between threads. All VM access,
// including destruction, is exclusive; TvmExecutable serializes calls.
unsafe impl Send for Program {}

/// A native Relax VM executable. StableHLO compilation produces CPU kernels;
/// native callbacks can also consume retained Metal tensors. Input devices are
/// checked before invoking either kind of kernel.
#[derive(Clone)]
pub struct TvmExecutable {
    program: Arc<Mutex<Program>>,
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
    queue: crate::portable::pending::Queue,
    device: i32,
}

impl TvmExecutable {
    pub fn compile(source: &str, entry: &str, importer: &std::path::Path) -> Result<Self> {
        let graph = import(source, entry, importer)?;
        if graph.outputs.is_empty() {
            return Err("TVM function must return a tensor".into());
        }
        let aliases = graph.nodes.is_empty().then(|| graph.outputs.clone());
        let mut shapes = graph.inputs.clone();
        let mut dtypes = graph.input_dtypes.clone();
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
                constants.push(TvmTensor::put(
                    spec(&node.shape, &node.dtype)?,
                    &node.constant,
                )?);
                value
            } else {
                lower(node, &values, &shapes, &dtypes)?
            };
            values.push(value);
            shapes.push(node.shape.clone());
            dtypes.push(node.dtype.clone());
        }
        let output_shapes = graph
            .outputs
            .iter()
            .map(|&i| shapes[i].clone())
            .collect::<Vec<_>>();
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
                registration: None,
                constants,
                aliases,
            })),
            device: 1,
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

    /// Compile a Relax call to an owned native tensor kernel. The callback
    /// must complete all writes before returning immutable resident results.
    /// This integrates a native kernel; it does not translate that kernel.
    pub fn native(
        inputs: Vec<Tensor>,
        outputs: Vec<Tensor>,
        device: &str,
        kernel: impl Fn(&[TvmTensor]) -> Result<Vec<TvmTensor>> + Send + Sync + 'static,
    ) -> Result<Self> {
        for spec in inputs.iter().chain(&outputs) {
            crate::portable::layout(spec)?;
        }
        let device = match device {
            "cpu" => 1,
            "metal" => {
                super::device::metal()?;
                8
            }
            _ => return Err(format!("unsupported TVM native device {device}")),
        };
        let registration =
            super::callback::Registration::new(inputs.clone(), outputs.clone(), kernel)?;
        let module =
            super::compiler::external(&registration.name, &inputs, &outputs, i64::from(device))?;
        Ok(Self {
            program: Arc::new(Mutex::new(Program {
                module: Some(module),
                registration: Some(registration),
                constants: vec![],
                aliases: None,
            })),
            device,
            inputs,
            outputs,
            queue: crate::portable::pending::Queue::new()?,
        })
    }

    pub fn submit(&self, inputs: Vec<TvmTensor>) -> Result<crate::portable::Pending<TvmTensor>> {
        if inputs.len() != self.inputs.len()
            || inputs
                .iter()
                .zip(&self.inputs)
                .any(|(v, s)| v.spec() != s || v.device != self.device)
        {
            return Err("TVM input tensors differ from compiled signature or device".into());
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
            || inputs
                .iter()
                .zip(&self.inputs)
                .any(|(v, s)| v.spec() != s || v.device != self.device)
        {
            return Err("TVM input tensors differ from compiled signature or device".into());
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
