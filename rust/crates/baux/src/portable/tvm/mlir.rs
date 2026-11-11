//! Owned snapshot of verified MLIR. C handles never escape the parser session.
//! ABI: IREE 3.13 compiler/mlir_interop.h and MLIR C API IR.h/BuiltinTypes.h.
use super::graph::{Graph, Node};
use crate::portable::Result;
use std::ffi::c_void;
use std::sync::Once;

#[repr(C)]
#[derive(Clone, Copy)]
struct Handle(*mut c_void);
#[repr(C)]
#[derive(Clone, Copy)]
struct Text {
    data: *const u8,
    length: isize,
}
impl Text {
    fn new(s: &str) -> Self {
        Self {
            data: s.as_ptr(),
            length: s.len() as isize,
        }
    }
    unsafe fn owned(self) -> String {
        if self.length == 0 {
            return String::new();
        }
        String::from_utf8_lossy(unsafe {
            std::slice::from_raw_parts(self.data, self.length as usize)
        })
        .into_owned()
    }
}
unsafe extern "C" {
    #[link_name = "ireeCompilerGlobalInitialize"]
    fn initialize();
    #[link_name = "ireeCompilerSessionCreate"]
    fn session() -> *mut c_void;
    #[link_name = "ireeCompilerSessionDestroy"]
    fn destroy(s: *mut c_void);
    #[link_name = "ireeCompilerSessionBorrowContext"]
    fn context(s: *mut c_void) -> Handle;
    #[link_name = "mlirModuleCreateParse"]
    fn parse(c: Handle, source: Text) -> Handle;
    #[link_name = "mlirModuleDestroy"]
    fn release(m: Handle);
    #[link_name = "mlirModuleGetOperation"]
    fn operation(m: Handle) -> Handle;
    #[link_name = "mlirModuleGetBody"]
    fn body(m: Handle) -> Handle;
    #[link_name = "mlirOperationVerify"]
    fn verify(op: Handle) -> bool;
    #[link_name = "mlirOperationGetName"]
    fn identifier(op: Handle) -> Handle;
    #[link_name = "mlirIdentifierStr"]
    fn name(id: Handle) -> Text;
    #[link_name = "mlirOperationGetAttributeByName"]
    fn attribute(op: Handle, name: Text) -> Handle;
    #[link_name = "mlirStringAttrGetValue"]
    fn string(attr: Handle) -> Text;
    #[link_name = "mlirOperationGetNumRegions"]
    fn regions(op: Handle) -> isize;
    #[link_name = "mlirOperationGetRegion"]
    fn region(op: Handle, index: isize) -> Handle;
    #[link_name = "mlirRegionGetFirstBlock"]
    fn block(r: Handle) -> Handle;
    #[link_name = "mlirBlockGetNextInRegion"]
    fn next_block(b: Handle) -> Handle;
    #[link_name = "mlirBlockGetFirstOperation"]
    fn first(b: Handle) -> Handle;
    #[link_name = "mlirOperationGetNextInBlock"]
    fn next(op: Handle) -> Handle;
    #[link_name = "mlirBlockGetNumArguments"]
    fn arguments(b: Handle) -> isize;
    #[link_name = "mlirBlockGetArgument"]
    fn argument(b: Handle, index: isize) -> Handle;
    #[link_name = "mlirOperationGetNumOperands"]
    fn operands(op: Handle) -> isize;
    #[link_name = "mlirOperationGetOperand"]
    fn operand(op: Handle, index: isize) -> Handle;
    #[link_name = "mlirOperationGetNumResults"]
    fn results(op: Handle) -> isize;
    #[link_name = "mlirOperationGetResult"]
    fn result(op: Handle, index: isize) -> Handle;
    #[link_name = "mlirValueGetType"]
    fn value_type(v: Handle) -> Handle;
    #[link_name = "mlirTypeIsARankedTensor"]
    fn ranked(t: Handle) -> bool;
    #[link_name = "mlirShapedTypeHasStaticShape"]
    fn static_shape(t: Handle) -> bool;
    #[link_name = "mlirShapedTypeGetElementType"]
    fn element(t: Handle) -> Handle;
    #[link_name = "mlirTypePrint"]
    fn print_type(t: Handle, cb: unsafe extern "C" fn(Text, *mut c_void), data: *mut c_void);
    #[link_name = "mlirShapedTypeGetRank"]
    fn rank(t: Handle) -> i64;
    #[link_name = "mlirShapedTypeGetDimSize"]
    fn dimension(t: Handle, i: isize) -> i64;
    #[link_name = "mlirAttributePrint"]
    fn print(attr: Handle, cb: unsafe extern "C" fn(Text, *mut c_void), data: *mut c_void);
    #[link_name = "mlirAttributeIsADenseI64Array"]
    fn dense_i64(attr: Handle) -> bool;
    #[link_name = "mlirDenseArrayGetNumElements"]
    fn array_length(attr: Handle) -> isize;
    #[link_name = "mlirDenseI64ArrayGetElement"]
    fn array_element(attr: Handle, i: isize) -> i64;
    #[link_name = "mlirAttributeIsADenseFPElements"]
    fn dense_float(attr: Handle) -> bool;
    #[link_name = "mlirElementsAttrGetNumElements"]
    fn elements(attr: Handle) -> i64;
    #[link_name = "mlirDenseElementsAttrGetFloatValue"]
    fn float_value(attr: Handle, i: isize) -> f32;
}

struct Session(*mut c_void);
impl Drop for Session {
    fn drop(&mut self) {
        unsafe { destroy(self.0) }
    }
}
struct Module(Handle);
impl Drop for Module {
    fn drop(&mut self) {
        unsafe { release(self.0) }
    }
}

unsafe fn signature(value: Handle) -> Result<(Vec<i64>, String)> {
    let t = unsafe { value_type(value) };
    if !unsafe { ranked(t) && static_shape(t) } {
        return Err("TVM lowering requires static ranked tensors".into());
    }
    let element = unsafe { element(t) };
    let mut dtype = String::new();
    unsafe { print_type(element, append, (&raw mut dtype).cast()) };
    let dtype = match dtype.as_str() {
        "i1" => "bool",
        "i32" => "int32",
        "i64" => "int64",
        "ui32" => "uint32",
        "ui64" => "uint64",
        "f16" => "float16",
        "bf16" => "bfloat16",
        "f32" => "float32",
        "f64" => "float64",
        _ => return Err(format!("unsupported TVM tensor dtype {dtype}")),
    };
    let shape = (0..unsafe { rank(t) })
        .map(|i| unsafe { dimension(t, i as isize) })
        .collect();
    Ok((shape, dtype.into()))
}

unsafe extern "C" fn append(text: Text, data: *mut c_void) {
    // MLIR calls this synchronously; data borrows the local String in inspect.
    unsafe { (&mut *data.cast::<String>()).push_str(&text.owned()) };
}

unsafe fn inspect(op: Handle, inputs: Vec<usize>) -> Result<Node> {
    let name = unsafe { name(identifier(op)).owned() };
    if unsafe { regions(op) != 0 || results(op) != 1 } {
        return Err(format!(
            "unsupported StableHLO operation {name}: expected one result and no regions"
        ));
    }
    let (shape, dtype) = unsafe { signature(result(op, 0)) }?;
    let mut node = Node {
        name,
        inputs,
        shape,
        dtype,
        dimensions: vec![],
        reducer: String::new(),
        dot: String::new(),
        constant: vec![],
    };
    let key = match node.name.as_str() {
        "stablehlo.transpose" => "permutation",
        "stablehlo.broadcast_in_dim" => "broadcast_dimensions",
        _ => "",
    };
    if !key.is_empty() {
        let a = unsafe { attribute(op, Text::new(key)) };
        if a.0.is_null() || !unsafe { dense_i64(a) } {
            return Err(format!("missing {key}"));
        }
        node.dimensions = (0..unsafe { array_length(a) })
            .map(|i| unsafe { array_element(a, i) })
            .collect();
    }
    if node.name == "stablehlo.dot_general" {
        if !unsafe { attribute(op, Text::new("algorithm")) }.0.is_null() {
            return Err("native TVM does not support explicit dot algorithms".into());
        }
        let a = unsafe { attribute(op, Text::new("dot_dimension_numbers")) };
        if a.0.is_null() {
            return Err("missing dot dimension numbers".into());
        }
        unsafe { print(a, append, (&raw mut node.dot).cast()) };
    }
    if node.name == "stablehlo.constant" {
        let a = unsafe { attribute(op, Text::new("value")) };
        if a.0.is_null() || !unsafe { dense_float(a) } {
            return Err("expected dense float32 constant".into());
        }
        node.constant = (0..unsafe { elements(a) })
            .map(|i| unsafe { float_value(a, i as isize) }.to_bits())
            .collect();
    }
    Ok(node)
}

unsafe fn reduction(op: Handle, inputs: &[usize]) -> Result<Vec<(Handle, Node)>> {
    let count = usize::try_from(unsafe { results(op) }).map_err(|error| error.to_string())?;
    if count == 0 || unsafe { operands(op) } != (count * 2) as isize || unsafe { regions(op) } != 1
    {
        return Err("StableHLO reduce requires N tensors, N initial values and N results".into());
    }
    let dimensions = unsafe { attribute(op, Text::new("dimensions")) };
    if dimensions.0.is_null() || !unsafe { dense_i64(dimensions) } {
        return Err("StableHLO reduce is missing dimensions".into());
    }
    let dimensions = (0..unsafe { array_length(dimensions) })
        .map(|index| unsafe { array_element(dimensions, index) })
        .collect::<Vec<_>>();
    let body = unsafe { block(region(op, 0)) };
    if body.0.is_null()
        || !unsafe { next_block(body) }.0.is_null()
        || unsafe { arguments(body) } != (count * 2) as isize
    {
        return Err("StableHLO reduce requires one region block with 2N arguments".into());
    }
    let mut reducers = Vec::with_capacity(count);
    let mut reducer_results = Vec::with_capacity(count);
    let mut current = unsafe { first(body) };
    while !current.0.is_null() {
        let current_name = unsafe { name(identifier(current)).owned() };
        if current_name == "stablehlo.return" {
            if unsafe { operands(current) } != count as isize
                || (0..count).any(|index| {
                    unsafe { operand(current, index as isize) }.0
                        != reducer_results
                            .get(index)
                            .map_or(std::ptr::null_mut(), |value: &Handle| value.0)
                })
            {
                return Err("StableHLO reduce must return each independent reducer result".into());
            }
            break;
        }
        let index = reducers.len();
        if index >= count
            || unsafe { regions(current) } != 0
            || unsafe { results(current) } != 1
            || unsafe { operands(current) } != 2
        {
            return Err("StableHLO reduce requires independent binary reducers".into());
        }
        let lhs = unsafe { argument(body, index as isize) };
        let rhs = unsafe { argument(body, (index + count) as isize) };
        let first = unsafe { operand(current, 0) };
        let second = unsafe { operand(current, 1) };
        let commutative_pair =
            (first.0 == lhs.0 && second.0 == rhs.0) || (first.0 == rhs.0 && second.0 == lhs.0);
        if !commutative_pair {
            return Err("StableHLO reduce requires independent binary reducers".into());
        }
        let reducer = current_name
            .strip_prefix("stablehlo.")
            .filter(|name| {
                matches!(
                    *name,
                    "add" | "multiply" | "minimum" | "maximum" | "and" | "or" | "xor"
                )
            })
            .ok_or("StableHLO reduce requires independent binary reducers")?;
        reducers.push(reducer.to_string());
        reducer_results.push(unsafe { result(current, 0) });
        current = unsafe { next(current) };
    }
    if reducers.len() != count || current.0.is_null() {
        return Err("StableHLO reduce region is incomplete".into());
    }
    (0..count)
        .map(|index| {
            let output = unsafe { result(op, index as isize) };
            let (shape, dtype) = unsafe { signature(output) }?;
            Ok((
                output,
                Node {
                    name: "stablehlo.reduce".into(),
                    inputs: vec![inputs[index], inputs[index + count]],
                    shape,
                    dtype,
                    dimensions: dimensions.clone(),
                    reducer: reducers[index].clone(),
                    dot: String::new(),
                    constant: vec![],
                },
            ))
        })
        .collect()
}

impl Graph {
    pub fn parse(source: &str, entry: &str) -> Result<Self> {
        static INIT: Once = Once::new();
        INIT.call_once(|| unsafe { initialize() });
        let session = Session(unsafe { session() });
        if session.0.is_null() {
            return Err("could not create MLIR session".into());
        }
        let context = unsafe { context(session.0) };
        if context.0.is_null() {
            return Err("could not activate MLIR context".into());
        }
        let module = unsafe { parse(context, Text::new(source)) };
        if module.0.is_null() {
            return Err("invalid StableHLO module".into());
        }
        let module = Module(module);
        if !unsafe { verify(operation(module.0)) } {
            return Err("StableHLO verification failed".into());
        }
        let mut op = unsafe { first(body(module.0)) };
        while !op.0.is_null() {
            if unsafe { name(identifier(op)).owned() } == "func.func" {
                let symbol = unsafe { attribute(op, Text::new("sym_name")) };
                if !symbol.0.is_null() && unsafe { string(symbol).owned() } == entry {
                    return unsafe { Self::read(op) };
                }
            }
            op = unsafe { next(op) };
        }
        Err(format!("StableHLO entry {entry:?} not found"))
    }

    unsafe fn read(function: Handle) -> Result<Self> {
        let b = unsafe { block(region(function, 0)) };
        if b.0.is_null() || !unsafe { next_block(b) }.0.is_null() {
            return Err("TVM lowering requires a single function block".into());
        }
        let mut values = std::collections::HashMap::new();
        let mut graph = Self {
            inputs: vec![],
            input_dtypes: vec![],
            nodes: vec![],
            outputs: vec![],
        };
        for i in 0..unsafe { arguments(b) } {
            let arg = unsafe { argument(b, i) };
            values.insert(arg.0 as usize, i as usize);
            let (shape, dtype) = unsafe { signature(arg) }?;
            graph.inputs.push(shape);
            graph.input_dtypes.push(dtype);
        }
        let mut op = unsafe { first(b) };
        while !op.0.is_null() {
            let inputs = (0..unsafe { operands(op) })
                .map(|i| {
                    values
                        .get(&(unsafe { operand(op, i) }.0 as usize))
                        .copied()
                        .ok_or_else(|| "unknown SSA operand".to_string())
                })
                .collect::<Result<Vec<_>>>()?;
            if unsafe { name(identifier(op)).owned() } == "func.return" {
                graph.outputs = inputs;
                return Ok(graph);
            }
            if unsafe { name(identifier(op)).owned() } == "stablehlo.reduce" {
                for (result, node) in unsafe { reduction(op, &inputs) }? {
                    values.insert(result.0 as usize, graph.inputs.len() + graph.nodes.len());
                    graph.nodes.push(node);
                }
                op = unsafe { next(op) };
                continue;
            }
            let node = unsafe { inspect(op, inputs) }?;
            values.insert(
                unsafe { result(op, 0) }.0 as usize,
                graph.inputs.len() + graph.nodes.len(),
            );
            graph.nodes.push(node);
            op = unsafe { next(op) };
        }
        Err("StableHLO function has no return".into())
    }
}
