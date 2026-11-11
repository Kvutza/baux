//! Process isolation prevents incompatible LLVM versions in IREE and TVM
//! from coalescing weak C++ symbols on macOS. Parsing is still native C FFI.
mod graph;
mod mlir;
mod portable {
    pub type Result<T> = std::result::Result<T, String>;
}
use std::io::Read;

fn run() -> portable::Result<()> {
    let entry = std::env::args().nth(1).ok_or("missing entry symbol")?;
    let mut source = String::new();
    std::io::stdin()
        .read_to_string(&mut source)
        .map_err(|e| e.to_string())?;
    let graph = graph::Graph::parse(&source, &entry)?;
    println!(
        "{}",
        deser_json::to_string(&graph).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
