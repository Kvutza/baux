use std::io;

use super::main_parity::AppResult;

pub(super) fn benchmark(args: &[String]) -> AppResult<()> {
    match args {
        [kind] if kind == "muon" => {
            let record = baux_cuda::muon::measure().map_err(io::Error::other)?;
            println!("{}", baux_cuda::artifact::json::pretty_string(&record)?);
            Ok(())
        }
        _ => Err(io::Error::other(
            "usage: bench muon; the packed-quantized materialization benchmark was retired",
        )
        .into()),
    }
}
