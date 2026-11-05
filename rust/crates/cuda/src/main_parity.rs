use std::error::Error;
use std::io;

use baux_cuda::{embed_parity, feedback_parity, target_name};
use cuda_core::CudaContext;

pub(crate) type AppResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

pub(super) fn parity() -> AppResult<()> {
    baux_cuda::proposal_check().map_err(io::Error::other)?;
    baux_cuda::neighbors::parity().map_err(io::Error::other)?;
    embed_parity().map_err(io::Error::other)?;
    feedback_parity().map_err(io::Error::other)?;
    let context = CudaContext::new(0)?;
    let target = target_name(&context).map_err(io::Error::other)?;
    println!("PARITY ok=true checks=proposal,neighbors,embedding,feedback target={target}");
    Ok(())
}
