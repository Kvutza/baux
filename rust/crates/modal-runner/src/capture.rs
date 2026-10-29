//! Optional native-module evidence attached to the existing PTX validation run.
//! The finite model fixture exercises production loaders; its trace is not a
//! proof that every full-workload branch or instruction has executed.

use crate::{ModalClient, Result, SandboxExecOptions};
use std::io::{self, Write};
use std::path::PathBuf;

pub(crate) fn destination() -> Result<Option<PathBuf>> {
    let Some(path) = std::env::var_os("BAUX_GPU_EVIDENCE") else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    std::fs::create_dir_all(&path)?;
    if path.join("modal-native.tar.gz").exists() || path.join("sandbox.txt").exists() {
        return Err("GPU evidence directory already contains a Modal run".into());
    }
    Ok(Some(path))
}

pub(crate) fn command() -> &'static str {
    r#"
mkdir -p /tmp/baux-gpu-evidence/native
nvidia-smi > /tmp/baux-gpu-evidence/device.txt
ptxas --version > /tmp/baux-gpu-evidence/ptxas.txt
nvdisasm --version > /tmp/baux-gpu-evidence/nvdisasm.txt
cupdir="$CUDA_HOME/extras/CUPTI"
if [ ! -f "$cupdir/include/cupti.h" ]; then cupdir="$CUDA_HOME/targets/x86_64-linux"; fi
cc -std=c11 -shared -fPIC -Wall -Wextra -Werror \
  -I"$cupdir/include" -I"$CUDA_HOME/include" \
  backends/cuda/native/capture.c -L"$cupdir/lib64" -L"$cupdir/lib" \
  -Wl,-rpath,"$cupdir/lib64" -Wl,-rpath,"$cupdir/lib" -lcupti \
  -o /tmp/baux-cupti.so
set +e
env BAUX_GPU_EVIDENCE=/tmp/baux-gpu-evidence/recipes \
  BAUX_CUPTI_IMAGES=/tmp/baux-gpu-evidence/native \
  LD_PRELOAD=/tmp/baux-cupti.so BAUX_SYNTH_GEMM=32 \
  /cache/target/release/baux-cuda model-check \
  2>&1 | tee /tmp/baux-gpu-evidence/model.log
model_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$model_status" > /tmp/baux-gpu-evidence/model.status
for cubin in /tmp/baux-gpu-evidence/native/*.cubin; do
  test -f "$cubin"
  sha256sum "$cubin" >> /tmp/baux-gpu-evidence/images.sha256
  nvdisasm "$cubin" > "$cubin.sass"
done
test "$model_status" -eq 0
"#
}

pub(crate) async fn pull(
    sandbox: &modal_rs::Sandbox,
    client: &mut ModalClient,
    destination: &std::path::Path,
) -> Result<()> {
    let archive = sandbox
        .exec(
            client,
            SandboxExecOptions::new(vec!["tar", "-czf", "-", "-C", "/tmp", "baux-gpu-evidence"])
                .with_timeout(120),
        )
        .await?;
    if !archive.exit_status.is_success() {
        return Err(io::Error::other("native evidence archive unavailable").into());
    }
    let bytes = archive.stdout.ok_or("native evidence archive is empty")?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination.join("modal-native.tar.gz"))?
        .write_all(&bytes)?;
    Ok(())
}
