use modal_rs::{Image, ModalClient, SandboxExecOptions, SandboxOptions, VolumeFromNameOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tempfile::NamedTempFile;

pub(crate) const CUDA_IMAGE: &str = "im-tZRy6QPZXIJyPrv4zZqPOM";
const CUDA_REV: &str = "6abfaa091e29a6275c1943895bfbc97efa306e98";
const RUST_NIGHTLY: &str = "nightly-2026-08-28";

pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(crate) fn tool_image() -> Result<Image> {
    let llvm = "/usr/lib/llvm-21";
    let cuda = "/opt/cuda-oxide";
    let path = format!(
        "/root/.cargo/bin:{llvm}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:\
         /usr/bin:/sbin:/bin"
    );
    let image = Image::from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04")
        .run_commands([
            "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl g++ gcc gnupg libc6-dev make pkg-config python3 xz-utils && rm -rf /var/lib/apt/lists/*".to_string(),
            "curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg && echo 'deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/noble/ llvm-toolchain-noble-21 main' > /etc/apt/sources.list.d/llvm-21.list && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang-21 libclang-common-21-dev libclang-21-dev lld-21 llvm-21 llvm-21-dev && rm -rf /var/lib/apt/lists/*".to_string(),
            format!(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain {RUST_NIGHTLY} && /root/.cargo/bin/rustup component add --toolchain {RUST_NIGHTLY} rust-src rustc-dev llvm-tools rust-analyzer clippy rustfmt"
            ),
            format!(
                "mkdir -p {cuda} && curl -fsSL https://github.com/NVlabs/cuda-oxide/archive/{CUDA_REV}.tar.gz | tar -xz --strip-components=1 -C {cuda} && PATH={path} LLVM_CONFIG_PATH=/usr/bin/llvm-config-21 LIBCLANG_PATH={llvm}/lib /root/.cargo/bin/cargo +{RUST_NIGHTLY} install --path {cuda}/crates/cargo-oxide --locked --force && cd {cuda} && PATH={path} LLVM_CONFIG_PATH=/usr/bin/llvm-config-21 LIBCLANG_PATH={llvm}/lib CUDA_OXIDE_LLC=/usr/bin/llc-21 CUDA_HOME=/usr/local/cuda RUSTUP_TOOLCHAIN={RUST_NIGHTLY} /root/.cargo/bin/cargo oxide setup"
            ),
        ])?
        .env([
            ("CUDA_HOME", "/usr/local/cuda"),
            ("CUDA_PATH", "/usr/local/cuda"),
            ("CUDA_TOOLKIT_PATH", "/usr/local/cuda"),
            ("CUDA_OXIDE_LLC", "/usr/bin/llc-21"),
            (
                "CUDA_OXIDE_BACKEND",
                "/opt/cuda-oxide/crates/rustc-codegen-cuda/target/x86_64-unknown-linux-gnu/debug/librustc_codegen_cuda.so",
            ),
            ("BAUX_FAISS_UNAVAILABLE", "1"),
            ("LIBCLANG_PATH", "/usr/lib/llvm-21/lib"),
            ("LLVM_CONFIG_PATH", "/usr/bin/llvm-config-21"),
            ("PATH", path.as_str()),
            ("PYTHONPATH", "/opt/baux/ops"),
            ("RUSTUP_TOOLCHAIN", RUST_NIGHTLY),
        ])?;
    Ok(image)
}

pub(crate) async fn stream_exec(
    sandbox: &modal_rs::Sandbox,
    client: &mut ModalClient,
    options: SandboxExecOptions,
) -> Result<modal_rs::SandboxExecExitStatus> {
    let mut stream = sandbox.exec_stream(client, options).await?;
    let mut stdout = stream.take_stdout().ok_or("stdout stream is missing")?;
    let mut stderr = stream.take_stderr().ok_or("stderr stream is missing")?;
    let wait = stream.take_wait().ok_or("exec wait handle is missing")?;
    let out_task = tokio::spawn(async move {
        while let Some(chunk) = stdout.recv().await {
            print!("{}", String::from_utf8_lossy(&chunk?));
        }
        Ok::<(), modal_rs::Error>(())
    });
    let err_task = tokio::spawn(async move {
        while let Some(chunk) = stderr.recv().await {
            eprint!("{}", String::from_utf8_lossy(&chunk?));
        }
        Ok::<(), modal_rs::Error>(())
    });
    let status = wait.await??;
    let (out, err) = tokio::join!(out_task, err_task);
    out??;
    err??;
    Ok(status)
}

fn repo_root() -> Result<PathBuf> {
    std::env::var_os("BUILD_WORKSPACE_DIRECTORY")
        .map(PathBuf::from)
        .map_or_else(
            || std::env::current_dir().map_err(Into::into),
            |root| Ok(root),
        )
}

pub(crate) fn source_tar() -> Result<NamedTempFile> {
    let root = repo_root()?;
    let files = Command::new("jj")
        .arg("-R")
        .arg(&root)
        .args(["file", "list", "-r", "@", "-T", "path ++ \"\\x00\""])
        .output()?;
    if !files.status.success() {
        return Err(io::Error::other(format!(
            "jj file list failed: {}",
            String::from_utf8_lossy(&files.stderr).trim()
        ))
        .into());
    }
    let present = files
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| {
            !path.is_empty() && root.join(String::from_utf8_lossy(path).as_ref()).exists()
        })
        .flat_map(|path| path.iter().copied().chain(std::iter::once(0)))
        .collect::<Vec<_>>();
    let target = NamedTempFile::new()?;
    let mut child = Command::new("tar")
        .args(["--null", "-T", "-", "-czf"])
        .arg(target.path())
        .current_dir(root)
        .env("COPYFILE_DISABLE", "1")
        .stdin(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("tar stdin is unavailable"))?
        .write_all(&present)?;
    let status = child.wait()?;
    if !status.success() {
        return Err(io::Error::other(format!("source archive failed: {status}")).into());
    }
    Ok(target)
}

pub(crate) async fn upload_source(
    sandbox: &modal_rs::Sandbox,
    client: &mut ModalClient,
    archive: &NamedTempFile,
) -> Result<()> {
    let source = std::fs::read(archive.path())?;
    let upload = sandbox
        .exec(
            client,
            SandboxExecOptions::new(vec![
                "bash",
                "-lc",
                "rm -rf /opt/baux && mkdir -p /opt/baux && tar -xzf - -C /opt/baux",
            ])
            .with_stdin(source)
            .with_timeout(120),
        )
        .await?;
    if upload.exit_status.is_success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("source upload failed: {:?}", upload.exit_status)).into())
    }
}

pub(crate) async fn cache_sandbox(
    client: &mut ModalClient,
    app: &modal_rs::App,
) -> Result<modal_rs::Sandbox> {
    let cache = client
        .volumes()
        .from_name(
            "baux-cuda-build",
            "main",
            VolumeFromNameOptions::create_if_missing(),
        )
        .await?;
    let image = match std::env::var("BAUX_MODAL_IMAGE") {
        Ok(image_id) => Image::from_id(image_id),
        Err(_) => {
            let mut image = tool_image()?;
            image.build(client, app).await?;
            image
        }
    };
    Ok(client
        .sandboxes()
        .create(
            app,
            &image,
            SandboxOptions::default()
                .with_gpu_type("T4")
                .with_milli_cpu(4_000)
                .with_memory_mb(8_192)
                .with_timeout(1_200)
                .with_volume(&cache, "/cache"),
        )
        .await?)
}

pub(crate) async fn make_sandbox(
    client: &mut ModalClient,
    app: &modal_rs::App,
) -> Result<modal_rs::Sandbox> {
    let volume = client
        .volumes()
        .from_name("baux-ckpt", "main", VolumeFromNameOptions::default())
        .await?
        .read_only();
    let cache = client
        .volumes()
        .from_name(
            "baux-cuda-build",
            "main",
            VolumeFromNameOptions::create_if_missing(),
        )
        .await?;
    let image = match std::env::var("BAUX_MODAL_IMAGE") {
        Ok(image_id) => Image::from_id(image_id),
        Err(_) => {
            let mut image = tool_image()?;
            image.build(client, app).await?;
            image
        }
    };
    let sandbox = client
        .sandboxes()
        .create(
            app,
            &image,
            SandboxOptions::default()
                .with_gpu_type("T4")
                .with_milli_cpu(4_000)
                .with_memory_mb(16_384)
                .with_timeout(3_600)
                .with_volume(&volume, "/ckpt")
                .with_volume(&cache, "/cache"),
        )
        .await?;
    Ok(sandbox)
}

pub(crate) async fn pull_file(
    sandbox: &modal_rs::Sandbox,
    client: &mut ModalClient,
    remote: &str,
    label: &str,
) -> Result<Vec<u8>> {
    let out = sandbox
        .exec(
            client,
            SandboxExecOptions::new(vec!["cat", remote]).with_timeout(60),
        )
        .await?;
    if !out.exit_status.is_success() {
        return Err(io::Error::other(format!("{label} download failed")).into());
    }
    out.stdout
        .ok_or_else(|| io::Error::other(format!("{label} was empty")).into())
}

pub(crate) async fn pull_artifacts(
    sandbox: &modal_rs::Sandbox,
    client: &mut ModalClient,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let result = pull_file(
        sandbox,
        client,
        "/tmp/baux-generation/result.json",
        "generation result",
    )
    .await?;
    let tokens = pull_file(
        sandbox,
        client,
        "/tmp/baux-generation/tokens.json",
        "generation tokens",
    )
    .await?;
    Ok((result, tokens))
}

pub(crate) async fn image_run(client: &mut ModalClient, app: &modal_rs::App) -> Result<()> {
    let mut image = tool_image()?;
    image.build(client, app).await?;
    println!("MODAL_IMAGE ok=true image={}", image.id()?);
    Ok(())
}
