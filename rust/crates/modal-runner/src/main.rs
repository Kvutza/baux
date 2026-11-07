use modal_rs::{AppOptions, Image, ModalClient, SandboxExecOptions, SandboxOptions};
use std::io;
use std::path::{Path, PathBuf};

mod mhc4;
mod sandbox;

use sandbox::{
    CUDA_IMAGE, Result, cache_sandbox, image_run, make_sandbox, pull_artifacts, pull_file,
    source_tar, stream_exec, upload_source,
};

enum Action {
    Bo {
        output: PathBuf,
        candidates: u32,
        history: u32,
        iterations: u32,
    },
    Image,
    Generate {
        output: PathBuf,
        context: u32,
        prompt: u32,
        visits: u32,
        temperature: f32,
        seed: u64,
        unroll: u32,
        chunk: u32,
    },
    Mhc4(mhc4::Config),
    Wheel {
        output: PathBuf,
    },
}

fn parse_bo<I: Iterator<Item = std::ffi::OsString>>(mut args: I) -> Result<Action> {
    let mut output = PathBuf::from("results/cuda/bf16-bo-curve-t4.json");
    let mut candidates = 4_u32;
    let mut history = 32_u32;
    let mut iterations = 50_u32;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| {
            io::Error::other(format!("missing value for {}", flag.to_string_lossy()))
        })?;
        match flag.to_str() {
            Some("--output") => output = value.into(),
            Some("--candidates") => candidates = value.to_string_lossy().parse()?,
            Some("--history") => history = value.to_string_lossy().parse()?,
            Some("--iterations") => iterations = value.to_string_lossy().parse()?,
            _ => {
                return Err(io::Error::other(
                    "usage: baux-modal bo --output PATH --candidates N --history N --iterations N",
                )
                .into());
            }
        }
    }
    if candidates == 0 || history == 0 || iterations == 0 {
        return Err(io::Error::other("BO probe dimensions must be positive").into());
    }
    Ok(Action::Bo {
        output,
        candidates,
        history,
        iterations,
    })
}

fn parse_generate<I: Iterator<Item = std::ffi::OsString>>(mut args: I) -> Result<Action> {
    let mut output = PathBuf::from("results/cuda/generation-t4");
    let mut context = 4_096_u32;
    let mut prompt = 128_u32;
    let mut visits = 2_u32;
    let mut temperature = 0.8_f32;
    let mut seed = 17_u64;
    let mut unroll = 2_u32;
    let mut chunk = 0_u32;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| {
            io::Error::other(format!("missing value for {}", flag.to_string_lossy()))
        })?;
        match flag.to_str() {
            Some("--output") => output = value.into(),
            Some("--context") => context = value.to_string_lossy().parse()?,
            Some("--prompt") => prompt = value.to_string_lossy().parse()?,
            Some("--visits") => visits = value.to_string_lossy().parse()?,
            Some("--temperature") => temperature = value.to_string_lossy().parse()?,
            Some("--seed") => seed = value.to_string_lossy().parse()?,
            Some("--unroll") => unroll = value.to_string_lossy().parse()?,
            Some("--chunk") => chunk = value.to_string_lossy().parse()?,
            _ => {
                return Err(io::Error::other(
                    "usage: baux-modal generate --output DIR --context N --prompt N --visits N \
                     --temperature F --seed N --unroll N",
                )
                .into());
            }
        }
    }
    if context == 0
        || prompt == 0
        || prompt >= context
        || visits == 0
        || !temperature.is_finite()
        || temperature < 0.0
        || unroll == 0
        || unroll > 32
        || (chunk != 0
            && (!(64..=4096).contains(&chunk) || !chunk.is_power_of_two() || context % chunk != 0))
    {
        return Err(
            io::Error::other("invalid generation dimensions or sampling parameters").into(),
        );
    }
    Ok(Action::Generate {
        output,
        context,
        prompt,
        visits,
        temperature,
        seed,
        unroll,
        chunk,
    })
}

fn parse_wheel<I: Iterator<Item = std::ffi::OsString>>(args: I) -> Result<Action> {
    let mut output = PathBuf::from("/tmp/baux-cuda-wheel.whl");
    let mut args = args;
    if let Some(path) = args.next() {
        output = path.into();
    }
    if args.next().is_some() {
        return Err(io::Error::other("usage: baux-modal wheel [OUTPUT]").into());
    }
    Ok(Action::Wheel { output })
}

fn parse_args() -> Result<Action> {
    let mut args = std::env::args_os().skip(1);
    let action = args
        .next()
        .ok_or_else(|| io::Error::other("usage: baux-modal <bo | generate | image | wheel>"))?;
    match action.to_str() {
        Some("bo") => parse_bo(args),
        Some("image") if args.next().is_none() => Ok(Action::Image),
        Some("generate") => parse_generate(args),
        Some("mhc4") => Ok(Action::Mhc4(mhc4::parse(args)?)),
        Some("wheel") => parse_wheel(args),
        _ => Err(io::Error::other("usage: baux-modal <bo | generate | image | wheel>").into()),
    }
}

fn wheel_cmd() -> String {
    let wheel = wheel_path();
    format!(
        "set -euo pipefail; cd /opt/baux; rm -rf /tmp/baux-wheel /tmp/baux-wheel-env; mkdir -p \
         /tmp/baux-wheel; python -m venv /tmp/baux-wheel-env; /tmp/baux-wheel-env/bin/python -m \
         pip install --quiet --upgrade 'pip>=26.2.2' 'certifi>=2026.10.19'; \
         /tmp/baux-wheel-env/bin/python -m pip install --quiet 'jax[cuda12]'; export \
         PATH=/tmp/baux-wheel-env/bin:$PATH; export XLA_PYTHON_CLIENT_PREALLOCATE=false; unset \
         LD_LIBRARY_PATH; PARITY=$(./admin/tools/buck2w --isolation-dir cuda build //:cuda-parity \
         --target-platforms //:linux-x86_64-platform --local-only --num-threads 4 \
         --show-full-simple-output); cat \"$PARITY\"; ./admin/tools/buck2w --isolation-dir cuda build \
         //:cuda-wheel --target-platforms //:linux-x86_64-platform --local-only --num-threads 4 \
         --out {wheel}; /tmp/baux-wheel-env/bin/python -m pip install --quiet {wheel}; \
         /tmp/baux-wheel-env/bin/python admin/ops/cusprbench.py; /tmp/baux-wheel-env/bin/python \
         admin/ops/bf16_bench.py",
    )
}

fn wheel_path() -> String {
    format!(
        "/tmp/baux-wheel/baux-{}+cuda75-cp314-cp314-manylinux_2_28_x86_64.whl",
        env!("CARGO_PKG_VERSION")
    )
}

fn bo_cmd(candidates: u32, history: u32, iterations: u32) -> String {
    format!(
        "set -euo pipefail; cd /opt/baux; CARGO_HOME=/cache/cargo CARGO_TARGET_DIR=/cache/target \
         ./admin/tools/cuda-run build; /cache/target/release/baux-cuda parity; \
         /cache/target/release/baux-cuda bf16-curve \
         {candidates} {history} {iterations} > /tmp/baux-bo-result.json; cat \
         /tmp/baux-bo-result.json"
    )
}

fn generation_cmd(
    context: u32,
    prompt: u32,
    visits: u32,
    temperature: f32,
    seed: u64,
    unroll: u32,
    chunk: u32,
    prompt_file: bool,
) -> String {
    let mut env = String::new();
    if std::env::var_os("BAUX_CUDA_PROFILE").is_some() {
        env.push_str("export BAUX_CUDA_PROFILE=1; ");
    }
    if let Ok(rows) = std::env::var("BAUX_CUDA_REPAIR_CHUNK") {
        if let Ok(rows) = rows.parse::<u32>() {
            env.push_str(&format!("export BAUX_CUDA_REPAIR_CHUNK={rows}; "));
        }
    }
    let prompt_arg = if prompt_file {
        " /tmp/baux-prompt.json"
    } else {
        ""
    };
    format!(
        "set -euo pipefail; cd /opt/baux; {env}CARGO_HOME=/cache/cargo \
         CARGO_TARGET_DIR=/cache/target ./admin/tools/cuda-run build; test -f \
         /ckpt/looped-571e8175/checkpoint.safetensors; /cache/target/release/baux-cuda \
         model-check; /cache/target/release/baux-cuda diffusion-check; \
         /cache/target/release/baux-cuda context 4096 128 1; rm -rf /tmp/baux-generation; \
         /cache/target/release/baux-cuda generation-bench \
         /ckpt/looped-571e8175/checkpoint.safetensors /tmp/baux-generation {context} {prompt} \
         {visits} {temperature} {seed} {unroll} {chunk}{prompt_arg}"
    )
}

async fn bo_run(
    client: &mut ModalClient,
    app: &modal_rs::App,
    output: &Path,
    candidates: u32,
    history: u32,
    iterations: u32,
) -> Result<()> {
    let archive = source_tar()?;
    let sandbox = cache_sandbox(client, app).await?;
    let run: Result<Vec<u8>> = async {
        upload_source(&sandbox, client, &archive).await?;
        let status = stream_exec(
            &sandbox,
            client,
            SandboxExecOptions::new(vec![
                "bash",
                "-lc",
                &bo_cmd(candidates, history, iterations),
            ])
            .with_timeout(1_100),
        )
        .await?;
        if !status.is_success() {
            return Err(io::Error::other(format!("CUDA BO benchmark failed: {status:?}")).into());
        }
        pull_file(
            &sandbox,
            client,
            "/tmp/baux-bo-result.json",
            "CUDA BO artifact",
        )
        .await
    }
    .await;
    let stopped = sandbox.terminate(client).await;
    let artifact = run?;
    stopped?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output, artifact)?;
    println!("MODAL_BO ok=true output={}", output.display());
    Ok(())
}

fn env_config(chunk: u32, output: &Path) -> Result<Option<Vec<u8>>> {
    let prompt_data = std::env::var_os("BAUX_CUDA_PROMPT")
        .map(std::fs::read)
        .transpose()?;
    if let Ok(value) = std::env::var("BAUX_CUDA_REPAIR_CHUNK") {
        let rows = value.parse::<u32>()?;
        if !(64..=4096).contains(&rows) || !rows.is_power_of_two() || chunk == 0 {
            return Err(
                "BAUX_CUDA_REPAIR_CHUNK requires streamed execution and a power of two from 64 \
                 through 4096"
                    .into(),
            );
        }
    }
    if output.exists() {
        return Err("generation output already exists; choose a fresh result directory".into());
    }
    Ok(prompt_data)
}

#[allow(clippy::too_many_arguments)]
async fn generation_run(
    client: &mut ModalClient,
    app: &modal_rs::App,
    output: &Path,
    context: u32,
    prompt: u32,
    visits: u32,
    temperature: f32,
    seed: u64,
    unroll: u32,
    chunk: u32,
) -> Result<()> {
    let prompt_data = env_config(chunk, output)?;
    let archive = source_tar()?;
    let sandbox = make_sandbox(client, app).await?;
    let run: Result<(Vec<u8>, Vec<u8>)> = async {
        upload_source(&sandbox, client, &archive).await?;
        if let Some(data) = &prompt_data {
            let upload = sandbox
                .exec(
                    client,
                    SandboxExecOptions::new(vec!["bash", "-lc", "cat > /tmp/baux-prompt.json"])
                        .with_stdin(data.clone())
                        .with_timeout(30),
                )
                .await?;
            if !upload.exit_status.is_success() {
                return Err("prompt upload failed".into());
            }
        }
        let status = stream_exec(
            &sandbox,
            client,
            SandboxExecOptions::new(vec![
                "bash",
                "-lc",
                &generation_cmd(
                    context,
                    prompt,
                    visits,
                    temperature,
                    seed,
                    unroll,
                    chunk,
                    prompt_data.is_some(),
                ),
            ])
            .with_timeout(3_500),
        )
        .await?;
        if !status.is_success() {
            return Err(io::Error::other(format!("CUDA generation failed: {status:?}")).into());
        }
        pull_artifacts(&sandbox, client).await
    }
    .await;
    let stopped = sandbox.terminate(client).await;
    let (result, tokens) = run?;
    stopped?;
    std::fs::create_dir_all(output)?;
    std::fs::write(output.join("result.json"), &result)?;
    std::fs::write(output.join("tokens.json"), tokens)?;
    print!("{}", String::from_utf8_lossy(&result));
    println!("GENERATION_T4 ok=true output={}", output.display());
    Ok(())
}

async fn wheel_run(client: &mut ModalClient, app: &modal_rs::App, output: &Path) -> Result<()> {
    let archive = source_tar()?;
    let image_id = std::env::var("BAUX_MODAL_IMAGE").unwrap_or_else(|_| CUDA_IMAGE.to_string());
    let image = Image::from_id(image_id);
    let sandbox = client
        .sandboxes()
        .create(
            app,
            &image,
            SandboxOptions::default()
                .with_gpu_type("T4")
                .with_milli_cpu(8_000)
                .with_memory_mb(16_384)
                .with_timeout(3_600),
        )
        .await?;
    let run: Result<()> = async {
        upload_source(&sandbox, client, &archive).await?;
        let status = stream_exec(
            &sandbox,
            client,
            SandboxExecOptions::new(vec!["bash", "-lc", &wheel_cmd()]).with_timeout(3_600),
        )
        .await?;
        if !status.is_success() {
            return Err(io::Error::other(format!("wheel gate failed: {status:?}")).into());
        }
        let wheel = pull_file(&sandbox, client, &wheel_path(), "wheel").await?;
        std::fs::write(output, wheel)?;
        Ok(())
    }
    .await;
    let stopped = sandbox.terminate(client).await;
    run?;
    stopped?;
    println!("MODAL_WHEEL ok=true output={}", output.display());
    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("baux-modal: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let _output = modal_rs::enable_output();
    let action = parse_args()?;
    let mut client = ModalClient::connect().await?;
    let app = client
        .get_or_create_app("baux-rust-modal", "main", AppOptions::create_if_missing())
        .await?;
    match action {
        Action::Bo {
            output,
            candidates,
            history,
            iterations,
        } => bo_run(&mut client, &app, &output, candidates, history, iterations).await,
        Action::Image => image_run(&mut client, &app).await,
        Action::Generate {
            output,
            context,
            prompt,
            visits,
            temperature,
            seed,
            unroll,
            chunk,
        } => {
            generation_run(
                &mut client,
                &app,
                &output,
                context,
                prompt,
                visits,
                temperature,
                seed,
                unroll,
                chunk,
            )
            .await
        }
        Action::Mhc4(config) => mhc4::run(&mut client, &app, config).await,
        Action::Wheel { output } => wheel_run(&mut client, &app, &output).await,
    }
}
