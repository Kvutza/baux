use baux::artifact::json::{self, Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::Command;

const CORRECTIONS: &[&str] = &[
    "fit.rs (compute_loglik & loo_neighbors): local evaluates -0.5*ln(2*pi) - ln(se) - 0.5*((y - \
     mu)/se)^2 without squaring se (avoiding binary64 se*se underflow below 1e-154) and excludes \
     held-out LOO rows by exact row ID (id != held) rather than distance threshold 1e-15",
    "knn (ExactBackend vs FaissBackend and smnsrch): local in-memory ExactBackend computes exact \
     binary64 squared-L2 distances (matching baux::semantics::neighbor and upstream \
     small_n_search f64 bit-for-bit), while local BpAnnDisk smnsrch uses binary32 resident flat \
     cache (matching upstream FaissBackend f32 bit-for-bit)",
    "acquisition.rs (UCBAcquisition & ParetoAcquisition::nondomin_2d): local uses a deterministic \
     streaming min-heap without advancing RNG in UCB and an O(n log n) 2D sweep in Pareto \
     non-dominated sorting",
];

pub fn stage_deps(
    dep_files: &[String],
    artifacts: &BTreeMap<String, PathBuf>,
    extern_names: &BTreeMap<String, String>,
    dest_dir: &Path,
) -> Result<(), String> {
    if dest_dir.exists() {
        fs::remove_dir_all(dest_dir).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
    for file in dep_files {
        let src = Path::new(file);
        if let Some(name) = src.file_name() {
            let dst = dest_dir.join(name);
            if !dst.exists() {
                symlink(src, dst).map_err(|e| e.to_string())?;
            }
        }
    }
    for (name, rlib_path) in artifacts {
        let staged = extern_names.get(name).cloned().unwrap_or_else(|| {
            rlib_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_owned()
        });
        let dst = dest_dir.join(staged);
        if !dst.exists() {
            symlink(rlib_path, dst).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn extern_args(
    deps_dir: &Path,
    artifacts: &BTreeMap<String, PathBuf>,
    extern_names: &BTreeMap<String, String>,
    crate_name: &str,
) -> Vec<String> {
    if let Some(ext) = extern_names.get(crate_name) {
        let rlib = deps_dir.join(ext);
        let rmeta = rlib.with_extension("rmeta");
        if rmeta.exists() {
            return vec![
                "--extern".into(),
                format!("{crate_name}={}", rmeta.display()),
                "--extern".into(),
                format!("{crate_name}={}", rlib.display()),
            ];
        }
        return vec![
            "--extern".into(),
            format!("{crate_name}={}", rlib.display()),
        ];
    }
    let path = &artifacts[crate_name];
    vec![
        "--extern".into(),
        format!("{crate_name}={}", path.display()),
    ]
}

pub fn build_rustc(
    root: &Path,
    src: &Path,
    bin: &Path,
    deps_dir: &Path,
    artifacts: &BTreeMap<String, PathBuf>,
    extern_names: &BTreeMap<String, String>,
    crates: &[&str],
    extra: &[String],
    link_search: &[String],
    link_libs: &[String],
) -> Result<(), String> {
    let mut cmd = Command::new("rustc");
    cmd.current_dir(root)
        .args(["--edition=2024", &src.to_string_lossy(), "-L"])
        .arg(format!("dependency={}", deps_dir.display()));
    for crate_name in crates {
        cmd.args(extern_args(deps_dir, artifacts, extern_names, crate_name));
    }
    cmd.args(extra).args(["-o", &bin.to_string_lossy()]);
    for ls in link_search {
        cmd.args(["-L", ls]);
    }
    for ll in link_libs {
        cmd.args(["-l", ll]);
    }
    if cfg!(target_os = "macos") {
        cmd.args(["-l", "framework=Accelerate"]);
        if Path::new("/opt/homebrew/opt/libomp/lib").is_dir() {
            cmd.args(["-L", "native=/opt/homebrew/opt/libomp/lib", "-l", "omp"]);
        }
    }
    let out = cmd.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "rustc failed for {}:\n{}",
            src.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn join_csv(up_json: &Value, key: &str) -> String {
    let Some(seq) = up_json["primitives"][key].as_seq() else {
        return String::new();
    };
    seq.iter()
        .filter_map(|v| v.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

pub fn run_binaries(
    root: &Path,
    verify_dir: &Path,
    up_bin: &Path,
    loc_bin: &Path,
    numerical_so: &Path,
) -> Result<(Value, Value), String> {
    let run_dir = verify_dir.join(format!("run-{}", std::process::id()));
    if run_dir.exists() {
        let _ = fs::remove_dir_all(&run_dir);
    }
    fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;
    let up_out = Command::new(up_bin)
        .current_dir(root)
        .arg(run_dir.join("upstream"))
        .output()
        .map_err(|e| e.to_string())?;
    if !up_out.status.success() {
        let _ = fs::remove_dir_all(&run_dir);
        return Err(format!(
            "upstream harness failed: {}",
            String::from_utf8_lossy(&up_out.stderr)
        ));
    }
    let up_text = String::from_utf8_lossy(&up_out.stdout);
    let up_json: Value = json::from_str(up_text.trim()).map_err(|e| e.to_string())?;
    let loc_out = Command::new(loc_bin)
        .current_dir(root)
        .env("BAUX_NATIVE_TEST", "1")
        .arg(run_dir.join("local"))
        .arg(numerical_so)
        .arg(join_csv(&up_json, "pcg_u64"))
        .arg(join_csv(&up_json, "pcg_integers"))
        .output()
        .map_err(|e| e.to_string())?;
    let _ = fs::remove_dir_all(&run_dir);
    if !loc_out.status.success() {
        return Err(format!(
            "local harness failed: {}",
            String::from_utf8_lossy(&loc_out.stderr)
        ));
    }
    let loc_text = String::from_utf8_lossy(&loc_out.stdout);
    let loc_json: Value = json::from_str(loc_text.trim()).map_err(|e| e.to_string())?;
    Ok((up_json, loc_json))
}

fn slice_eq(left: &Value, right: &Value, key: &str, count: usize) -> bool {
    let Some(l_seq) = left[key].as_seq() else {
        return false;
    };
    let Some(r_seq) = right[key].as_seq() else {
        return false;
    };
    l_seq.len() >= count && r_seq.len() >= count && l_seq[..count] == r_seq[..count]
}

pub fn verify_outputs(up_json: &Value, loc_json: &Value) -> Result<Value, String> {
    if up_json["primitives"] != loc_json["primitives"] {
        return Err("upstream and local primitive outputs differ".into());
    }
    let up_tr = &up_json["trajectories"];
    let loc_tr = &loc_json["trajectories"];
    for exact in ["turbo_zero", "lhd_only"] {
        if up_tr[exact] != loc_tr[exact] {
            return Err(format!("bit-exact mismatch in trajectories.{exact}"));
        }
    }
    let (ut, lt) = (&up_tr["turbo_enn"], &loc_tr["turbo_enn"]);
    if !slice_eq(ut, lt, "ask_bits", 18)
        || !slice_eq(ut, lt, "posterior_mu_bits", 4)
        || !slice_eq(ut, lt, "incumbent_x_bits", 9)
        || ut["tr_length_bits"] != lt["tr_length_bits"]
        || ut["obs_counts"] != lt["obs_counts"]
        || ut["restart_generations"] != lt["restart_generations"]
    {
        return Err("bit-exact invariant mismatch in trajectories.turbo_enn".into());
    }
    let (um, lm) = (&up_tr["morbo_trust_region"], &loc_tr["morbo_trust_region"]);
    if um["sobol_seed_base"] != lm["sobol_seed_base"]
        || um["obs_counts"] != lm["obs_counts"]
        || !slice_eq(um, lm, "ask_bits", 12)
        || !slice_eq(um, lm, "posterior_mu_bits", 8)
        || !slice_eq(um, lm, "incumbent_x_bits", 6)
        || !slice_eq(um, lm, "tr_length_bits", 2)
    {
        return Err("bit-exact invariant mismatch in trajectories.morbo_trust_region".into());
    }
    let (ub, lb) = (&up_tr["disk_bpann"], &loc_tr["disk_bpann"]);
    if ub["train_rows_x_bits"] != lb["train_rows_x_bits"]
        || ub["train_rows_y_bits"] != lb["train_rows_y_bits"]
        || ub["disk_dist_bits"] != lb["mem_dist_bits"]
        || ub["disk_indices"] != lb["mem_indices"]
        || ub["mem_dist_bits"] != lb["disk_dist_bits"]
        || ub["mem_indices"] != lb["disk_indices"]
        || !slice_eq(ub, lb, "opt_ask_bits", 18)
        || !slice_eq(ub, lb, "opt_mu_bits", 6)
        || !slice_eq(ub, lb, "opt_incumbent_bits", 12)
    {
        return Err("bit-exact invariant mismatch in trajectories.disk_bpann".into());
    }
    Ok(json!({
        "primitives": up_json["primitives"].clone(),
        "shared_exact_trajectories": {
            "turbo_zero": up_tr["turbo_zero"].clone(),
            "lhd_only": up_tr["lhd_only"].clone(),
        },
        "enn_trajectories": {
            "turbo_enn": {"upstream": ut.clone(), "corrected_local": lt.clone()},
            "morbo_trust_region": {"upstream": um.clone(), "corrected_local": lm.clone()},
            "disk_bpann": {"upstream": ub.clone(), "corrected_local": lb.clone()},
        },
        "documented_local_corrections": CORRECTIONS,
    }))
}
