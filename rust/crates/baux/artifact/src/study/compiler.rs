//! Sum-of-Products study compiler and inheritance resolution.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use deser_value::{Map, Value};

use super::overlay::{
    clean_variants, deep_merge, normalize_paths, parse_value, prune_deletes, set_path, split_commas,
};
use super::{CompileOpts, CompiledStudy, StudyCell, StudyPrune};

struct ArmDef {
    name: String,
    baseline: bool,
    set: Value,
    sweep: Vec<(String, Vec<Value>)>,
}

struct RawStudySpec {
    rungs: Vec<u32>,
    prune: StudyPrune,
    arms: Vec<ArmDef>,
    had_study: bool,
}

#[derive(Clone)]
struct IntermediateCell {
    env_id: String,
    arm_id: String,
    baseline: bool,
    factors: Vec<(String, String)>,
    leaf: Value,
}

/// Compile a study TOML file from disk.
pub fn compile_file(path: &Path, opts: &CompileOpts<'_>) -> Result<CompiledStudy, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read file '{}': {e}", path.display()))?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut seen = Vec::new();
    let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    seen.push(canon);
    compile_internal(&text, base_dir, opts, &mut seen)
}

/// Compile a study TOML string relative to a base directory.
pub fn compile_str(
    text: &str,
    base_dir: &Path,
    opts: &CompileOpts<'_>,
) -> Result<CompiledStudy, String> {
    let mut seen = Vec::new();
    compile_internal(text, base_dir, opts, &mut seen)
}

fn compile_internal(
    text: &str,
    base_dir: &Path,
    opts: &CompileOpts<'_>,
    seen: &mut Vec<PathBuf>,
) -> Result<CompiledStudy, String> {
    let root: Value = crate::artifact::toml::from_str(text)
        .map_err(|e| format!("failed to parse study TOML: {e}"))?;
    let bases = load_bases(&root, base_dir, seen)?;

    let mut top_overrides = root.clone();
    if let Some(m) = top_overrides.as_map_mut() {
        m.remove("extends");
        m.remove("study");
    }
    let top_overrides = normalize_paths(top_overrides);

    let study_spec = extract_spec(&root)?;
    let mut intermediate_cells = Vec::new();

    for base in &bases {
        let mut merged_base = base.clone();
        deep_merge(&mut merged_base, &top_overrides);
        clean_variants(&mut merged_base, &top_overrides);
        prune_deletes(&mut merged_base);

        let env_id = compute_env(&merged_base);
        for arm in &study_spec.arms {
            expand_arm(&merged_base, &env_id, arm, &mut intermediate_cells);
        }
    }

    for cli_arg in opts.cli_sets {
        intermediate_cells = apply_arg(intermediate_cells, cli_arg)?;
    }

    apply_scalars(&mut intermediate_cells, opts);
    resolve_baselines(&mut intermediate_cells);

    let prune = opts.prune.unwrap_or(study_spec.prune);
    let cells = finalize_cells(intermediate_cells)?;
    let is_multi = cells.len() > 1 || study_spec.had_study;

    Ok(CompiledStudy {
        is_multi,
        rungs: study_spec.rungs,
        prune,
        cells,
    })
}

fn load_bases(
    root: &Value,
    base_dir: &Path,
    seen: &mut Vec<PathBuf>,
) -> Result<Vec<Value>, String> {
    let extends_val = match root.as_map().and_then(|m| m.get("extends")) {
        Some(v) if !v.is_null() => v,
        _ => return Ok(vec![Value::from(Map::new())]),
    };

    let mut base_paths = Vec::new();
    if let Some(s) = extends_val.as_str() {
        base_paths.push(s.to_string());
    } else if let Some(seq) = extends_val.as_seq() {
        for item in seq.iter() {
            if let Some(s) = item.as_str() {
                base_paths.push(s.to_string());
            }
        }
    }

    let mut result = Vec::new();
    for rel_path in base_paths {
        let full_path = base_dir.join(&rel_path);
        let canon = full_path
            .canonicalize()
            .unwrap_or_else(|_| full_path.clone());
        if seen.contains(&canon) {
            return Err(format!(
                "inheritance cycle detected: '{}' is already in ancestor chain",
                canon.display()
            ));
        }
        seen.push(canon.clone());
        let text = std::fs::read_to_string(&full_path)
            .map_err(|e| format!("failed to read extends file '{}': {e}", full_path.display()))?;
        let inherited_root: Value = crate::artifact::toml::from_str(&text).map_err(|e| {
            format!(
                "failed to parse extends file '{}': {e}",
                full_path.display()
            )
        })?;
        let next_dir = full_path.parent().unwrap_or_else(|| Path::new("."));
        let sub_bases = load_bases(&inherited_root, next_dir, seen)?;
        seen.pop();

        let mut top = inherited_root.clone();
        if let Some(m) = top.as_map_mut() {
            m.remove("extends");
            m.remove("study");
        }
        let top = normalize_paths(top);

        for mut base in sub_bases {
            deep_merge(&mut base, &top);
            clean_variants(&mut base, &top);
            prune_deletes(&mut base);
            result.push(base);
        }
    }
    Ok(result)
}

fn extract_spec(root: &Value) -> Result<RawStudySpec, String> {
    let study_map = match root
        .as_map()
        .and_then(|m| m.get("study"))
        .and_then(|v| v.as_map())
    {
        Some(m) => m,
        None => return Ok(default_spec()),
    };

    let rungs = extract_rungs(study_map);
    let prune = match study_map.get("prune").and_then(|v| v.as_str()) {
        Some(s) => StudyPrune::parse_opt(s)?,
        None => StudyPrune::None,
    };

    let raw_arms = study_map
        .get("arm")
        .or_else(|| study_map.get("arms"))
        .and_then(|v| v.as_seq());

    let mut arms = Vec::new();
    if let Some(seq) = raw_arms {
        for arm_val in seq.iter() {
            if let Some(arm) = parse_arm(arm_val) {
                arms.push(arm);
            }
        }
    }

    Ok(RawStudySpec {
        rungs,
        prune,
        arms,
        had_study: true,
    })
}

fn default_spec() -> RawStudySpec {
    RawStudySpec {
        rungs: Vec::new(),
        prune: StudyPrune::None,
        arms: vec![ArmDef {
            name: "default".into(),
            baseline: true,
            set: Value::from(Map::new()),
            sweep: Vec::new(),
        }],
        had_study: false,
    }
}

fn extract_rungs(study_map: &Map) -> Vec<u32> {
    let seq = match study_map.get("rungs").and_then(|v| v.as_seq()) {
        Some(s) => s,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    for item in seq.iter() {
        if let Some(n) = item.as_u64() {
            out.push(n as u32);
        }
    }
    out
}

fn parse_arm(arm_val: &Value) -> Option<ArmDef> {
    let m = arm_val.as_map()?;
    let name = m
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("arm")
        .to_string();
    let baseline = m.get("baseline").and_then(|v| v.as_bool()).unwrap_or(false);
    let set = m
        .get("set")
        .cloned()
        .unwrap_or_else(|| Value::from(Map::new()));
    let norm_set = normalize_paths(set);
    let sweep = parse_sweep(m.get("sweep"));
    Some(ArmDef {
        name,
        baseline,
        set: norm_set,
        sweep,
    })
}

fn parse_sweep(sweep_val: Option<&Value>) -> Vec<(String, Vec<Value>)> {
    let sw_map = match sweep_val.and_then(|v| v.as_map()) {
        Some(m) => m,
        None => return Vec::new(),
    };
    let mut keys: Vec<String> = sw_map
        .keys()
        .filter_map(|k| k.as_str().map(String::from))
        .collect();
    keys.sort();
    let mut sweep = Vec::new();
    for k in keys {
        if let Some(candidates) = sw_map.get(&k) {
            let cand_list = candidates
                .as_seq()
                .map(|s| s.to_vec())
                .unwrap_or_else(|| vec![candidates.clone()]);
            sweep.push((k, cand_list));
        }
    }
    sweep
}

fn expand_arm(merged_base: &Value, env_id: &str, arm: &ArmDef, out: &mut Vec<IntermediateCell>) {
    let mut arm_base = merged_base.clone();
    deep_merge(&mut arm_base, &arm.set);
    clean_variants(&mut arm_base, &arm.set);
    prune_deletes(&mut arm_base);

    if arm.sweep.is_empty() {
        out.push(IntermediateCell {
            env_id: env_id.to_string(),
            arm_id: arm.name.clone(),
            baseline: arm.baseline,
            factors: Vec::new(),
            leaf: arm_base,
        });
        return;
    }

    let combos = cartesian_product(&arm.sweep);
    for combo in combos {
        let mut cell_leaf = arm_base.clone();
        let mut factors = Vec::with_capacity(combo.len());
        let mut factor_overlay = Value::from(Map::new());
        for (k, v) in &combo {
            let slug_val = value_slug(v);
            factors.push((k.clone(), slug_val));
            set_path(&mut cell_leaf, k, v.clone());
            set_path(&mut factor_overlay, k, v.clone());
        }
        clean_variants(&mut cell_leaf, &factor_overlay);
        prune_deletes(&mut cell_leaf);
        out.push(IntermediateCell {
            env_id: env_id.to_string(),
            arm_id: arm.name.clone(),
            baseline: arm.baseline,
            factors,
            leaf: cell_leaf,
        });
    }
}

fn cartesian_product(sweep: &[(String, Vec<Value>)]) -> Vec<Vec<(String, Value)>> {
    let mut result = vec![Vec::new()];
    for (key, candidates) in sweep {
        let mut next = Vec::new();
        for prev in &result {
            for cand in candidates {
                let mut current = prev.clone();
                current.push((key.clone(), cand.clone()));
                next.push(current);
            }
        }
        result = next;
    }
    result
}

fn apply_arg(cells: Vec<IntermediateCell>, cli_arg: &str) -> Result<Vec<IntermediateCell>, String> {
    let (path, raw_val) = cli_arg
        .split_once('=')
        .ok_or_else(|| format!("invalid cli set: expected 'path=val', got '{cli_arg}'"))?;
    let path = path.trim();
    let items = split_commas(raw_val);

    if items.len() > 1 {
        let mut new_cells = Vec::with_capacity(cells.len() * items.len());
        for cell in cells {
            for item in &items {
                let parsed = parse_value(item);
                let mut next_cell = cell.clone();
                next_cell
                    .factors
                    .push((path.to_string(), value_slug(&parsed)));
                set_path(&mut next_cell.leaf, path, parsed.clone());
                let mut overlay = Value::from(Map::new());
                set_path(&mut overlay, path, parsed);
                clean_variants(&mut next_cell.leaf, &overlay);
                prune_deletes(&mut next_cell.leaf);
                new_cells.push(next_cell);
            }
        }
        Ok(new_cells)
    } else {
        let parsed = parse_value(items.first().map(|s| s.as_str()).unwrap_or(""));
        let mut updated = cells;
        for cell in &mut updated {
            set_path(&mut cell.leaf, path, parsed.clone());
            let mut overlay = Value::from(Map::new());
            set_path(&mut overlay, path, parsed.clone());
            clean_variants(&mut cell.leaf, &overlay);
            prune_deletes(&mut cell.leaf);
        }
        Ok(updated)
    }
}

fn apply_scalars(cells: &mut [IntermediateCell], opts: &CompileOpts<'_>) {
    for cell in cells {
        if let Some(r) = opts.rounds {
            set_path(&mut cell.leaf, "rounds", Value::from(r as u64));
        }
        if let Some(rep) = opts.repetitions {
            set_path(&mut cell.leaf, "repetitions", Value::from(rep as u64));
        }
    }
}

fn resolve_baselines(cells: &mut [IntermediateCell]) {
    let mut envs = Vec::new();
    for cell in cells.iter() {
        if !envs.contains(&cell.env_id) {
            envs.push(cell.env_id.clone());
        }
    }
    for env in envs {
        let has_baseline = cells.iter().any(|c| c.env_id == env && c.baseline);
        if !has_baseline {
            if let Some(first_arm) = cells
                .iter()
                .find(|c| c.env_id == env)
                .map(|c| c.arm_id.clone())
            {
                for c in cells.iter_mut() {
                    if c.env_id == env && c.arm_id == first_arm {
                        c.baseline = true;
                    }
                }
            }
        }
    }
}

fn finalize_cells(cells: Vec<IntermediateCell>) -> Result<Vec<StudyCell>, String> {
    let distinct_envs: HashSet<&str> = cells.iter().map(|c| c.env_id.as_str()).collect();
    let multi_envs = distinct_envs.len() > 1;

    let mut finalized = Vec::with_capacity(cells.len());
    for mut cell in cells {
        let base_slug = if cell.factors.is_empty() {
            cell.arm_id.clone()
        } else {
            let slugs: Vec<&str> = cell.factors.iter().map(|(_, v)| v.as_str()).collect();
            format!("{}_{}", cell.arm_id, slugs.join("_"))
        };
        let cell_id = if multi_envs {
            format!("{}__{base_slug}", cell.env_id)
        } else {
            base_slug
        };

        if let Some(m) = cell.leaf.as_map_mut() {
            m.remove("extends");
            m.remove("study");
        }

        let rounds = cell
            .leaf
            .as_map()
            .and_then(|m| m.get("rounds"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        let repetitions = cell
            .leaf
            .as_map()
            .and_then(|m| m.get("repetitions"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        let experiment = cell
            .leaf
            .as_map()
            .and_then(|m| m.get("experiment"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let canonical_toml = crate::artifact::toml::pretty_string(&cell.leaf)
            .map_err(|e| format!("failed to serialize cell canonical TOML: {e}"))?;

        finalized.push(StudyCell {
            env_id: cell.env_id,
            arm_id: cell.arm_id,
            cell_id,
            baseline: cell.baseline,
            factors: cell.factors,
            experiment,
            rounds,
            repetitions,
            canonical_toml,
            value: cell.leaf,
        });
    }
    Ok(finalized)
}

fn compute_env(leaf: &Value) -> String {
    let experiment = leaf
        .as_map()
        .and_then(|m| m.get("experiment"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if experiment == "oceananigans" {
        let task = leaf
            .as_map()
            .and_then(|m| m.get("ocean"))
            .and_then(|v| v.as_map())
            .and_then(|m| m.get("task"))
            .and_then(|v| v.as_str())
            .unwrap_or("kelvin-helmholtz");
        format!("oceananigans:{task}")
    } else if experiment == "fuse" {
        let fuse = leaf
            .as_map()
            .and_then(|m| m.get("fuse"))
            .and_then(|v| v.as_map());
        let case = fuse
            .and_then(|m| m.get("case"))
            .and_then(|v| v.as_str())
            .unwrap_or("ARC");
        let wf = fuse
            .and_then(|m| m.get("workflow"))
            .and_then(|v| v.as_str())
            .unwrap_or("whole-facility");
        format!("fuse:{case}:{wf}")
    } else {
        experiment.to_string()
    }
}

fn value_slug(val: &Value) -> String {
    if let Some(s) = val.as_str() {
        s.to_string()
    } else if let Some(n) = val.as_u64() {
        n.to_string()
    } else if let Some(n) = val.as_i64() {
        n.to_string()
    } else if let Some(f) = val.as_f64() {
        f.to_string()
    } else if let Some(b) = val.as_bool() {
        b.to_string()
    } else {
        crate::artifact::toml::pretty_string(val).unwrap_or_else(|_| "val".into())
    }
}
