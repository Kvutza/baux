//! Persistent, bounded-cache Hardcaml worker process and RTL export.
use super::{
    architecture::{Config, Design},
    flow, matvec,
};
use baux::artifact::json::{self, Value, json};
use std::{
    collections::HashMap,
    fs,
    io::{self, BufRead, Write},
    path::Path,
    process::{self, Command},
    time::Instant,
};

pub(super) struct Worker {
    matrix: Option<(matvec::Config, matvec::Shape, u64)>,
    flow: Option<(flow::Config, flow::Model, u64)>,
    planned_cycles: HashMap<Design, usize>,
    child: process::Child,
    input: process::ChildStdin,
    output: io::BufReader<process::ChildStdout>,
}

impl Worker {
    pub(super) fn spawn(
        binary: &Path,
        config: &Config,
        shape: Option<matvec::Shape>,
        model: Option<flow::Model>,
        seed: u64,
    ) -> Result<Self, String> {
        let mut child = Command::new(binary)
            .arg(if config.flow.is_some() {
                "flow-worker"
            } else if config.matrix.is_some() {
                "matvec-worker"
            } else {
                "worker"
            })
            .arg(config.cache_capacity.to_string())
            .stdin(process::Stdio::piped())
            .stdout(process::Stdio::piped())
            .stderr(process::Stdio::inherit())
            .spawn()
            .map_err(|e| e.to_string())?;
        let input = child.stdin.take().ok_or("worker stdin missing")?;
        let output = io::BufReader::new(child.stdout.take().ok_or("worker stdout missing")?);
        let mut worker = Self {
            matrix: config.matrix.clone().zip(shape).map(|(m, s)| (m, s, seed)),
            flow: config.flow.clone().zip(model).map(|(f, m)| (f, m, seed)),
            planned_cycles: HashMap::new(),
            child,
            input,
            output,
        };
        let ready = worker.read()?;
        if ready["ready"].as_bool() != Some(true) || ready["version"].as_u64() != Some(1) {
            return Err("invalid Hardcaml worker handshake".into());
        }
        worker.plan(config)?;
        Ok(worker)
    }

    fn read(&mut self) -> Result<Value, String> {
        let mut line = String::new();
        if self
            .output
            .read_line(&mut line)
            .map_err(|e| e.to_string())?
            == 0
        {
            return Err("Hardcaml worker closed its output".into());
        }
        let value: Value = json::from_str(&line).map_err(|e| e.to_string())?;
        if let Some(error) = value["error"].as_str() {
            return Err(error.into());
        }
        Ok(value)
    }

    pub(super) fn evaluate(
        &mut self,
        d: Design,
        target_seed: u32,
        weights: &[f64],
        context: &Value,
    ) -> Result<Value, String> {
        let request = if let Some((flow, model, seed)) = &self.flow {
            flow.request(*model, &json!(d), target_seed, weights, context, *seed)
        } else if let Some((matrix, shape, seed)) = &self.matrix {
            matrix.request(*shape, &json!(d), target_seed, weights, context, *seed)
        } else {
            json!({"version": 1, "dimension": weights.len(), "bits": d.bits,
            "lanes": d.lanes, "pipeline": d.pipeline, "target_seed": target_seed, "weights": weights})
        };
        json::write_line(&mut self.input, &request).map_err(|e| e.to_string())?;
        self.input.flush().map_err(|e| e.to_string())?;
        self.read()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Worker {
    pub(super) fn expected_cycles(
        &self,
        design: Design,
        dimension: usize,
        context: &Value,
    ) -> Option<usize> {
        if self.flow.is_some() {
            return self.planned_cycles.get(&design).copied();
        }
        Some(match &self.matrix {
            Some((matrix, shape, _)) => matrix.query_cycles(
                *shape,
                design.lanes,
                design.banks,
                design.pipeline,
                design.queue,
                context["window"].as_u64().unwrap_or(0) as usize,
            ),
            None => design.cycles(dimension),
        })
    }

    pub(super) fn is_flow(&self) -> bool {
        self.flow.is_some()
    }
    pub(super) fn is_matrix(&self) -> bool {
        self.matrix.is_some()
    }
    pub(super) fn export_rtl(&mut self, design: Design, directory: &Path) -> Result<f64, String> {
        let start = Instant::now();
        let mut request = if let Some((flow, model, seed)) = &self.flow {
            flow.request(*model, &json!(design), 0, &[], &json!({}), *seed)
        } else if let Some((matrix, shape, seed)) = &self.matrix {
            matrix.request(*shape, &json!(design), 0, &[], &json!({}), *seed)
        } else {
            return Ok(0.0);
        };
        request["op"] = json!("rtl");
        json::write_line(&mut self.input, &request).map_err(|e| e.to_string())?;
        self.input.flush().map_err(|e| e.to_string())?;
        let response = self.read()?;
        fs::write(
            directory.join("final.v"),
            response["rtl"].as_str().ok_or("missing RTL")?,
        )
        .map_err(|e| e.to_string())?;
        Ok(start.elapsed().as_secs_f64())
    }
}

impl Worker {
    pub(super) fn generate(
        &mut self,
        design: Design,
        weights: &[f64],
        regime: u32,
        window: usize,
        directory: &Path,
    ) -> Result<Value, String> {
        let Some((flow, model, seed)) = &self.flow else {
            return Ok(json!(null));
        };
        let mut request = flow.request(
            *model,
            &json!(design),
            regime,
            weights,
            &json!({"window":window,"phase":"prequential"}),
            *seed,
        );
        request["op"] = json!("generate");
        json::write_line(&mut self.input, &request).map_err(|e| e.to_string())?;
        self.input.flush().map_err(|e| e.to_string())?;
        let result = self.read()?;
        fs::write(
            directory.join("final-checkpoint.json"),
            json::pretty_string(&result["checkpoint"]).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        fs::write(
            directory.join("generation.json"),
            json::pretty_string(&result).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(
            json!({"cycles":result["cycles"],"neural_function_evaluations":result["neural_function_evaluations"],
            "metrics":result["metrics"],"max_reference_error":result["max_reference_error"],"kernel_counters":result["kernel_counters"]}),
        )
    }
}

impl Worker {
    fn plan(&mut self, config: &Config) -> Result<(), String> {
        let Some((flow, model, seed)) = self.flow.clone() else {
            return Ok(());
        };
        for design in config.designs() {
            let mut request = flow.request(model, &json!(design), 0, &[], &json!({}), seed);
            request["op"] = json!("plan");
            json::write_line(&mut self.input, &request).map_err(|e| e.to_string())?;
            self.input.flush().map_err(|e| e.to_string())?;
            let result = self.read()?;
            let cost = result["cycles"].as_u64().ok_or("missing planned cycles")? as usize;
            if cost == 0 || config.cycle_budget / cost < 3 {
                return Err(format!(
                    "FlowLOB cycle-budget {} needs at least {} for {:?}",
                    config.cycle_budget,
                    3 * cost,
                    design
                ));
            }
            self.planned_cycles.insert(design, cost);
        }
        Ok(())
    }
}

impl Worker {
    pub(super) fn snapshot(binary: &Path, output: &Path) -> Result<std::path::PathBuf, String> {
        let snapshot = output.join("worker.exe");
        fs::copy(binary, &snapshot).map_err(|e| e.to_string())?;
        fs::write(
            output.join("worker.sha256"),
            baux::artifact::runs::file_digest(&snapshot)?,
        )
        .map_err(|e| e.to_string())?;
        Ok(snapshot)
    }
}
