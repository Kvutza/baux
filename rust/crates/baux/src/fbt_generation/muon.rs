use super::generation::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn select(
    run: &RunConfig,
    config: &GenerationConfig,
    output: &Path,
    objective: &baux::artifact::json::Value,
    tokenizer: Option<&ByteDecoder>,
    validation: &[GenerationTask],
    dataset: Option<(&Path, CorpusCaps)>,
) -> Option<Result<(), String>> {
    matches!(
        run.optimizer(),
        crate::config::TrainingOptimizer::MuonV1 | crate::config::TrainingOptimizer::AdamwV1
    )
    .then(|| {
        Session::new(
            run, config, output, objective, tokenizer, validation, dataset,
        )?
        .run()
    })
}

struct Session<'a> {
    run: &'a RunConfig,
    config: &'a GenerationConfig,
    output: &'a Path,
    tokenizer: Option<&'a ByteDecoder>,
    runtime: std::sync::Arc<Runtime>,
    decoder: decode::Decoder,
    verifier: block_decode::BlockDecoder,
    evaluator: Option<Evaluator>,
    trainer: train::Trainer,
    optimizer: train::Optimizer,
    weights: CandidateWeights,
    source: Option<(crate::pretrain_data::PretrainDataset, CorpusCaps)>,
    validation: Vec<GenerationTask>,
    seeds: GenerationSeeds,
    architecture: ResidualArchitecture,
    records: BufWriter<File>,
    validation_records: BufWriter<File>,
    validation_initial: Vec<f32>,
    validation_final: Vec<f32>,
    walls: Vec<f64>,
    first_nll: Option<f32>,
    last_nll: Option<f32>,
    started: Instant,
}

struct RoundResult {
    before: f32,
    after: f32,
    forward: f64,
    rescore: f64,
    backward: f64,
    update: f64,
    wall: f64,
}

impl<'a> Session<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        run: &'a RunConfig,
        config: &'a GenerationConfig,
        output: &'a Path,
        objective: &baux::artifact::json::Value,
        tokenizer: Option<&'a ByteDecoder>,
        validation: &[GenerationTask],
        dataset: Option<(&Path, CorpusCaps)>,
    ) -> Result<Self, String> {
        verify_policy(config)?;
        let architecture = model_arch(run)?;
        if architecture != ResidualArchitecture::Mhc4 {
            return Err("Matrix optimizer training currently requires fbt-pisa1-mhc4-v1".into());
        }
        let resources = prepare_resources(
            config,
            output,
            validation,
            tokenizer,
            architecture,
            run.scorer_stage_samples,
            None,
        )?;
        let verifier = resources
            .verifier
            .ok_or("Matrix optimizer mHC4 training requires accepted-prefix generation")?;
        let seeds = GenerationSeeds::new(run, config);
        let initial = CandidateWeights::initialized(
            &resources.runtime,
            seeds.model,
            config.initialization,
            architecture,
        )?;
        if let Some(path) = &config.checkpoint {
            load_checkpoint(&initial, path)?;
        }
        let trainer = train::Trainer::new(&resources.runtime, &initial, resources.context)?;
        let optimizer = train::Optimizer::new(&trainer, run.optimizer())?;
        let source = dataset
            .map(|(path, caps)| {
                crate::pretrain_data::PretrainDataset::load(path).map(|data| (data, caps))
            })
            .transpose()?;
        let records = BufWriter::new(
            File::create(output.join("optimization.jsonl")).map_err(|error| error.to_string())?,
        );
        let validation_records = BufWriter::new(
            File::create(output.join("validation.jsonl")).map_err(|error| error.to_string())?,
        );
        write_config(
            run,
            config,
            output,
            objective,
            &resources.runtime,
            resources.context,
            seeds.model,
            seeds.proposal,
            seeds.acquisition,
            seeds.sampling,
        )?;
        Ok(Self {
            run,
            config,
            output,
            tokenizer,
            runtime: resources.runtime,
            decoder: resources.decoder,
            verifier,
            evaluator: resources.evaluator,
            trainer,
            optimizer,
            weights: initial,
            source,
            validation: validation.to_vec(),
            seeds,
            architecture,
            records,
            validation_records,
            validation_initial: Vec::new(),
            validation_final: Vec::new(),
            walls: Vec::with_capacity(run.rounds() as usize),
            first_nll: None,
            last_nll: None,
            started: Instant::now(),
        })
    }

    fn optimizer_name(&self) -> &'static str {
        match self.run.optimizer() {
            crate::config::TrainingOptimizer::MuonV1 => "muon",
            crate::config::TrainingOptimizer::AdamwV1 => "adamw-v1",
            _ => "unknown",
        }
    }

    fn run(mut self) -> Result<(), String> {
        self.validation_initial = self.validate(0, "validation-initial")?.rewards;
        for round in 0..self.run.rounds() {
            self.round(round)?;
            let step = round + 1;
            if step < self.run.rounds()
                && (learning_checkpoint(step)
                    || self
                        .run
                        .validation_interval
                        .is_some_and(|interval| step % interval == 0))
            {
                self.validate(step, &format!("validation-{step:06}"))?;
            }
        }
        self.validation_final = self
            .validate(self.run.rounds(), "validation-final")?
            .rewards;
        self.finish()
    }

    fn validate(&mut self, round: u32, directory: &str) -> Result<Validation, String> {
        let started = Instant::now();
        let validation = validate_generation(
            &self.runtime,
            &self.decoder,
            Some(&self.verifier),
            &self.weights,
            self.trainer.buffer(),
            self.config,
            self.seeds.sampling,
            &self.output.join(directory),
            self.evaluator.as_mut(),
            self.tokenizer,
            &self.validation,
        )?;
        let optimizer = self.optimizer_name();
        baux::artifact::json::write_line(
            &mut self.validation_records,
            &baux::artifact::json::json!({
                "schema":"baux.pretrain_validation.v1","optimizer":optimizer,"round":round,
                "reward":validation.mean(),"reward_stddev":validation.stddev(),
                "reward_standard_error":validation.standard_error(),"tasks":validation.rewards.len(),
                "task_rewards":&validation.rewards,"task_objectives":&validation.objectives,
                "objective_means":validation.objective_means(),
                "validation_seconds":started.elapsed().as_secs_f64(),"used_for_acceptance":false,
            }),
        )
        .map_err(|error| error.to_string())?;
        self.validation_records
            .flush()
            .map_err(|error| error.to_string())?;
        Ok(validation)
    }

    fn condition(&self, round: u32) -> Result<(GenerationConfig, usize), String> {
        let mut config = self.config.clone();
        let Some((data, caps)) = &self.source else {
            return Ok((config, 0));
        };
        let (task, offset) = setup::corpus_window(
            data,
            &self.config.tasks[0],
            caps.prompt,
            caps.output,
            self.seeds.sampling,
            round,
        )?;
        config.max_tokens =
            u32::try_from(task.expected.len()).map_err(|error| error.to_string())?;
        config.tasks = vec![task];
        Ok((config, offset))
    }

    fn round(&mut self, round: u32) -> Result<(), String> {
        let started = Instant::now();
        let (config, corpus_offset) = self.condition(round)?;
        let path = self.output.join(format!("round-{:06}", round + 1));
        let evaluation = self.evaluate(&config, round, &path, true)?;
        let before = nll(&evaluation)?;
        self.first_nll.get_or_insert(before);
        let backward = self.trainer.backward(
            &self.runtime,
            &self.verifier,
            &config.tasks[0],
            config.max_tokens,
        )?;
        let update = self.optimizer.encode(&self.runtime, &self.trainer)?;
        let post = self.evaluate(&config, round, &path.join("post-update"), false)?;
        let after = nll(&post)?;
        self.last_nll = Some(after);
        let result = RoundResult {
            before,
            after,
            forward: gpu_time(&evaluation),
            rescore: gpu_time(&post),
            backward,
            update,
            wall: started.elapsed().as_secs_f64(),
        };
        self.walls.push(result.wall);
        self.record(round, &config, corpus_offset, &result)?;
        eprintln!(
            "training.round step={} optimizer={} nll={:.6}->{:.6} delta={:+.6} forward_ms={:.1} \
             backward_ms={:.1} update_ms={:.1} rescore_ms={:.1} wall_ms={:.1}",
            round + 1,
            self.optimizer_name(),
            before,
            after,
            after - before,
            result.forward * 1000.0,
            backward * 1000.0,
            update * 1000.0,
            result.rescore * 1000.0,
            result.wall * 1000.0,
        );
        Ok(())
    }

    fn evaluate(
        &mut self,
        config: &GenerationConfig,
        round: u32,
        path: &Path,
        persist: bool,
    ) -> Result<Evaluation, String> {
        score(
            &self.runtime,
            &self.decoder,
            Some((&self.verifier, None)),
            self.trainer.row(),
            config,
            crate::hash::splitmix64(self.seeds.sampling ^ u64::from(round)),
            path,
            self.evaluator.as_mut(),
            self.tokenizer,
            persist && (round == 0 || round + 1 == self.run.rounds()),
        )
    }

    fn record(
        &mut self,
        round: u32,
        config: &GenerationConfig,
        corpus_offset: usize,
        result: &RoundResult,
    ) -> Result<(), String> {
        let optimizer = self.optimizer_name();
        baux::artifact::json::write_line(
            &mut self.records,
            &baux::artifact::json::json!({
                "round":round + 1,"optimizer":optimizer,"objective":"cross-entropy",
                "nll_before":result.before,"nll_after":result.after,
                "nll_delta":result.after - result.before,
                "input_tokens":config.tasks[0].prompt.len(),
                "target_tokens":config.max_tokens,"corpus_offset":corpus_offset,
                "forward_gpu_ms":result.forward * 1000.0,
                "backward_gpu_ms":result.backward * 1000.0,
                "update_gpu_ms":result.update * 1000.0,
                "rescore_gpu_ms":result.rescore * 1000.0,
                "round_wall_ms":result.wall * 1000.0,
            }),
        )
        .map_err(|error| error.to_string())?;
        self.records.flush().map_err(|error| error.to_string())
    }

    fn finish(mut self) -> Result<(), String> {
        if self.config.record_tensor_updates {
            self.weights.write_stats(
                self.trainer.weights(),
                &self.output.join("tensor-state.jsonl"),
            )?;
        }
        if self.config.save_final_checkpoint {
            let path = self
                .config
                .save_checkpoint
                .clone()
                .unwrap_or_else(|| self.output.join("checkpoint.safetensors"));
            save_checkpoint(self.trainer.weights(), &path, self.architecture)?;
        }
        self.walls.sort_by(f64::total_cmp);
        let loop_seconds = self.walls.iter().sum::<f64>();
        let mean = self.walls.iter().sum::<f64>() / self.walls.len() as f64;
        let median = self.walls[self.walls.len() / 2];
        let goal = self
            .walls
            .iter()
            .all(|seconds| *seconds * 1000.0 <= f64::from(self.run.target_ms()));
        self.write_result(loop_seconds, mean, median, goal)?;
        eprintln!(
            "training.complete optimizer={} rounds={} initial_nll={:.6} final_observed_nll={:.6} \
             loop_seconds={:.3} mean_round_ms={:.1} median_round_ms={:.1} target_round_ms={} \
             goal_met={}",
            self.optimizer_name(),
            self.run.rounds(),
            self.first_nll.unwrap(),
            self.last_nll.unwrap(),
            loop_seconds,
            mean * 1000.0,
            median * 1000.0,
            self.run.target_ms(),
            goal,
        );
        Ok(())
    }

    fn write_result(
        &self,
        loop_seconds: f64,
        mean: f64,
        median: f64,
        goal: bool,
    ) -> Result<(), String> {
        let schema = match self.run.optimizer() {
            crate::config::TrainingOptimizer::MuonV1 => "baux.muon-pretraining.v1",
            crate::config::TrainingOptimizer::AdamwV1 => "baux.adamw-pretraining.v1",
            _ => "baux.matrix-pretraining.v1",
        };
        baux::artifact::json::pretty_writer(
            File::create(self.output.join("result.json")).map_err(|error| error.to_string())?,
            &baux::artifact::json::json!({
                "schema":schema,"status":"completed","optimizer":self.optimizer_name(),
                "rounds":self.run.rounds(),"parameters":self.architecture.parameter_count(),
                "initial_nll":self.first_nll.unwrap(),"final_observed_nll":self.last_nll.unwrap(),
                "loop_seconds":loop_seconds,"elapsed_seconds":self.started.elapsed().as_secs_f64(),
                "mean_round_ms":mean * 1000.0,
                "median_round_ms":median * 1000.0,"target_round_ms":self.run.target_ms(),
                "goal_met":goal,"validation_tasks":self.validation.len(),
                "validation_initial":&self.validation_initial,"validation_final":&self.validation_final,
            }),
        )
        .map_err(|error| error.to_string())
    }
}

fn gpu_time(evaluation: &Evaluation) -> f64 {
    evaluation
        .rollouts
        .iter()
        .map(|rollout| rollout.gpu_seconds)
        .sum()
}

fn nll(evaluation: &Evaluation) -> Result<f32, String> {
    evaluation
        .rollouts
        .first()
        .and_then(|rollout| rollout.conditional_target_nll)
        .ok_or_else(|| "Matrix optimizer training requires conditional target NLL".into())
}
