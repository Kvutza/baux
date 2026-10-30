//! Stateful matrix optimization on the same fixed corpus objective as `long`.
//! Repeated training-window exposure is recorded separately from scorer work.
use super::*;
use crate::config::{ModelInitialization, RunConfig};
use crate::pretrain_data::{CorpusBatch, CorpusSet};

struct Session<'a> {
    run: &'a RunConfig,
    runtime: std::sync::Arc<Runtime>,
    decoder: decode::Decoder,
    scorer: block_decode::BlockDecoder,
    trainer: train::Trainer,
    optimizer: train::Optimizer,
    batch: CorpusBatch,
    validation: Option<CorpusSet>,
}

impl<'a> Session<'a> {
    fn new(run: &'a RunConfig, path: &std::path::Path) -> Result<Self, String> {
        run.validate_experiment()?;
        let dataset = CorpusSet::load(path)?;
        let batch = dataset.causal_batch(CONTEXT as usize, run.reference_seed(), 0)?;
        if batch.batch.tokens.len() != CONTEXT as usize
            || batch.batch.loss_mask.iter().any(|&mask| mask != 1)
        {
            return Err("matrix training requires a complete fully scored corpus window".into());
        }
        let validation = run
            .validation_dataset
            .as_deref()
            .map(CorpusSet::load_eval)
            .transpose()?;
        let runtime = Runtime::shared()?;
        let architecture =
            ResidualArchitecture::from_model(run.model.ok_or("matrix training requires a model")?);
        dataset.require_vocabulary(architecture.vocabulary())?;
        if let Some(validation) = &validation {
            validation.require_vocabulary(architecture.vocabulary())?;
        }
        let initial = CandidateWeights::initialized(
            &runtime,
            run.model_seed(),
            ModelInitialization::FbtPisa1Mhc4,
            architecture,
        )?;
        let trainer = train::Trainer::new(&runtime, &initial, CONTEXT)?;
        drop(initial);
        let optimizer = train::Optimizer::new(&trainer, run.optimizer())?;
        let decoder = decode::Decoder::with_visits(
            &runtime,
            CONTEXT,
            false,
            architecture.layer_steps().len(),
        )?;
        let scorer =
            block_decode::BlockDecoder::with_architecture(&runtime, CONTEXT, architecture)?;
        Ok(Self {
            run,
            runtime,
            decoder,
            scorer,
            trainer,
            optimizer,
            batch,
            validation,
        })
    }

    fn validate(&self, round: u32) -> Result<Option<baux::artifact::json::Value>, String> {
        self.validation
            .as_ref()
            .map(|data| {
                long::validate(
                    &self.scorer,
                    &self.runtime,
                    &self.decoder,
                    self.trainer.row(),
                    data,
                    CONTEXT as usize,
                    self.run.reference_seed(),
                    round,
                )
            })
            .transpose()
    }

    fn round(
        &mut self,
        round: u32,
    ) -> Result<(baux::artifact::json::Value, f64, f64, f64), String> {
        let started = Instant::now();
        let (before, backward) = self.trainer.backward_corpus(
            &self.runtime,
            &self.decoder,
            &self.scorer,
            &self.batch.batch,
        )?;
        let update = self.optimizer.encode(&self.runtime, &self.trainer)?;
        self.scorer.clear_pair();
        let after = self.scorer.score_batch(
            &self.runtime,
            &self.decoder,
            self.trainer.row(),
            &self.batch.batch,
        )?;
        let wall = started.elapsed().as_secs_f64();
        let gpu = before.gpu_seconds + backward + update + after.gpu_seconds;
        let record = baux::artifact::json::json!({
            "round":round,"optimizer":self.run.optimizer(),
            "objective":"fixed-corpus-causal-likelihood","loss":"cross-entropy",
            "training_schedule":"repeated-initial-window","source":&self.batch.source,
            "corpus_offset":self.batch.batch.offset,"model_seed":self.run.model_seed(),
            "sampling_seed":self.run.reference_seed(),"scored_tokens":before.quality.scored_tokens,
            "cumulative_tokens":u64::from(round) * u64::from(CONTEXT),
            "cumulative_scoring_tokens":u64::from(round) * 2 * u64::from(CONTEXT),
            "corpus_epochs":null,"persistent_optimizer_state":true,
            "nll_before":before.quality.mean_nll,"nll_after":after.quality.mean_nll,
            "forward_gpu_seconds":before.gpu_seconds,"backward_gpu_seconds":backward,
            "update_gpu_seconds":update,"rescore_gpu_seconds":after.gpu_seconds,
            "wall_seconds":wall,"update_statistics_available":false,
        });
        eprintln!(
            "pretrain.matrix round={round} nll={:.9}->{:.9} wall_seconds={wall:.3}",
            before.quality.mean_nll, after.quality.mean_nll
        );
        Ok((record, wall, gpu, backward + update))
    }
}

pub(super) fn run(run: &RunConfig, path: &std::path::Path) -> Result<ActualBoResult, String> {
    let mut session = Session::new(run, path)?;
    let mut validation = Vec::new();
    if let Some(record) = session.validate(0)? {
        validation.push(record);
    }
    let started = Instant::now();
    let mut wall = Vec::new();
    let mut gpu = Vec::new();
    let mut optimizer_seconds = Vec::new();
    let mut optimization_records = Vec::new();
    for round in 1..=run.rounds() {
        let (record, seconds, device, optimizer) = session.round(round)?;
        optimization_records.push(record);
        wall.push(seconds);
        gpu.push(device);
        optimizer_seconds.push(optimizer);
        if (round % run.validation_interval.unwrap_or(run.rounds()) == 0 || round == run.rounds())
            && let Some(record) = session.validate(round)?
        {
            validation.push(record);
        }
    }
    let loop_seconds = started.elapsed().as_secs_f64();
    wall.sort_by(f64::total_cmp);
    gpu.sort_by(f64::total_cmp);
    Ok(ActualBoResult {
        parameters: ResidualArchitecture::Mhc4.parameter_count(),
        loop_seconds,
        median_wall_seconds: wall[wall.len() / 2],
        min_wall_seconds: wall[0],
        max_wall_seconds: wall[wall.len() - 1],
        median_gpu_seconds: gpu[gpu.len() / 2],
        accepted: run.rounds(),
        learning_seconds: loop_seconds,
        validation,
        optimizer_seconds,
        optimization_records,
        updates: UpdateLog {
            rounds: run.rounds(),
            accepted_rounds: run.rounds(),
            blocks: Vec::new(),
        },
    })
}
