//! GPU geometry for foreign evaluated weights entering a local model.
//! The full row stays resident; only bounded distance scalars reach the host.

use super::*;
use crate::apple_gpu::thread_group;

pub(super) struct ImportGeometry {
    pub(super) distances: Vec<(usize, f32)>,
    pub(super) center: f32,
}

impl SearchState {
    pub(super) fn measure_import(&self, incoming: &Buffer) -> Result<ImportGeometry, String> {
        self.check_healthy()?;
        if self.async_command.is_some() {
            return Err("Finish the resident GPU command before measuring an import".into());
        }
        if self.history == 0 || self.resident_history == 0 {
            return Err("Measure the regional incumbent before importing observations".into());
        }
        if incoming.length() != self.row_bytes() {
            return Err("Imported resident row has the wrong byte length".into());
        }
        if self.axis.is_some() {
            return Err(
                "Resident cross-region import does not support a scalar search axis".into(),
            );
        }

        let pairs = self.resident_history.div_ceil(2);
        let tiles = u32::try_from(self.tiles.len()).map_err(|_| "Too many resident tiles")?;
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.import_pipeline);
        for pair in 0..pairs {
            let first = pair * 2;
            let params = Params {
                initialize: pair as u32,
                history: (self.resident_history - first).min(2) as u32,
                tiles,
                ..Params::default()
            };
            let buffers = [
                incoming,
                &self.history_rows[first],
                &self.history_rows[(first + 1).min(self.resident_history - 1)],
                &self.base,
                &self.leaves_gpu,
                &self.tiles_gpu,
                &self.partials,
            ];
            for (slot, buffer) in buffers.iter().enumerate() {
                encoder.set_buffer(slot as u64, Some(buffer), 0);
            }
            encoder.set_bytes(
                7,
                size_of::<Params>() as u64,
                (&params as *const Params).cast(),
            );
            encoder
                .dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        }
        encoder.end_encoding();

        let reducer = command.new_compute_command_encoder();
        reducer.set_compute_pipeline_state(&self.import_reduction);
        reducer.set_buffer(0, Some(&self.partials), 0);
        reducer.set_buffer(1, Some(&self.pool_distances), 0);
        reducer.set_bytes(2, size_of::<u32>() as u64, (&tiles as *const u32).cast());
        reducer.dispatch_thread_groups(thread_group(pairs as u64), thread_group(256));
        reducer.end_encoding();
        finish(command)?;

        let values = read::<f32>(&self.pool_distances, 2 * MAX_HISTORY + 1);
        self.map_import(&values)
    }

    fn map_import(&self, values: &[f32]) -> Result<ImportGeometry, String> {
        let center = values[2 * MAX_HISTORY];
        if !center.is_finite() || center < 0.0 {
            return Err("Resident import produced an invalid incumbent distance".into());
        }
        let base_index = self.identities[..self.history]
            .iter()
            .position(|&identity| identity == self.base_id);
        if self.implicit_history && base_index.is_none() {
            return Err("Resident incumbent is missing from logical history".into());
        }
        let mut distances = (0..self.history)
            .map(|row| {
                let estimate = base_index.map_or(0.0, |base| {
                    center + self.pairwise_distances[base * MAX_HISTORY + row]
                });
                (row, estimate)
            })
            .collect::<Vec<_>>();
        let mut matched = 0;
        for physical in 0..self.resident_history {
            let identity = self.resident_identities[physical];
            if let Some(logical) = self.identities[..self.history]
                .iter()
                .position(|&candidate| candidate == identity)
            {
                let distance = values[physical];
                if !distance.is_finite() || distance < 0.0 {
                    return Err("Resident import produced an invalid history distance".into());
                }
                distances[logical] = (logical, distance);
                matched += 1;
            }
        }
        if !self.implicit_history && matched != self.history {
            return Err("Explicit resident history has unmapped weight rows".into());
        }
        Ok(ImportGeometry { distances, center })
    }

    /// Retain a foreign evaluated row as local model evidence. Its reward and
    /// geometry enter history, while the local incumbent, proposal ledger, and
    /// trust-region success/failure counters remain owned by this region.
    pub(super) fn import_observation(
        &mut self,
        incoming: &Buffer,
        value: f32,
        variance: f32,
    ) -> Result<(), String> {
        self.check_idle()?;
        check_scores(&[value], &[variance])?;
        let observation =
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance);
        self.import_objectives(incoming, observation)
    }

    pub(super) fn import_objectives(
        &mut self,
        incoming: &Buffer,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<(), String> {
        let crate::objective_observation::ObjectiveEstimate {
            mean: value,
            variance,
        } = observation.control();
        self.objective_history.validate(observation)?;
        let identity = self
            .observation
            .checked_add(1)
            .ok_or("Observation ID overflow")?;
        let target = history::choose_slot(self)?;
        let geometry = self.measure_import(incoming)?;
        let prior_history = self.history;
        let slot = if self.resident_history == self.history_rows.len() {
            0
        } else {
            self.resident_history
        };
        let mut mutated = false;
        let result = (|| {
            // A retained history row may alias the incumbent. Replace that
            // allocation before writing the foreign weights into the archive.
            if std::ptr::eq::<metal::BufferRef>(&*self.history_rows[slot], &*self.base) {
                self.history_rows[slot] = self.runtime.buffer::<u16>(self.dimensions);
            }
            mutated = true;
            self.copy(incoming, &self.history_rows[slot])?;
            self.observation = identity;
            self.resident_identities[slot] = identity;
            if self.resident_history == self.history_rows.len() {
                self.history_rows.rotate_left(1);
                self.resident_identities.rotate_left(1);
            } else {
                self.resident_history += 1;
            }
            self.record_observation(
                identity,
                target,
                (value, variance),
                observation,
                Some(geometry.distances),
                false,
            )?;
            if self.implicit_history
                && prior_history < self.initial_observations
                && self.history == self.initial_observations
                && !self.has_morbo()
            {
                self.trust.restart();
                self.trust_observations = self.history;
                self.trust.seed_baseline(f64::from(self.best), self.history);
                self.length = self.trust.length();
            }
            Ok(())
        })();
        if mutated && result.is_err() {
            self.poisoned = true;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_geometry() {
        let base = [0x3f80_u16; 4];
        let blocks = vec![
            ParamBlock::new(1, 0, 2, 0.25, 1.0).unwrap(),
            ParamBlock::new(2, 2, 2, 0.25, 0.5).unwrap(),
        ];
        let state =
            SearchState::new(&base, 0.0, 0.0, blocks, 4, 1, TRLengthConfig::default()).unwrap();
        let incoming = Runtime::shared()
            .unwrap()
            .buffer_with(&[0x4000_u16, 0x4040, 0x4000, 0x4040]);
        let geometry = state.measure_import(&incoming).unwrap();
        assert!((geometry.center - 7.5).abs() < 1.0e-4);
        assert_eq!(geometry.distances.len(), 1);
        assert!((geometry.distances[0].1 - 7.5).abs() < 1.0e-4);
    }

    #[test]
    fn foreign_history() {
        let base = [0x3f80_u16; 4];
        let blocks = vec![ParamBlock::new(1, 0, 4, 0.25, 1.0).unwrap()];
        let mut state =
            SearchState::new(&base, 1.0, 0.0, blocks, 2, 1, TRLengthConfig::default()).unwrap();
        let incoming = Runtime::shared().unwrap().buffer_with(&[0x4000_u16; 4]);
        let controller = state.controller_info().unwrap();
        let incumbent = state.read_best().unwrap();
        state.import_observation(&incoming, 5.0, 0.0).unwrap();
        assert_eq!(state.history, 2);
        assert_eq!(state.outcomes[1], 5.0);
        assert_eq!(state.best().unwrap(), 1.0);
        assert_eq!(state.read_best().unwrap(), incumbent);
        assert_eq!(
            state.controller_info().unwrap().success_counter,
            controller.success_counter
        );
        assert_eq!(
            state.controller_info().unwrap().failure_counter,
            controller.failure_counter
        );
        state.import_observation(&incoming, 6.0, 0.0).unwrap();
        assert_eq!(state.history, 2);
        assert_eq!(state.outcomes[..2], [5.0, 6.0]);
        assert_eq!(state.read_best().unwrap(), incumbent);
    }
}
