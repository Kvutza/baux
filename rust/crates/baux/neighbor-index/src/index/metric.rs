use baux::neighbor_index::error::BpannError;
use baux::neighbor_index::index::build::BpannIndex;
use baux::neighbor_index::index::page::Page;
use baux::neighbor_index::index::sync::IncrementalIndex;
#[cfg(test)]
#[path = "metric_tests.rs"]
mod tests;

fn page_vectors(page: &Page) -> Vec<&[f32]> {
    match page {
        Page::Internal { centroids, .. } => centroids.iter().map(Vec::as_slice).collect(),
        Page::Leaf {
            vectors,
            stored_centroid,
            ..
        } => vectors
            .iter()
            .map(Vec::as_slice)
            .chain(stored_centroid.iter().map(Vec::as_slice))
            .collect(),
    }
}

fn scale_vector(vector: &mut [f32], factors: &[f64]) {
    for (v, &factor) in vector.iter_mut().zip(factors) {
        *v = (f64::from(*v) * factor) as f32;
    }
}

impl BpannIndex {
    pub fn rescale(mut self, factors: &[f64]) -> Self {
        let mut new_pages: Vec<Page> = self.pages.to_vec();
        for page in &mut new_pages {
            match page {
                Page::Internal { centroids, .. } => {
                    for vector in centroids {
                        scale_vector(vector, factors);
                    }
                }
                Page::Leaf {
                    vectors,
                    stored_centroid,
                    ..
                } => {
                    for vector in vectors {
                        scale_vector(vector, factors);
                    }
                    if let Some(vector) = stored_centroid {
                        scale_vector(vector, factors);
                    }
                }
            }
        }
        self.pages = new_pages.into();
        self
    }
}

impl IncrementalIndex {
    pub fn check_rescale(&self, factors: &[f64]) -> Result<(), BpannError> {
        let valid = factors.iter().all(|v| v.is_finite() && *v > 0.0)
            && self
                .indices
                .iter()
                .flat_map(|index| index.pages.iter())
                .flat_map(page_vectors)
                .all(|row| {
                    row.len() == factors.len()
                        && row
                            .iter()
                            .zip(factors)
                            .all(|(&v, &f)| ((v as f64 * f) as f32).is_finite())
                });
        if !valid {
            return Err(BpannError::InvalidParameter(
                "metric rescale exceeds FP32 index range".into(),
            ));
        }
        Ok(())
    }

    pub fn rescale(mut self, factors: &[f64]) -> Self {
        self.rescale_pending(factors);
        self.indices = self
            .indices
            .into_iter()
            .map(|index| index.rescale(factors))
            .collect();
        self
    }

    pub fn metric_rescale(self, factors: &[f64]) -> Result<Self, BpannError> {
        self.check_rescale(factors)?;
        Ok(self.rescale(factors))
    }
}
