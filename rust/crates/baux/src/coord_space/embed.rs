use baux::semantics::categorical::{
    CatSpec, argmax_metric, log_decode, log_encode, log_width, periodic_decode, periodic_encode,
    rademacher, simplex_decode, simplex_encode, snap_code,
};

use super::Embedding;

/// Isometric `S^1` periodic circle embedding mapping `[low, high)` into `[0,
/// 1]^2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicEmbed {
    pub low: f64,
    pub high: f64,
}

impl Embedding for PeriodicEmbed {
    type Value = f64;
    fn latent_dim(&self) -> usize {
        2
    }
    fn encode(&self, value: &f64, out: &mut [f64]) {
        let enc = periodic_encode(&[*value], self.low, self.high).expect("periodic_encode");
        out[..2].copy_from_slice(&enc);
    }
    fn decode(&self, latent: &[f64], _weights: Option<&[f64]>) -> f64 {
        periodic_decode(&latent[..2], self.low, self.high).expect("periodic_decode")[0]
    }
}

/// Fisher-Rao Hellinger square-root simplex embedding on `\Delta^{K-1}` into
/// `S^{K-1}_+ \subset [0, 1]^K`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimplexEmbed<const K: usize>;

impl<const K: usize> Embedding for SimplexEmbed<K> {
    type Value = [f64; K];
    fn latent_dim(&self) -> usize {
        K
    }
    fn is_tied(&self) -> bool {
        true
    }
    fn encode(&self, value: &[f64; K], out: &mut [f64]) {
        let enc = simplex_encode(value, 1, K).expect("simplex_encode");
        out[..K].copy_from_slice(&enc);
    }
    fn decode(&self, latent: &[f64], _weights: Option<&[f64]>) -> [f64; K] {
        simplex_decode(&latent[..K], 1, K)
            .expect("simplex_decode")
            .try_into()
            .expect("simplex length K")
    }
}

/// Multiplicative Riemannian log-scale embedding on `[low, high] \subset (0,
/// \infty)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogEmbed {
    pub low: f64,
    pub high: f64,
}

impl Embedding for LogEmbed {
    type Value = f64;
    fn latent_dim(&self) -> usize {
        1
    }
    fn encode(&self, value: &f64, out: &mut [f64]) {
        out[0] = log_encode(&[*value], self.low, self.high).expect("log_encode")[0];
    }
    fn decode(&self, latent: &[f64], _weights: Option<&[f64]>) -> f64 {
        log_decode(&latent[..1], self.low, self.high).expect("log_decode")[0]
    }
}

/// Isometric Rademacher categorical embedding mapping `0..card` into `[0,
/// 1]^bits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Categorical {
    pub card: usize,
    pub bits: usize,
    pub seed: u64,
}

impl Categorical {
    pub fn new(card: usize, seed: u64) -> Self {
        let bits = log_width(card.max(1), 48).expect("log_width");
        Self { card, bits, seed }
    }
}

impl Embedding for Categorical {
    type Value = usize;
    fn latent_dim(&self) -> usize {
        self.bits
    }
    fn encode(&self, value: &usize, out: &mut [f64]) {
        let spec = CatSpec {
            variable: 0,
            width: self.bits,
            seed: self.seed,
            sep: (self.bits as f64).sqrt(),
        };
        let code = rademacher(&[*value as i64], spec).expect("rademacher");
        for (dst, &c) in out[..self.bits].iter_mut().zip(&code) {
            *dst = 0.5 * (c + 1.0);
        }
    }
    fn decode(&self, latent: &[f64], weights: Option<&[f64]>) -> usize {
        let centered: Vec<f64> = latent[..self.bits].iter().map(|&u| 2.0 * u - 1.0).collect();
        let domain: Vec<i64> = (0..self.card as i64).collect();
        let spec = CatSpec {
            variable: 0,
            width: self.bits,
            seed: self.seed,
            sep: (self.bits as f64).sqrt(),
        };
        if let Some(w) = weights {
            let codes = rademacher(&domain, spec).expect("rademacher");
            argmax_metric(&centered, &codes, &w[..self.bits], 1, self.card, self.bits)
                .expect("argmax_metric")[0] as usize
        } else {
            snap_code(&centered, 1, &domain, spec).expect("snap_code")[0] as usize
        }
    }
}

impl<A: Embedding, B: Embedding> Embedding for (A, B) {
    type Value = (A::Value, B::Value);
    fn latent_dim(&self) -> usize {
        self.0.latent_dim() + self.1.latent_dim()
    }
    fn is_tied(&self) -> bool {
        self.0.is_tied() || self.1.is_tied()
    }
    fn encode(&self, value: &Self::Value, out: &mut [f64]) {
        let split = self.0.latent_dim();
        self.0.encode(&value.0, &mut out[..split]);
        self.1
            .encode(&value.1, &mut out[split..split + self.1.latent_dim()]);
    }
    fn decode(&self, latent: &[f64], weights: Option<&[f64]>) -> Self::Value {
        let split = self.0.latent_dim();
        let end = split + self.1.latent_dim();
        let (wa, wb) = match weights {
            Some(w) => (Some(&w[..split]), Some(&w[split..end])),
            None => (None, None),
        };
        (
            self.0.decode(&latent[..split], wa),
            self.1.decode(&latent[split..end], wb),
        )
    }
}
