//! Isometric Rademacher and one-hot categorical embeddings with max-dot and
//! Riemannian Mahalanobis snapping, plus non-Cartesian manifold charts
//! (periodic circle `S^1`, Fisher-Rao probability simplex `Delta^{d-1}`, and
//! multiplicative log-scale `(0, inf)`).
//!
//! References:
//! - Johnson and Lindenstrauss (1984), *Extensions of Lipschitz Mappings into a
//!   Hilbert Space*, Contemp. Math. 26, 189-206.
//! - Achlioptas (2003), *Database-Friendly Random Projections:
//!   Johnson-Lindenstrauss with Binary Coins*, JCSS 66(4), 671-687.
//! - Bhattacharyya (1943), *On a Measure of Divergence between Two Statistical
//!   Populations Defined by Their Probability Distributions*, Bull. Calcutta
//!   Math. Soc. 35, 99-109.
//! - Cencov (1982), *Statistical Decision Rules and Optimal Inference*,
//!   Translations of Mathematical Monographs 53, AMS.

/// Categorical embedding family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CatEncoding {
    #[default]
    Auto,
    OneHot,
    Rademacher,
}

/// Rademacher codebook specification for a single categorical variable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CatSpec {
    pub variable: usize,
    pub width: usize,
    pub seed: u64,
    pub sep: f64,
}

#[inline(always)]
fn code_hash(category: i64, variable: usize, seed: u64) -> u64 {
    let mut x = (category as u64) ^ seed ^ (variable as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Logarithmic Rademacher code width `min(ceil(2 * log2(card) + 7), cap)`.
pub fn log_width(card: usize, cap: usize) -> Result<usize, String> {
    if cap == 0 || cap > 64 {
        return Err("dim_cap must be in 1..=64".to_string());
    }
    if card == 0 {
        return Err("card must be >= 1".to_string());
    }
    if card == 1 {
        return Ok(1);
    }
    let w = (2.0 * (card as f64).log2() + 7.0).ceil() as usize;
    Ok(w.min(cap))
}

/// Encode `categories` into a row-major `(categories.len(), spec.width)`
/// Rademacher matrix with entries `+- sep / sqrt(width)`.
pub fn rademacher(categories: &[i64], spec: CatSpec) -> Result<Vec<f64>, String> {
    if spec.width == 0 || spec.width > 64 {
        return Err("width must be in 1..=64".to_string());
    }
    if !spec.sep.is_finite() || spec.sep <= 0.0 {
        return Err("separation must be positive and finite".to_string());
    }
    let scale = spec.sep / (spec.width as f64).sqrt();
    let mut out = vec![0.0; categories.len() * spec.width];
    for (i, &cat) in categories.iter().enumerate() {
        let bits = code_hash(cat, spec.variable, spec.seed);
        for j in 0..spec.width {
            out[i * spec.width + j] = if ((bits >> j) & 1) == 1 {
                scale
            } else {
                -scale
            };
        }
    }
    Ok(out)
}

/// Encode `categories` in `0..card` into a row-major `(categories.len(), card)`
/// one-hot matrix.
pub fn one_hot(categories: &[i64], card: usize) -> Result<Vec<f64>, String> {
    if card == 0 || categories.iter().any(|&v| v < 0 || (v as usize) >= card) {
        return Err("one-hot ids must be in 0..card".to_string());
    }
    let mut out = vec![0.0; categories.len() * card];
    for (i, &id) in categories.iter().enumerate() {
        out[i * card + id as usize] = 1.0;
    }
    Ok(out)
}

/// Feature-hash a row-major `(rows, cols)` integer matrix into `(rows, dim)`.
pub fn hash_cols(
    categories: &[i64],
    rows: usize,
    cols: usize,
    dim: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    if dim == 0 {
        return Err("dim must be >= 1".to_string());
    }
    if categories.len() != rows * cols {
        return Err("categories length mismatch".to_string());
    }
    let mut out = vec![0.0; rows * dim];
    let scale = 1.0 / (cols.max(1) as f64).sqrt();
    for r in 0..rows {
        for c in 0..cols {
            let h = code_hash(categories[r * cols + c], c, seed);
            let slot = (h % dim as u64) as usize;
            out[r * dim + slot] += if (h >> 63) == 1 { scale } else { -scale };
        }
    }
    Ok(out)
}

fn argmax_dot(block: &[f64], rows: usize, width: usize, table: &[f64], domain: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &block[r * width..(r + 1) * width];
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (j, _) in domain.iter().enumerate() {
            let code = &table[j * width..(j + 1) * width];
            let score: f64 = row.iter().zip(code.iter()).map(|(a, b)| a * b).sum();
            if score > best_score {
                best_score = score;
                best = j;
            }
        }
        out.push(domain[best]);
    }
    out
}

/// Snap a row-major `(rows, spec.width)` continuous block back to the nearest
/// Rademacher codeword in `domain`.
pub fn snap_code(
    block: &[f64],
    rows: usize,
    domain: &[i64],
    spec: CatSpec,
) -> Result<Vec<i64>, String> {
    if domain.is_empty() {
        return Err("domain must be non-empty".to_string());
    }
    if block.len() != rows * spec.width {
        return Err("block length mismatch".to_string());
    }
    let table = rademacher(domain, spec)?;
    Ok(argmax_dot(block, rows, spec.width, &table, domain))
}

/// Snap a row-major `(rows, domain.len())` continuous block back to the
/// nearest one-hot category in `domain`.
pub fn snap_onehot(block: &[f64], rows: usize, domain: &[i64]) -> Result<Vec<i64>, String> {
    if domain.is_empty() {
        return Err("domain must be non-empty".to_string());
    }
    let card = domain.len();
    if block.len() != rows * card {
        return Err("block length mismatch".to_string());
    }
    let ids: Vec<i64> = (0..card as i64).collect();
    let table = one_hot(&ids, card)?;
    Ok(argmax_dot(block, rows, card, &table, domain))
}

/// Riemannian Mahalanobis snapping `argmax_c sum_j g_j v_{r, j} e_{c, j}` for
/// positive finite metric weights `g_j`.
pub fn argmax_metric(
    continuous: &[f64],
    codes: &[f64],
    weights: &[f64],
    rows: usize,
    card: usize,
    width: usize,
) -> Result<Vec<i64>, String> {
    if card == 0 || width == 0 {
        return Err("card and width must be >= 1".to_string());
    }
    if continuous.len() != rows * width || codes.len() != card * width || weights.len() != width {
        return Err("argmax_metric shape mismatch".to_string());
    }
    if continuous
        .iter()
        .chain(codes.iter())
        .any(|v| !v.is_finite())
        || weights.iter().any(|&g| !g.is_finite() || g <= 0.0)
    {
        return Err("inputs must be finite and weights positive".to_string());
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &continuous[r * width..(r + 1) * width];
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for c in 0..card {
            let code = &codes[c * width..(c + 1) * width];
            let score = super::neighbor::simd_dot(row, code, weights);
            if score > best_score {
                best_score = score;
                best = c;
            }
        }
        out.push(best as i64);
    }
    Ok(out)
}

/// Isometric `S^1` embedding mapping `theta in [low, high)` to
/// `(1/2 + cos(2 pi u) / (2 pi), 1/2 + sin(2 pi u) / (2 pi)) in [0, 1]^2`.
#[core::contracts::ensures(|ret: &Result<Vec<f64>, String>| match ret { Ok(uv) => uv.iter().all(|&v| (0.0..=1.0).contains(&v)), Err(_) => true })]
pub fn periodic_encode(angles: &[f64], low: f64, high: f64) -> Result<Vec<f64>, String> {
    if !low.is_finite() || !high.is_finite() || high <= low || angles.iter().any(|a| !a.is_finite())
    {
        return Err("invalid periodic bounds or non-finite angles".to_string());
    }
    let period = high - low;
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut out = Vec::with_capacity(angles.len() * 2);
    for &theta in angles {
        let phi = two_pi * ((theta - low) / period).rem_euclid(1.0);
        out.push(0.5 + phi.cos() / two_pi);
        out.push(0.5 + phi.sin() / two_pi);
    }
    Ok(out)
}

/// Retract 2D periodic coordinates `(cx, cy) in [0, 1]^2` back to `[low, high)`
/// via `atan2`.
pub fn periodic_decode(coords: &[f64], low: f64, high: f64) -> Result<Vec<f64>, String> {
    if !low.is_finite() || !high.is_finite() || high <= low {
        return Err("invalid periodic bounds".to_string());
    }
    if !coords.len().is_multiple_of(2) || coords.iter().any(|c| !c.is_finite()) {
        return Err("periodic coords must be finite pairs".to_string());
    }
    let period = high - low;
    let two_pi = 2.0 * std::f64::consts::PI;
    Ok(coords
        .chunks_exact(2)
        .map(|pair| {
            let phi = (pair[1] - 0.5).atan2(pair[0] - 0.5).rem_euclid(two_pi);
            low + (phi / two_pi) * period
        })
        .collect())
}

/// Fisher-Rao Hellinger square-root map `psi_i(p) = sqrt(p_i)` onto the
/// positive orthant of the unit sphere `S^{dim - 1}_+`.
pub fn simplex_encode(probs: &[f64], rows: usize, dim: usize) -> Result<Vec<f64>, String> {
    if dim == 0 || probs.len() != rows * dim {
        return Err("simplex_encode shape mismatch".to_string());
    }
    let mut out = Vec::with_capacity(rows * dim);
    for r in 0..rows {
        let row = &probs[r * dim..(r + 1) * dim];
        if row.iter().any(|&p| !p.is_finite() || p < 0.0) {
            return Err("probabilities must be nonnegative and finite".to_string());
        }
        let sum: f64 = row.iter().sum();
        if sum <= 1.0e-12 {
            return Err("probability row sum must be positive".to_string());
        }
        out.extend(row.iter().map(|&p| (p / sum).sqrt()));
    }
    Ok(out)
}

/// Squared retraction from sphere coordinates back onto the probability simplex
/// `Delta^{dim - 1}`.
pub fn simplex_decode(coords: &[f64], rows: usize, dim: usize) -> Result<Vec<f64>, String> {
    if dim == 0 || coords.len() != rows * dim || coords.iter().any(|c| !c.is_finite()) {
        return Err("simplex_decode shape mismatch or non-finite coords".to_string());
    }
    let mut out = Vec::with_capacity(rows * dim);
    for r in 0..rows {
        let row = &coords[r * dim..(r + 1) * dim];
        let sq: Vec<f64> = row.iter().map(|&z| z.max(0.0).powi(2)).collect();
        let sum: f64 = sq.iter().sum();
        if sum <= 1.0e-24 {
            out.extend(std::iter::repeat_n(1.0 / dim as f64, dim));
        } else {
            out.extend(sq.into_iter().map(|w| w / sum));
        }
    }
    Ok(out)
}

/// Multiplicative Riemannian chart mapping `x in [low, high] subset (0, inf)`
/// to `[0, 1]`.
pub fn log_encode(values: &[f64], low: f64, high: f64) -> Result<Vec<f64>, String> {
    if !low.is_finite() || !high.is_finite() || low <= 0.0 || high <= low {
        return Err("log bounds must satisfy 0 < low < high".to_string());
    }
    if values.iter().any(|&v| !v.is_finite() || v <= 0.0) {
        return Err("log values must be positive and finite".to_string());
    }
    let log_low = low.ln();
    let span = (high / low).ln();
    Ok(values.iter().map(|&v| (v.ln() - log_low) / span).collect())
}

/// Inverse multiplicative Riemannian chart mapping `u in [0, 1]` back to
/// `[low, high] subset (0, inf)`.
pub fn log_decode(unit: &[f64], low: f64, high: f64) -> Result<Vec<f64>, String> {
    if !low.is_finite() || !high.is_finite() || low <= 0.0 || high <= low {
        return Err("log bounds must satisfy 0 < low < high".to_string());
    }
    if unit.iter().any(|u| !u.is_finite()) {
        return Err("unit coordinates must be finite".to_string());
    }
    let span = (high / low).ln();
    Ok(unit
        .iter()
        .map(|&u| low * (u.clamp(0.0, 1.0) * span).exp())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rademacher_roundtrip() {
        let domain = [-4_i64, 2, 7];
        let spec = CatSpec {
            variable: 2,
            width: 16,
            seed: 42,
            sep: 10.0,
        };
        let code = rademacher(&domain, spec).unwrap();
        let snapped = snap_code(&code, 3, &domain, spec).unwrap();
        assert_eq!(snapped, domain);
        assert_eq!(log_width(100, 48).unwrap(), 21);
        assert_eq!(log_width(1, 48).unwrap(), 1);
        assert_eq!(log_width(1000, 12).unwrap(), 12);
        assert_eq!(CatEncoding::default(), CatEncoding::Auto);
    }

    #[test]
    fn noisy_snap() {
        let domain: Vec<i64> = (0..12).map(|v| v * 3 - 5).collect();
        let width = log_width(domain.len(), 48).unwrap();
        let spec = CatSpec {
            variable: 0,
            width,
            seed: 7,
            sep: 10.0,
        };
        let mut code = rademacher(&domain, spec).unwrap();
        for (idx, val) in code.iter_mut().enumerate() {
            let delta = if idx % 2 == 0 { 0.35 } else { -0.35 };
            *val += delta;
        }
        let snapped = snap_code(&code, domain.len(), &domain, spec).unwrap();
        assert_eq!(snapped, domain);
    }

    #[test]
    fn onehot_hash() {
        let domain = [10_i64, 20, 30];
        let oh = one_hot(&[0, 2, 1], 3).unwrap();
        assert_eq!(oh, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let noisy = [0.8, 0.2, -0.1, 0.1, 0.2, 0.9, -0.2, 0.7, 0.1];
        let snapped = snap_onehot(&noisy, 3, &domain).unwrap();
        assert_eq!(snapped, vec![10, 30, 20]);
        let hashed = hash_cols(&[1, 2, 3, 4, 1, 2], 3, 2, 8, 99).unwrap();
        assert_eq!(hashed.len(), 24);
        assert_eq!(&hashed[0..8], &hashed[16..24]);
    }

    #[test]
    fn categorical_errors() {
        assert!(log_width(0, 16).is_err());
        assert!(log_width(4, 0).is_err());
        assert!(log_width(4, 65).is_err());
        let bad_w = CatSpec {
            variable: 0,
            width: 0,
            seed: 0,
            sep: 1.0,
        };
        assert!(rademacher(&[0], bad_w).is_err());
        let bad_sep = CatSpec {
            variable: 0,
            width: 8,
            seed: 0,
            sep: 0.0,
        };
        assert!(rademacher(&[0], bad_sep).is_err());
        assert!(one_hot(&[3], 3).is_err());
        assert!(one_hot(&[-1], 3).is_err());
        assert!(hash_cols(&[1, 2], 1, 2, 0, 0).is_err());
        assert!(hash_cols(&[1], 1, 2, 4, 0).is_err());
        assert!(snap_onehot(&[1.0], 1, &[]).is_err());
        assert!(snap_onehot(&[1.0], 1, &[1, 2]).is_err());
    }

    #[test]
    fn metric_charts() {
        let codes = [1.0, 1.0, -1.0, 1.0];
        let cont = [0.5, 0.9];
        assert_eq!(
            argmax_metric(&cont, &codes, &[1.0, 1.0], 1, 2, 2).unwrap(),
            vec![0]
        );
        assert_eq!(
            argmax_metric(&[0.5, -0.9], &codes, &[1.0, 10.0], 1, 2, 2).unwrap(),
            vec![0]
        );
        assert_eq!(
            argmax_metric(&[-0.5, 0.9], &codes, &[10.0, 1.0], 1, 2, 2).unwrap(),
            vec![1]
        );

        let angles = [0.0, 1.5, 4.5];
        let enc = periodic_encode(&angles, 0.0, 6.0).unwrap();
        let dec = periodic_decode(&enc, 0.0, 6.0).unwrap();
        for (a, b) in angles.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 1.0e-12);
        }

        let probs = [0.25, 0.25, 0.5];
        let sp = simplex_encode(&probs, 1, 3).unwrap();
        let norm2: f64 = sp.iter().map(|v| v * v).sum();
        assert!((norm2 - 1.0).abs() < 1.0e-12);
        let p_back = simplex_decode(&sp, 1, 3).unwrap();
        for (a, b) in probs.iter().zip(p_back.iter()) {
            assert!((a - b).abs() < 1.0e-12);
        }

        let vals = [1.0e-4, 1.0e-2, 1.0];
        let u = log_encode(&vals, 1.0e-4, 1.0).unwrap();
        assert!(
            (u[0] - 0.0).abs() < 1.0e-12
                && (u[1] - 0.5).abs() < 1.0e-12
                && (u[2] - 1.0).abs() < 1.0e-12
        );
        let v_back = log_decode(&u, 1.0e-4, 1.0).unwrap();
        for (a, b) in vals.iter().zip(v_back.iter()) {
            assert!((a - b).abs() < 1.0e-12);
        }
    }
}
