//! Small FP64 parity oracle. Never used to update model weights.
use super::{POLAR, Shape, Step};

pub fn half(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let mantissa = bits & 1023;
    if exponent == 0 {
        sign * f64::from(mantissa) * 2.0f64.powi(-24)
    } else if exponent == 31 {
        if mantissa == 0 {
            sign * f64::INFINITY
        } else {
            f64::NAN
        }
    } else {
        sign * f64::from(1024 + mantissa) * 2.0f64.powi(i32::from(exponent) - 25)
    }
}

pub fn fixture(shape: Shape, round: usize) -> Vec<f32> {
    let size = shape.rows as usize * shape.cols as usize;
    (0..shape.elements())
        .map(|i| {
            // An exactly zero bank checks isolation from the other bank's norm.
            if i / size == 0 {
                return 0.0;
            }
            let mut v = (i as u64).wrapping_add((round as u64 + 1) << 32);
            v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
            ((v >> 40) as f32 / 16777216.0 - 0.5) * 0.125
        })
        .collect()
}

fn product(a: &[f64], b: &[f64], m: usize, n: usize, k: usize, transpose: bool) -> Vec<f64> {
    let mut output = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for t in 0..k {
                output[i * n + j] +=
                    a[i * k + t] * b[if transpose { j * k + t } else { t * n + j }];
            }
        }
    }
    output
}

pub fn muon(shape: Shape, master: &mut [f64], momentum: &mut [f64], gradient: &[f32], step: Step) {
    let size = shape.rows as usize * shape.cols as usize;
    let wide = shape.wide();
    let (m, n) = (wide.rows as usize, wide.cols as usize);
    let beta = f64::from(step.momentum);
    for batch in 0..shape.batch as usize {
        let mut x = vec![0.0; size];
        for (j, value) in x.iter_mut().enumerate() {
            let source = if shape.rows > shape.cols {
                (j % n) * m + j / n
            } else {
                j
            };
            let i = batch * size + source;
            let g = f64::from(gradient[i]);
            momentum[i] = beta * momentum[i] + (1.0 - beta) * g;
            *value = (1.0 - beta) * g + beta * momentum[i];
        }
        let norm =
            (1.01 * x.iter().map(|v| v * v).sum::<f64>().sqrt()).max(f64::from(step.epsilon));
        for value in &mut x {
            *value /= norm;
        }
        for c in POLAR {
            let a = product(&x, &x, m, m, n, true);
            let square = product(&a, &a, m, m, m, false);
            let b = a
                .iter()
                .zip(square)
                .map(|(a, square)| f64::from(c.b) * a + f64::from(c.c) * square)
                .collect::<Vec<_>>();
            let bx = product(&b, &x, m, n, m, false);
            for (x, bx) in x.iter_mut().zip(bx) {
                *x = f64::from(c.a) * *x + bx;
            }
        }
        let rate = f64::from(step.rate);
        let scale = 0.2 * (shape.rows.max(shape.cols) as f64).sqrt() * f64::from(step.scale);
        for j in 0..size {
            let source = if shape.rows > shape.cols {
                (j % m) * n + j / m
            } else {
                j
            };
            let i = batch * size + j;
            master[i] = (1.0 - rate * f64::from(step.decay)) * master[i] - rate * scale * x[source];
        }
    }
}

pub fn adam(
    master: &mut [f64],
    first: &mut [f64],
    second: &mut [f64],
    gradient: &[f32],
    step: Step,
) {
    for i in 0..master.len() {
        let g = f64::from(gradient[i]);
        first[i] = f64::from(step.momentum) * first[i] + (1.0 - f64::from(step.momentum)) * g;
        second[i] = f64::from(step.beta2) * second[i] + (1.0 - f64::from(step.beta2)) * g * g;
        let update = (first[i] / f64::from(step.bias1))
            / ((second[i] / f64::from(step.bias2)).sqrt() + f64::from(step.epsilon));
        master[i] = (1.0 - f64::from(step.rate) * f64::from(step.decay)) * master[i]
            - f64::from(step.rate) * update;
    }
}

pub fn error(actual: &[u16], expected: &[f64]) -> Result<f64, String> {
    if actual.len() != expected.len() {
        return Err("optimizer parity extent mismatch".into());
    }
    let mut error = 0.0f64;
    for (&a, &e) in actual.iter().zip(expected) {
        let a = half(a);
        if !a.is_finite() || !e.is_finite() {
            return Err("optimizer parity produced a non-finite weight".into());
        }
        error = error.max((a - e).abs());
    }
    if error > 0.001 {
        return Err(format!("optimizer FP64 parity error {error} exceeds 0.001"));
    }
    Ok(error)
}
