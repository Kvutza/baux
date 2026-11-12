//! Batch half-float evaluation over JAX power-of-2 ShmSlab slots.

use super::TokamakWorker;
use baux::unified_mem::CMD_EVAL;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn batch_once(
    worker: &mut TokamakWorker,
    rows: &[&[u16]],
    low: &[f64],
    span: &[f64],
) -> Result<(Vec<f64>, u64), String> {
    let sess = worker.session.as_mut().ok_or("no active daemon session")?;
    if let Some(ref mut c) = sess.child
        && let Ok(Some(st)) = c.try_wait()
    {
        return Err(format!("daemon exited with {st}"));
    }
    let n = rows.len();
    let padded_n = if n > 1 {
        n.next_power_of_two().min(worker.spec.arms)
    } else {
        1
    };
    for (i, row) in rows.iter().enumerate() {
        unsafe { sess.slab.write_half(i, row.as_ptr(), low, span)? };
    }
    for slot in n..padded_n {
        unsafe { sess.slab.write_half(slot, rows[0].as_ptr(), low, span)? };
    }
    let timeout = worker.timeout;
    let (dt_us, _, _) = sess
        .slab
        .doorbell_wait(CMD_EVAL, padded_n, timeout, || Ok(()))?;
    let pen = worker.penalty;
    let scores = sess.slab.read_scores(padded_n)[..n]
        .iter()
        .map(|&s| if s.is_finite() && s > -1.0e8 { s } else { pen })
        .collect();
    let cost_us = dt_us / (n as u64).max(1);
    Ok((scores, cost_us))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn eval_halves(
    worker: &mut TokamakWorker,
    rows: &[&[u16]],
    low: &[f64],
    span: &[f64],
) -> Result<(Vec<f64>, u64), String> {
    if rows.is_empty() {
        return Ok((Vec::new(), 0));
    }
    if worker.evals_done >= worker.max_evals || worker.session.is_none() {
        worker.respawn()?;
    }
    if let Ok(res) = batch_once(worker, rows, low, span) {
        worker.evals_done += 1;
        return Ok(res);
    }
    worker.respawn()?;
    if let Ok(res) = batch_once(worker, rows, low, span) {
        worker.evals_done += 1;
        return Ok(res);
    }
    worker.respawn()?;
    Ok((vec![worker.penalty; rows.len()], 0))
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub fn eval_halves(
    _worker: &mut TokamakWorker,
    _rows: &[&[u16]],
    _low: &[f64],
    _span: &[f64],
) -> Result<(Vec<f64>, u64), String> {
    Err("eval_halves requires macOS aarch64".into())
}
