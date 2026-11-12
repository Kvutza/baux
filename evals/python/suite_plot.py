"""Export lossless per-round, per-seed Suite B trajectories for CairoMakie figures."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import click
import numpy as np


def _matrix(db_path: Path, experiment: str) -> dict[str, dict[int, list[float]]]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT config_id, rep, summary_json FROM jobs "
        "WHERE status='DONE' AND experiment=? ORDER BY config_id, rep",
        (experiment,),
    ).fetchall()
    conn.close()
    by_cfg: dict[str, dict[int, list[float]]] = {}
    for cid, rep, raw in rows:
        if not raw:
            continue
        traces = json.loads(raw).get("traces", [])
        by_cfg.setdefault(cid, {})[int(rep)] = [
            float(t["tracking_error"]) for t in traces
        ]
    return by_cfg


def _entry(
    mat: np.ndarray, grand: np.ndarray, grand_mean: np.ndarray, ref_final: np.ndarray
) -> dict:
    n = mat.shape[0]
    t_crit = 2.262157162854099 if n == 10 else 1.96
    inv_sqrt_n = 1.0 / math.sqrt(n)
    normed = mat - grand + grand_mean[None, :]
    diffs = mat[:, -1] - ref_final
    d_sem = float(np.std(diffs, ddof=1) * inv_sqrt_n)
    return {
        "n_seeds": n,
        "mean": np.mean(mat, axis=0).tolist(),
        "sem": (np.std(mat, axis=0, ddof=1) * inv_sqrt_n).tolist(),
        "paired_sem": (np.std(normed, axis=0, ddof=1) * inv_sqrt_n).tolist(),
        "final_seeds": mat[:, -1].tolist(),
        "diff_vs_ref_seeds": diffs.tolist(),
        "diff_mean": float(np.mean(diffs)),
        "diff_sem": d_sem,
        "diff_ci95": t_crit * d_sem,
        "wins_vs_ref": int(np.sum(diffs < -1e-9)),
        "ties_vs_ref": int(np.sum(np.abs(diffs) <= 1e-9)),
        "losses_vs_ref": int(np.sum(diffs > 1e-9)),
    }


def _paired(
    by_cfg: dict[str, dict[int, list[float]]], ref_id: str, common_reps: list[int]
) -> dict[str, dict]:
    mats = {
        cid: np.asarray([m[r] for r in common_reps], dtype=np.float64)
        for cid, m in sorted(by_cfg.items())
    }
    grand = np.mean(list(mats.values()), axis=0)
    grand_mean = np.mean(grand, axis=0)
    ref_final = mats[ref_id][:, -1]
    return {cid: _entry(mat, grand, grand_mean, ref_final) for cid, mat in mats.items()}


def _meta(campaign_dir: Path, experiment: str) -> tuple[float, dict]:
    report_list = json.loads((campaign_dir / "report.json").read_text())
    block = next(b for b in report_list if b["experiment"] == experiment)
    p_map = {
        r["enn_config"]: {
            "p_raw": r["p_unadjusted"],
            "p_holm": r["p_holm"],
            "t_stat": r["t_stat"],
        }
        for r in block.get("paired_tests_vs_zero_r10_ellipsoid", [])
    }
    return float(block["max_paired_init_drift"]), p_map


def export_payload(campaign_dir: Path, experiment: str, out_path: Path) -> None:
    """Write paired per-round mean, SEM, and seed-level summary JSON for plotting."""
    by_cfg = _matrix(campaign_dir / "suite_b.sqlite3", experiment)
    rep_sets = [set(m.keys()) for m in by_cfg.values() if m]
    common = sorted(set.intersection(*rep_sets)) if rep_sets else []
    if len(common) < 2:
        raise RuntimeError(f"Need >=2 complete paired seeds, found {len(common)}")
    drift, p_map = _meta(campaign_dir, experiment)
    summary = _paired(by_cfg, "turbo_zero_r10_ellipsoid_ball", common)
    default_p = {"p_raw": 1.0, "p_holm": 1.0, "t_stat": 0.0}
    for cid, entry in summary.items():
        entry.update(p_map.get(cid, default_p))
    payload = {
        "experiment": experiment,
        "n_paired_seeds": len(common),
        "rounds": len(next(iter(summary.values()))["mean"]),
        "max_paired_init_drift": drift,
        "configs": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))


def _callback(campaign_dir: str, experiment: str, out: str) -> None:
    export_payload(Path(campaign_dir).resolve(), experiment, Path(out).resolve())


main = click.Command(
    name="suite-plot",
    callback=_callback,
    params=[
        click.Option(["--campaign-dir"], required=True, type=str),
        click.Option(["--experiment"], required=True, type=str),
        click.Option(["--out"], required=True, type=str),
    ],
)

if __name__ == "__main__":
    main()
