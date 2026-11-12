"""Statistical aggregation, paired initial-error check, and Holm-adjusted reporting."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np


def _stats(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0
    sem = float(np.std(arr, ddof=1) / math.sqrt(n))
    return mean, sem


def _ttest(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    if n < 2:
        return 0.0, 1.0
    mean, sem = _stats(diffs)
    if sem <= 1e-15:
        return (0.0, 1.0) if abs(mean) <= 1e-15 else (math.copysign(1e9, mean), 0.0)
    t_stat = mean / sem
    try:
        from scipy.stats import t as student_t

        p_val = float(2.0 * student_t.sf(abs(t_stat), df=n - 1))
    except (ImportError, ValueError, ZeroDivisionError):
        p_val = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    return float(t_stat), min(1.0, max(0.0, p_val))


def _holm(p_vals: list[float]) -> list[float]:
    m = len(p_vals)
    order = sorted(range(m), key=lambda i: p_vals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_vals[idx])
        adj[idx] = min(1.0, running)
    return adj


def _seeds(db_path: Path) -> dict[str, dict[str, dict[int, dict]]]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT experiment, config_id, rep, summary_json FROM jobs WHERE status='DONE'"
    ).fetchall()
    conn.close()
    by_exp: dict[str, dict[str, dict[int, dict]]] = {}
    for exp, cid, rep, raw in rows:
        if not raw:
            continue
        payload = json.loads(raw)
        sr_list = payload.get("seed_runs", [])
        if not sr_list:
            continue
        by_exp.setdefault(exp, {}).setdefault(cid, {})[int(rep)] = sr_list[0]
    return by_exp


def _comparisons(cfg_map: dict[str, dict[int, dict]]) -> list[dict]:
    ref_id = "turbo_zero_r10_ellipsoid_ball"
    if ref_id not in cfg_map:
        return []
    ref_seeds = cfg_map[ref_id]
    raw_rows: list[dict] = []
    p_list: list[float] = []
    for cid in sorted(cfg_map):
        if not cid.startswith("turbo_enn_r1_"):
            continue
        shared = sorted(set(ref_seeds) & set(cfg_map[cid]))
        if not shared:
            continue
        diffs = [
            float(ref_seeds[r]["final_tracking_error"])
            - float(cfg_map[cid][r]["final_tracking_error"])
            for r in shared
        ]
        m_diff, s_diff = _stats(diffs)
        t_stat, p_val = _ttest(diffs)
        p_list.append(p_val)
        raw_rows.append(
            {
                "enn_config": cid,
                "vs_reference": ref_id,
                "paired_seeds": len(shared),
                "error_advantage_mean": m_diff,
                "error_advantage_sem": s_diff,
                "t_stat": t_stat,
                "p_unadjusted": p_val,
            }
        )
    holm = _holm(p_list)
    for row, p_adj in zip(raw_rows, holm, strict=True):
        row["p_holm"] = p_adj
    return raw_rows


def _drift(cfg_map: dict[str, dict[int, dict]]) -> float:
    all_reps = sorted({r for m in cfg_map.values() for r in m})
    max_drift = 0.0
    for r in all_reps:
        inits = [
            float(m[r]["initial_tracking_error"]) for m in cfg_map.values() if r in m
        ]
        if len(inits) >= 2:
            max_drift = max(max_drift, max(inits) - min(inits))
    return max_drift


def _anytime(seeds: list[dict]) -> dict[str, dict[str, float]]:
    if not seeds or not seeds[0].get("best_scores"):
        return {}
    horizon = min(len(s.get("best_scores", [])) for s in seeds)
    marks = [r for r in (10, 100, 500, 1000, 2500, 5000, 10000) if r <= horizon]
    if horizon > 0 and horizon not in marks:
        marks.append(horizon)
    out: dict[str, dict[str, float]] = {}
    for r in marks:
        m, sem = _stats([float(s["best_scores"][r - 1]) for s in seeds])
        out[str(r)] = {"score_mean": m, "score_sem": sem}
    return out


def _summarize(exp: str, cfg_map: dict[str, dict[int, dict]]) -> dict:
    table: list[dict] = []
    for cid in sorted(cfg_map):
        seeds = [cfg_map[cid][r] for r in sorted(cfg_map[cid])]
        im, isem = _stats([float(s["initial_tracking_error"]) for s in seeds])
        sm, ssem = _stats([float(s["final_score"]) for s in seeds])
        fm, fsem = _stats([float(s["final_tracking_error"]) for s in seeds])
        rm, rsem = _stats([float(s["tracking_error_reduction_pct"]) for s in seeds])
        tm, tsem = _stats([float(s["elapsed_seconds"]) for s in seeds])
        table.append(
            {
                "config_id": cid,
                "n_seeds": len(seeds),
                "init_err_mean": im,
                "init_err_sem": isem,
                "final_score_mean": sm,
                "final_score_sem": ssem,
                "final_err_mean": fm,
                "final_err_sem": fsem,
                "reduction_pct_mean": rm,
                "reduction_pct_sem": rsem,
                "elapsed_s_mean": tm,
                "elapsed_s_sem": tsem,
                "anytime": _anytime(seeds),
            }
        )
    table.sort(key=lambda r: r["final_err_mean"])
    return {
        "experiment": exp,
        "max_paired_init_drift": _drift(cfg_map),
        "configs": table,
        "paired_tests_vs_zero_r10_ellipsoid": _comparisons(cfg_map),
    }


def _markdown(summaries: list[dict]) -> str:
    lines = ["# Suite B Benchmark Report: TuRBO-zero (R=10) vs TuRBO-ENN (R=1)\n"]
    for block in summaries:
        exp = block["experiment"]
        drift = block["max_paired_init_drift"]
        lines.append(f"## `{exp}` (max paired init drift: `{drift:.2e}`)\n")
        lines.append(
            "| Config | Seeds | Init Error | Final Error | Reduction % | Wall Time (s) |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
        for r in block["configs"]:
            lines.append(
                f"| `{r['config_id']}` | {r['n_seeds']} | "
                f"{r['init_err_mean']:.3f} +/- {r['init_err_sem']:.3f} | "
                f"**{r['final_err_mean']:.3f} +/- {r['final_err_sem']:.3f}** | "
                f"{r['reduction_pct_mean']:.2f}% +/- {r['reduction_pct_sem']:.2f}% | "
                f"{r['elapsed_s_mean']:.2f} +/- {r['elapsed_s_sem']:.2f} |"
            )
        lines.append("")
        if block["paired_tests_vs_zero_r10_ellipsoid"]:
            lines.append("### Paired Tests vs `turbo_zero_r10_ellipsoid_ball`\n")
            lines.append(
                "| ENN (R=1) Geometry | Paired n | Error Advantage (Zero - ENN) | t-stat | p (raw) | p (Holm) |"
            )
            lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
            for t in block["paired_tests_vs_zero_r10_ellipsoid"]:
                lines.append(
                    f"| `{t['enn_config']}` | {t['paired_seeds']} | "
                    f"{t['error_advantage_mean']:+.3f} +/- {t['error_advantage_sem']:.3f} | "
                    f"{t['t_stat']:+.3f} | {t['p_unadjusted']:.4g} | {t['p_holm']:.4g} |"
                )
            lines.append("")
    return "\n".join(lines)


def build_report(campaign_dir: Path) -> str:
    """Aggregate SQLite campaign results and write report.json + report.md."""
    db_path = campaign_dir / "suite_b.sqlite3"
    by_exp = _seeds(db_path)
    summaries = [_summarize(e, by_exp[e]) for e in sorted(by_exp)]
    (campaign_dir / "report.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    md = _markdown(summaries)
    (campaign_dir / "report.md").write_text(md, encoding="utf-8")
    return md
