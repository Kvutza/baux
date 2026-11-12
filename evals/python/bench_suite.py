"""Crash-proof SQLite-WAL campaign runner for Suite B (TuRBO-zero R=10 vs TuRBO-ENN R=1)."""

from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import click

from evals.python.suite_report import build_report
from evals.python.suite_specs import SIM_TOKENS, write_specs

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    experiment TEXT NOT NULL,
    config_id TEXT NOT NULL,
    toml_path TEXT NOT NULL,
    ckpt_dir TEXT NOT NULL,
    rounds INTEGER NOT NULL,
    rep INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    strikes INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    evidence_dir TEXT DEFAULT '',
    summary_json TEXT DEFAULT '',
    last_error TEXT DEFAULT '',
    updated_at REAL NOT NULL
);
"""


class JobOutcome(NamedTuple):
    """Structured result returned by a single isolated tune worker."""

    job_id: str
    ok: bool
    advanced: bool
    evidence_dir: str
    summary_json: str
    error: str


class PoolCtx(NamedTuple):
    """Immutable execution parameters shared by pool dispatch helpers."""

    runner_bin: str
    repo_root: Path
    max_tokens: int
    ckpt_every: int


def _db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(SCHEMA_SQL)
    return conn


def _platforms() -> list[str]:
    if sys.platform == "darwin":
        return ["--target-platforms", "//:release-macos-arm64-platform"]
    return []


def _runner(repo_root: Path) -> str:
    buck = repo_root / "admin/tools/buck2w"
    cmd = [
        str(buck),
        "build",
        "--console",
        "simple",
        "--show-full-output",
        *_platforms(),
        "//rust/crates/runner:baux-runner",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if "rust/crates/runner:baux-runner" in line:
            cand = line.split()[-1].strip()
            full = Path(cand) if Path(cand).is_absolute() else (repo_root / cand)
            if full.exists():
                return str(full)
    raise RuntimeError(f"Could not resolve baux-runner binary:\n{proc.stdout}")


def _stamp(ckpt_dir: str, rep: int) -> float:
    live = Path(ckpt_dir) / f"seed_{rep}_live.cbor"
    done = Path(ckpt_dir) / f"seed_{rep}_done.cbor"
    t1 = live.stat().st_mtime if live.exists() else 0.0
    t2 = done.stat().st_mtime if done.exists() else 0.0
    return max(t1, t2)


def _evidence(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        if "Evidence:" in line:
            p = Path(line.split("Evidence:", 1)[1].strip())
            if p.exists():
                return p
    return None


def _execute(ctx: PoolCtx, row: tuple, slot_id: int) -> JobOutcome:
    job_id, _exp, _cid, toml_path, ckpt_dir, rounds, rep, _tok = row
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    before = _stamp(ckpt_dir, int(rep))
    env = os.environ.copy()
    env.update(
        {
            "BAUX_TUNE_REP": str(rep),
            "BAUX_TUNE_CKPT_DIR": str(ckpt_dir),
            "BAUX_CKPT_EVERY": str(ctx.ckpt_every),
            "BAUX_ISOLATE_DAEMON": "1",
            "BAUX_DAEMON_SLOT": str(slot_id),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    proc = subprocess.run(
        [ctx.runner_bin, "tune", str(toml_path), "--rounds", str(rounds)],
        cwd=str(ctx.repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    advanced = _stamp(ckpt_dir, int(rep)) > before
    if proc.returncode == 0:
        ev = _evidence(proc.stdout)
        if ev is not None and (ev / "summary.json").exists():
            summary = (ev / "summary.json").read_text(encoding="utf-8")
            return JobOutcome(job_id, True, True, str(ev), summary, "")
        return JobOutcome(job_id, False, advanced, "", "", "missing summary.json")
    tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-15:])
    return JobOutcome(job_id, False, advanced, "", "", f"rc={proc.returncode}: {tail}")


def _caffeinate() -> subprocess.Popen | None:
    if sys.platform != "darwin":
        return None
    cmd = ["caffeinate", "-dims", "-w", str(os.getpid())]
    try:
        return subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except OSError:
        return None


def _ping(sock_path: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(1.5)
        s.connect(str(sock_path))
        s.sendall(b"SHUTDOWN\n")
        s.recv(32)


def _shutdown() -> None:
    tdir = Path(tempfile.gettempdir())
    for sock_path in tdir.glob("baux_*_slot*.sock"):
        try:
            _ping(sock_path)
        except OSError:
            pass
    for pid_path in tdir.glob("baux_*_slot*.pid"):
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(pid, 9)
        except (OSError, ValueError):
            pass
        pid_path.unlink(missing_ok=True)


def _record(conn: sqlite3.Connection, res: JobOutcome) -> None:
    now = time.time()
    if res.ok:
        conn.execute(
            "UPDATE jobs SET status='DONE', evidence_dir=?, summary_json=?, last_error='', updated_at=? WHERE job_id=?",
            (res.evidence_dir, res.summary_json, now, res.job_id),
        )
        conn.commit()
        click.echo(f"[done] {res.job_id} -> {res.evidence_dir}")
        return
    cur = conn.execute(
        "SELECT strikes FROM jobs WHERE job_id=?", (res.job_id,)
    ).fetchone()
    strikes = 0 if res.advanced else (int(cur[0]) + 1 if cur else 1)
    status = "QUARANTINED" if strikes >= 3 else "PENDING"
    conn.execute(
        "UPDATE jobs SET status=?, strikes=?, last_error=?, updated_at=? WHERE job_id=?",
        (status, strikes, res.error, now, res.job_id),
    )
    conn.commit()
    msg = res.error.splitlines()[-1] if res.error else ""
    click.echo(f"[{status.lower()}] {res.job_id} (strikes={strikes}): {msg}")


def _drain(
    active: dict[Future, tuple[int, int]],
    free_slots: list[int],
    conn: sqlite3.Connection,
) -> int:
    freed = 0
    for fut in [f for f in active if f.done()]:
        tok, slot_id = active.pop(fut)
        freed += tok
        free_slots.append(slot_id)
        _record(conn, fut.result())
    return freed


def _dispatch(
    pool: ThreadPoolExecutor,
    conn: sqlite3.Connection,
    active: dict[Future, tuple[int, int]],
    free_slots: list[int],
    ctx: PoolCtx,
    used_tokens: int,
) -> int:
    avail = ctx.max_tokens - used_tokens
    rows = conn.execute(
        "SELECT job_id, experiment, config_id, toml_path, ckpt_dir, rounds, rep, tokens FROM jobs WHERE status='PENDING' AND tokens <= ? ORDER BY rep ASC, config_id ASC",
        (avail,),
    ).fetchall()
    for row in rows:
        tok = int(row[7])
        if used_tokens + tok > ctx.max_tokens or not free_slots:
            continue
        slot_id = free_slots.pop(0)
        conn.execute(
            "UPDATE jobs SET status='RUNNING', attempts=attempts+1, updated_at=? WHERE job_id=?",
            (time.time(), row[0]),
        )
        conn.commit()
        used_tokens += tok
        active[pool.submit(_execute, ctx, row, slot_id)] = (tok, slot_id)
    return used_tokens


def _pending(conn: sqlite3.Connection) -> bool:
    rem = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='PENDING'").fetchone()[0]
    return int(rem) > 0


def _drive(pool: ThreadPoolExecutor, conn: sqlite3.Connection, ctx: PoolCtx) -> None:
    used_tokens = 0
    free_slots = list(range(max(1, ctx.max_tokens)))
    active: dict[Future, tuple[int, int]] = {}
    while True:
        if active:
            used_tokens -= _drain(active, free_slots, conn)
        used_tokens = _dispatch(pool, conn, active, free_slots, ctx, used_tokens)
        if not active and not _pending(conn):
            return
        time.sleep(0.15)


def _loop(conn: sqlite3.Connection, ctx: PoolCtx) -> None:
    with ThreadPoolExecutor(max_workers=max(1, ctx.max_tokens)) as pool:
        _drive(pool, conn, ctx)


def cmd_init(
    campaign_dir: str, experiment: tuple[str, ...], rounds: int, seeds: int
) -> None:
    """Initialize SQLite campaign jobs and TOML specs."""
    cdir = Path(campaign_dir).resolve()
    conn = _db(cdir / "suite_b.sqlite3")
    added = 0
    for exp in experiment:
        for cid, toml_path, tokens in write_specs(cdir / "configs", exp, rounds, seeds):
            for rep in range(seeds):
                jid = f"{exp}__{cid}__rep{rep}"
                ckpt = cdir / "checkpoints" / exp / cid
                cur = conn.execute(
                    "INSERT OR IGNORE INTO jobs (job_id, experiment, config_id, toml_path, ckpt_dir, rounds, rep, tokens, status, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)",
                    (
                        jid,
                        exp,
                        cid,
                        str(toml_path),
                        str(ckpt),
                        rounds,
                        rep,
                        tokens,
                        time.time(),
                    ),
                )
                added += cur.rowcount
    conn.commit()
    conn.close()
    click.echo(f"Initialized campaign at {cdir} ({added} new seed-jobs queued).")


def cmd_run(campaign_dir: str, max_tokens: int, ckpt_every: int) -> None:
    """Execute pending campaign jobs under bounded worker concurrency."""
    repo_root = Path(__file__).resolve().parents[2]
    cdir = Path(campaign_dir).resolve()
    conn = _db(cdir / "suite_b.sqlite3")
    conn.execute("UPDATE jobs SET status='PENDING' WHERE status='RUNNING'")
    conn.commit()
    ctx = PoolCtx(_runner(repo_root), repo_root, max_tokens, ckpt_every)
    caff = _caffeinate()
    try:
        _loop(conn, ctx)
    finally:
        _shutdown()
        if caff is not None:
            caff.terminate()
        conn.close()
    click.echo(build_report(cdir))


def cmd_status(campaign_dir: str) -> None:
    """Print job status counts grouped by experiment."""
    conn = _db(Path(campaign_dir).resolve() / "suite_b.sqlite3")
    rows = conn.execute(
        "SELECT experiment, status, COUNT(*) FROM jobs GROUP BY experiment, status ORDER BY experiment, status"
    ).fetchall()
    conn.close()
    for exp, st, cnt in rows:
        click.echo(f"{exp:18s} {st:12s} {cnt}")


def cmd_report(campaign_dir: str) -> None:
    """Generate and print the campaign summary report."""
    click.echo(build_report(Path(campaign_dir).resolve()))


cli = click.Group(
    "bench_suite",
    commands={
        "init": click.Command(
            "init",
            callback=cmd_init,
            params=[
                click.Option(["--campaign-dir"], required=True),
                click.Option(
                    ["--experiment"],
                    type=click.Choice(list(SIM_TOKENS)),
                    multiple=True,
                    default=("tokamak-tokagym",),
                ),
                click.Option(["--rounds"], type=int, default=10000),
                click.Option(["--seeds"], type=int, default=10),
            ],
        ),
        "run": click.Command(
            "run",
            callback=cmd_run,
            params=[
                click.Option(["--campaign-dir"], required=True),
                click.Option(["--max-tokens"], type=int, default=4),
                click.Option(["--ckpt-every"], type=int, default=250),
            ],
        ),
        "status": click.Command(
            "status",
            callback=cmd_status,
            params=[click.Option(["--campaign-dir"], required=True)],
        ),
        "report": click.Command(
            "report",
            callback=cmd_report,
            params=[click.Option(["--campaign-dir"], required=True)],
        ),
    },
)

if __name__ == "__main__":
    cli()
