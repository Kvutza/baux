use clap::{Subcommand, ValueEnum};
use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub enum RadarAction {
    /// Build the compiler-resolved Rust SCIP index.
    Index {
        /// Rebuild even when the cached index is current.
        #[arg(long)]
        force: bool,
        /// Target repository or crate directory to analyze (defaults to
        /// workspace root).
        #[arg(long)]
        target: Option<PathBuf>,
    },
    /// Emit a compiler-verified RepoMap (centrality-ranked symbols, signatures,
    /// and call counts).
    Map {
        /// Target repository or crate directory to analyze (defaults to
        /// workspace root).
        #[arg(long)]
        target: Option<PathBuf>,
        /// Filter to documents whose path contains this substring.
        #[arg(long)]
        filter: Option<String>,
        /// Maximum number of ranked symbols to display.
        #[arg(long, default_value_t = 80)]
        limit: usize,
    },
    /// Find a compiler-resolved Rust symbol and its exact source locations.
    Symbol {
        /// Exact or partial symbol name.
        name: String,
        /// Include reference locations as well as definitions.
        #[arg(long)]
        refs: bool,
        /// Maximum number of source locations to print.
        #[arg(long, default_value_t = 40)]
        limit: usize,
        /// Target repository or crate directory to analyze (defaults to
        /// workspace root).
        #[arg(long)]
        target: Option<PathBuf>,
    },
    /// Report the availability of every execution target.
    Doctor,
    /// Search for code patterns with enclosing AST symbols and complexity.
    Find {
        /// Pattern or literal text to search.
        pattern: String,
        /// Treat pattern as regular expression instead of literal string.
        #[arg(long, short = 'e')]
        regex: bool,
        /// Source context lines around matches.
        #[arg(long, short = 'C', default_value_t = 0)]
        context: usize,
        /// Target repository or crate directory to analyze (defaults to
        /// workspace root).
        #[arg(long)]
        target: Option<PathBuf>,
    },
    /// Transitive blast radius and reachability of a symbol across the call
    /// graph.
    Impact {
        /// Symbol identifier (e.g. "path/to/file.rs:fn_name" or "fn_name").
        symbol: String,
        /// Target repository or crate directory to analyze (defaults to
        /// workspace root).
        #[arg(long)]
        target: Option<PathBuf>,
    },
    /// Direct and transitive callers of a symbol.
    Callers {
        /// Symbol identifier.
        symbol: String,
        /// Target repository or crate directory to analyze (defaults to
        /// workspace root).
        #[arg(long)]
        target: Option<PathBuf>,
    },
    /// Direct and transitive callees called by a symbol.
    Callees {
        /// Symbol identifier.
        symbol: String,
        /// Target repository or crate directory to analyze (defaults to
        /// workspace root).
        #[arg(long)]
        target: Option<PathBuf>,
    },
    /// Compute Tarjan SCCs, minimum feedback-arc-set cuts, and effort/payoff
    /// frontier over SCIP.
    Split {
        /// Target repository or crate directory to analyze (defaults to
        /// workspace root).
        #[arg(long)]
        target: Option<PathBuf>,
        /// Module path depth to group files into candidate crates/workspaces.
        #[arg(long, default_value_t = 2)]
        granularity: usize,
        /// Include crate root (`lib.rs`/`main.rs`) barrel re-exports in the
        /// module graph.
        #[arg(long)]
        include_root: bool,
    },
    /// Inspect and gate BO plus generation performance evidence.
    #[command(alias = "perf")]
    Performance {
        #[command(subcommand)]
        command: PerformanceAction,
    },
}

pub fn run(
    root: &Path,
    action: RadarAction,
    performance: impl FnOnce(&Path, PerformanceAction) -> Result<(), String>,
) -> Result<(), String> {
    match action {
        RadarAction::Index { force, target } => {
            crate::radar_scip::index(root, target.as_deref(), force)
        }
        RadarAction::Map {
            target,
            filter,
            limit,
        } => crate::radar_scip::map(root, target.as_deref(), filter.as_deref(), limit),
        RadarAction::Symbol {
            name,
            refs,
            limit,
            target,
        } => crate::radar_scip::symbol(root, target.as_deref(), &name, refs, limit),
        RadarAction::Doctor => crate::radar_scip::doctor(root),
        RadarAction::Find {
            pattern,
            regex,
            context,
            target,
        } => crate::radar_scip::find(root, target.as_deref(), &pattern, regex, context),
        RadarAction::Impact { symbol, target } => {
            crate::radar_scip::impact(root, target.as_deref(), &symbol)
        }
        RadarAction::Callers { symbol, target } => {
            crate::radar_scip::callers(root, target.as_deref(), &symbol)
        }
        RadarAction::Callees { symbol, target } => {
            crate::radar_scip::callees(root, target.as_deref(), &symbol)
        }
        RadarAction::Split {
            target,
            granularity,
            include_root,
        } => crate::radar_scip::split(root, target.as_deref(), granularity, !include_root),
        RadarAction::Performance { command } => performance(root, command),
    }
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub enum PerformanceAction {
    /// Show a compact decision report or the complete evidence document.
    Show {
        /// Experiment whose exact workload is being certified.
        #[arg(default_value = "performance/configs/mhc4-1m-causal-1000.toml")]
        config: PathBuf,
        /// Accelerator lowering used by the experiment.
        #[arg(long, visible_alias = "backend", value_enum, default_value_t = PerformanceTarget::Metal)]
        target: PerformanceTarget,
        /// Completed run directory supplying measured evidence.
        #[arg(long)]
        run: Option<PathBuf>,
        /// Print the complete machine-readable evidence document.
        #[arg(long)]
        json: bool,
    },
    /// Write a new measured baseline after a completed matching-source run.
    Record {
        #[arg(default_value = "performance/configs/mhc4-1m-causal-1000.toml")]
        config: PathBuf,
        #[arg(long, visible_alias = "backend", value_enum, default_value_t = PerformanceTarget::Metal)]
        target: PerformanceTarget,
        /// Completed run directory produced by the current hot-path source.
        #[arg(long)]
        run: PathBuf,
        /// Baseline artifact to create or replace.
        #[arg(long, default_value = "performance/mhc4-1m-metal.json")]
        output: PathBuf,
    },
    /// Reject hot-path regressions against a measured baseline.
    Gate {
        #[arg(default_value = "performance/configs/mhc4-1m-causal-1000.toml")]
        config: PathBuf,
        #[arg(long, visible_alias = "backend", value_enum, default_value_t = PerformanceTarget::Metal)]
        target: PerformanceTarget,
        /// Fresh completed run. Required only after the hot path changes.
        #[arg(long)]
        run: Option<PathBuf>,
        #[arg(long, default_value = "performance/mhc4-1m-metal.json")]
        baseline: PathBuf,
        /// Allowed p95 increase for noisy device measurements, as a percentage.
        #[arg(long, default_value_t = 5)]
        tolerance_percent: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PerformanceTarget {
    Metal,
    CudaOxide,
}

impl PerformanceTarget {
    pub fn name(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::CudaOxide => "cuda-oxide",
        }
    }
}
