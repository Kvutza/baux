//! Required arithmetic checks for certified workers, explicit unchecked legacy
//! use.
//!
//! A checked result concerns scalar arithmetic and comparison only. It does not
//! certify calibration, measurement error, GPU execution or compiler
//! correctness.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Paired = 0,
    Combined = 1,
    Separate = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Unchecked,
    Checked,
}

impl Outcome {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unchecked => "unchecked",
            Self::Checked => "arithmetic-checked",
        }
    }
}

/// Exact words from one observed threshold and comparison, never tensor data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trace {
    pub kind: Kind,
    pub improvement: u64,
    pub multiplier: u64,
    pub left: u64,
    pub right: u64,
    pub threshold: u64,
    pub accepted: bool,
}

impl Trace {
    pub fn observed(
        kind: Kind,
        improvement: f64,
        multiplier: f64,
        left: f64,
        right: f64,
        threshold: f64,
        accepted: bool,
    ) -> Self {
        Self {
            kind,
            improvement: improvement.to_bits(),
            multiplier: multiplier.to_bits(),
            left: left.to_bits(),
            right: right.to_bits(),
            threshold: threshold.to_bits(),
            accepted,
        }
    }
}

pub type Verifier = fn(Trace) -> Result<(), String>;
static VERIFIER: OnceLock<Verifier> = OnceLock::new();
static REQUIRED: AtomicBool = AtomicBool::new(false);

/// Install the host's process-lifetime checker before any worker execution.
///
/// # Safety
/// Success must implement the checked Lean numerical predicate for every trace.
/// The callback and any foreign library it uses must remain valid for the
/// process. Tests may inject an explicitly identified test oracle in isolated
/// processes; that does not establish a production certificate.
pub unsafe fn install(verifier: Verifier) -> Result<(), String> {
    VERIFIER
        .set(verifier)
        .map_err(|_| "numerical verifier is already installed".into())
}

/// Require checking for this process; absence fails before the worker starts.
pub fn require() -> Result<(), String> {
    if VERIFIER.get().is_none() {
        return Err("required numerical verifier is not installed".into());
    }
    REQUIRED.store(true, Ordering::Release);
    Ok(())
}

/// Unsupported/legacy paths stay visibly unchecked, or fail in a required
/// worker.
pub fn unchecked(reason: &str) -> Result<Outcome, String> {
    if REQUIRED.load(Ordering::Acquire) {
        Err(format!(
            "required numerical verification unavailable: {reason}"
        ))
    } else {
        Ok(Outcome::Unchecked)
    }
}

/// Check the observed words before mutating optimizer state or committing
/// weights.
pub fn check(trace: Trace) -> Result<Outcome, String> {
    match VERIFIER.get() {
        Some(verifier) => {
            verifier(trace)?;
            Ok(Outcome::Checked)
        }
        None => unchecked("no checker installed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_capture() {
        let trace = Trace::observed(Kind::Paired, -0.0, 2.0, 1.0, 0.0, 2.0, false);
        assert_eq!(trace.improvement, (-0.0f64).to_bits());
        assert_eq!(trace.threshold, 2.0f64.to_bits());
        assert_eq!(trace.right, 0);
    }

    #[test]
    fn absent_verifier() {
        // No test installs a process-global permissive verifier.
        assert_eq!(unchecked("legacy").unwrap(), Outcome::Unchecked);
        assert!(require().unwrap_err().contains("not installed"));
    }
    #[test]
    fn required_mode() {
        if std::env::var_os("BAUX_NUMERICAL_TEST").is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "numerical::tests::required_mode"])
                .env("BAUX_NUMERICAL_TEST", "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        fn reject(_: Trace) -> Result<(), String> {
            Err("intentional rejecting test verifier".into())
        }
        // SAFETY: this verifier never succeeds, so it cannot falsely certify a
        // trace.
        unsafe { install(reject) }.unwrap();
        require().unwrap();
        assert!(unchecked("unsupported path").is_err());
        assert!(unsafe { install(reject) }.is_err());
        assert!(
            check(Trace::observed(
                Kind::Paired,
                1.0,
                0.25,
                1.0,
                0.25,
                0.1,
                true
            ))
            .is_err()
        );
    }
}
