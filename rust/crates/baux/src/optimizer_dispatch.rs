//! Pure public-API dispatch decisions shared by production and extraction.
//!
//! Keep these functions free of optimizer representation details: they are the
//! source-visible ordering of the guards that precede the mutating ask/tell
//! bodies.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AskGate {
    ZeroArms,
    RestartLocal,
    Dispatch,
}

pub(crate) fn ask_gate(num_arms: usize, is_morbo: bool, needs_restart: bool) -> AskGate {
    if num_arms == 0 {
        AskGate::ZeroArms
    } else if !is_morbo && needs_restart {
        AskGate::RestartLocal
    } else {
        AskGate::Dispatch
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TellGate {
    InvalidX,
    InvalidRows,
    Empty,
    Dispatch,
}

pub(crate) fn tell_gate(num_dim: usize, x_rows: usize, x_cols: usize, y_rows: usize) -> TellGate {
    if x_cols != num_dim {
        TellGate::InvalidX
    } else if y_rows != x_rows {
        TellGate::InvalidRows
    } else if x_rows == 0 {
        TellGate::Empty
    } else {
        TellGate::Dispatch
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VarianceGate {
    Set(bool),
    Keep,
    Reject,
}

pub(crate) fn variance_gate(current: Option<bool>, present: bool) -> VarianceGate {
    match current {
        None => VarianceGate::Set(present),
        Some(expected) if expected != present => VarianceGate::Reject,
        Some(_) => VarianceGate::Keep,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ask_order() {
        assert_eq!(ask_gate(0, false, true), AskGate::ZeroArms);
        assert_eq!(ask_gate(1, false, true), AskGate::RestartLocal);
        assert_eq!(ask_gate(1, true, true), AskGate::Dispatch);
        assert_eq!(ask_gate(1, false, false), AskGate::Dispatch);
    }

    #[test]
    fn tell_order() {
        assert_eq!(tell_gate(3, 0, 2, 1), TellGate::InvalidX);
        assert_eq!(tell_gate(3, 0, 3, 1), TellGate::InvalidRows);
        assert_eq!(tell_gate(3, 0, 3, 0), TellGate::Empty);
        assert_eq!(tell_gate(3, 1, 3, 1), TellGate::Dispatch);
    }

    #[test]
    fn variance_order() {
        assert_eq!(variance_gate(None, true), VarianceGate::Set(true));
        assert_eq!(variance_gate(Some(true), true), VarianceGate::Keep);
        assert_eq!(variance_gate(Some(true), false), VarianceGate::Reject);
    }
}
