use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum BpannError {
    #[error("Invalid shape: expected {expected:?}, got {got:?}")]
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

impl From<ndarray::ShapeError> for BpannError {
    fn from(e: ndarray::ShapeError) -> Self {
        BpannError::InvalidParameter(e.to_string())
    }
}

impl From<std::io::Error> for BpannError {
    fn from(e: std::io::Error) -> Self {
        BpannError::InvalidParameter(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let shape_err = BpannError::InvalidShape {
            expected: vec![2, 3],
            got: vec![2, 4],
        };
        assert_eq!(
            shape_err.to_string(),
            "Invalid shape: expected [2, 3], got [2, 4]"
        );

        let param_err = BpannError::InvalidParameter("bad k".to_string());
        assert_eq!(param_err.to_string(), "Invalid parameter: bad k");
    }

    #[test]
    fn from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
        let bpann_err = BpannError::from(io_err);
        assert_eq!(
            bpann_err,
            BpannError::InvalidParameter("missing file".to_string())
        );
    }
}
