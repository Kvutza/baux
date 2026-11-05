//! Explicit versioned independent observations. Reporting rewards never define
//! control.
#[cfg(test)]
use super::code_environment::{Estimate, validate_vector};
pub(crate) use super::code_environment::{RewardResult, VectorResult, native_vector, parse_vector};
use super::generation::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn vector() -> VectorResult {
        baux::artifact::json::from_str(r#"{"schema":"baux.generation_objectives.v2","rewards":[1000.0],"objectives":[{"mean":-100.0,"variance":0.25},{"mean":100.0,"variance":0.5}],"control":{"mean":-2.0,"variance":0.125}}"#).unwrap()
    }

    #[test]
    fn explicit_control() {
        let result = validate_vector(vector(), 1).unwrap();
        let observation = result.observation().unwrap();
        assert_eq!(result.rewards, [1000.0]);
        assert_eq!(observation.control().mean, -2.0);
        assert_eq!(observation.control().variance, 0.125);
        assert_eq!(observation.estimates()[0].mean, -100.0);
        assert_eq!(observation.estimates()[1].variance, 0.5);
    }

    #[test]
    fn protocol_guards() {
        let mut wrong = vector();
        wrong.schema = "baux.generation_reward.v1".into();
        assert!(validate_vector(wrong, 1).is_err());
        assert!(validate_vector(vector(), 2).is_err());
        for width in [0, 1, 9] {
            let mut wrong = vector();
            wrong.objectives.resize(
                width,
                Estimate {
                    mean: 0.0,
                    variance: 0.0,
                },
            );
            assert!(validate_vector(wrong, 1).is_err());
        }
        for value in [-1.0, f32::INFINITY, f32::NAN] {
            let mut wrong = vector();
            wrong.control.variance = value;
            assert!(validate_vector(wrong, 1).is_err());
            let mut wrong = vector();
            wrong.objectives[0].variance = value;
            assert!(validate_vector(wrong, 1).is_err());
        }
    }

    #[test]
    fn scalar_compatibility() {
        for rewards in [vec![1.0], vec![-2.0, 3.0, 4.0]] {
            let expected = if rewards.len() == 1 {
                (rewards[0], 0.0)
            } else {
                mean_variance(&rewards)
            };
            let result = RewardResult {
                rewards,
                vector: None,
                variance: None,
                control_only: false,
            };
            let control = result.observation().unwrap().control();
            assert_eq!(control.mean.to_bits(), expected.0.to_bits());
            assert_eq!(control.variance.to_bits(), expected.1.to_bits());
        }
        assert!(baux::artifact::json::from_str::<VectorResult>(r#"{"rewards":[0.0]}"#).is_err());
    }

    #[test]
    fn native_objectives() {
        let mut result = native_vector(
            vec![0.25, 0.75],
            &[vec![-2.0, -1.0], vec![0.25, 0.75]],
            &["likelihood", "quality"],
        )
        .unwrap();
        let observation = result.observation().unwrap();
        assert_eq!(result.rewards, [0.25, 0.75]);
        assert_eq!(observation.control().mean, 0.5);
        assert_eq!(observation.estimates()[0].mean, -1.5);
        assert_eq!(observation.estimates()[1].mean, 0.5);
        result.control_only = true;
        result.variance = Some(0.125);
        let control = result.observation().unwrap();
        assert!(!control.is_vector());
        assert_eq!(control.control().mean, 0.5);
        assert_eq!(control.control().variance, 0.125);
        assert!(native_vector(vec![0.0], &[vec![0.0], Vec::new()], &["a", "b"]).is_err());
    }
}
