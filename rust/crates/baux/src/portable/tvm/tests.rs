//! End-to-end model compilation, not just ABI or individual kernel checks.
use super::{TvmExecutable, TvmTensor};
use crate::portable::Tensor;

// JAX 0.12 jit(model).lower(...).compiler_ir("stablehlo"), where model is
// hidden=tanh(x@w1+b1); return hidden@w2+b2,hidden. Parameters stay explicit.
const MLP: &str = r#"module @jit_model {
  func.func public @main(%x: tensor<3x8xf32>, %w1: tensor<8x16xf32>, %b1: tensor<16xf32>, %w2: tensor<16x4xf32>, %b2: tensor<4xf32>) -> (tensor<3x4xf32>, tensor<3x16xf32>) {
    %0 = stablehlo.dot_general %x, %w1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x8xf32>, tensor<8x16xf32>) -> tensor<3x16xf32>
    %1 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<3x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<3x16xf32>
    %4 = stablehlo.tanh %3 : tensor<3x16xf32>
    %5 = stablehlo.dot_general %4, %w2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x16xf32>, tensor<16x4xf32>) -> tensor<3x4xf32>
    %6 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<3x4xf32>
    %8 = stablehlo.add %5, %7 : tensor<3x4xf32>
    return %8, %4 : tensor<3x4xf32>, tensor<3x16xf32>
  }
}"#;

fn compile(source: &str) -> Result<TvmExecutable, String> {
    TvmExecutable::compile(
        source,
        "main",
        std::path::Path::new(&std::env::var("BAUX_MLIR_IMPORTER").unwrap()),
    )
}
fn put(shape: Vec<usize>, values: &[f32]) -> TvmTensor {
    TvmTensor::put(
        Tensor::new(shape, "float32").unwrap(),
        &values
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
fn numbers(tensor: &TvmTensor) -> Vec<f32> {
    tensor
        .read()
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn mlp_forward() {
    let start = std::time::Instant::now();
    let model = compile(MLP).unwrap();
    let compilation = start.elapsed();
    let shapes = [vec![3, 8], vec![8, 16], vec![16], vec![16, 4], vec![4]];
    let mut worst = 0f32;
    for seed in 0..3 {
        let values = shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| {
                (0..shape.iter().product())
                    .map(|j| ((j * 17 + i * 11 + seed * 23) as f32 * 0.31).sin() * 0.25)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut hidden = vec![0f32; 48];
        let mut expected = vec![0f32; 12];
        for row in 0..3 {
            for col in 0..16 {
                hidden[row * 16 + col] = ((0..8)
                    .map(|k| values[0][row * 8 + k] * values[1][k * 16 + col])
                    .sum::<f32>()
                    + values[2][col])
                    .tanh();
            }
            for col in 0..4 {
                expected[row * 4 + col] = (0..16)
                    .map(|k| hidden[row * 16 + k] * values[3][k * 4 + col])
                    .sum::<f32>()
                    + values[4][col];
            }
        }
        let inputs = shapes
            .iter()
            .zip(&values)
            .map(|(s, v)| put(s.clone(), v))
            .collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let outputs = model.submit(inputs.clone()).unwrap().wait().unwrap();
        let elapsed = start.elapsed();
        drop(inputs);
        for (output, reference) in outputs.iter().zip([expected, hidden]) {
            for (actual, expected) in numbers(output).iter().zip(&reference) {
                worst = worst.max((actual - expected).abs());
                assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
            }
        }
        eprintln!(
            "MLP seed={seed}, compile={compilation:?}, run={elapsed:?}, max_abs_error={worst}"
        );
    }
    assert!(model.run(&[]).is_err());
}

#[test]
fn scalar_empty() {
    let model = compile(
        r#"module { func.func @main(%x: tensor<0x2xf32>) -> (tensor<0x2xf32>, tensor<f32>) {
        %c = stablehlo.constant dense<0.25> : tensor<f32>
        return %x, %c : tensor<0x2xf32>, tensor<f32>
    }}"#,
    )
    .unwrap();
    let outputs = model.submit(vec![put(vec![0, 2], &[])]).unwrap();
    drop(model);
    let outputs = outputs.wait().unwrap();
    assert!(numbers(&outputs[0]).is_empty());
    assert_eq!(numbers(&outputs[1]), [0.25]);
}

#[test]
fn constants_aliases() {
    let model = compile(r#"module { func.func @main(%x: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
        %c = stablehlo.constant dense<[1.25, -2.0]> : tensor<2xf32>
        %y = stablehlo.add %x, %c : tensor<2xf32>
        return %x, %y, %y : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
    }}"#).unwrap();
    let x = put(vec![2], &[2., 3.]);
    let results = model.run(&[&x]).unwrap();
    drop(model);
    assert_eq!(numbers(&results[0]), [2., 3.]);
    assert_eq!(numbers(&results[1]), [3.25, 1.]);
    assert_eq!(numbers(&results[2]), [3.25, 1.]);
}

#[test]
fn rejects_unsupported() {
    assert!(compile("not MLIR").is_err());
    let error = compile(
        r#"module { func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
        %0 = stablehlo.floor %x : tensor<2xf32>
        return %0 : tensor<2xf32>
    }}"#,
    )
    .err()
    .unwrap();
    assert!(
        error.contains("unsupported StableHLO operation stablehlo.floor"),
        "{error}"
    );
}
