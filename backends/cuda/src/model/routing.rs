use super::{EXPERTS, FbtModel, Stage, VOCAB, WIDTH, mark, weights::Layer};
use crate::{CudaResult, cuda_error, read_prefix};
use baux_cuda_device::{MatmulShape, MoeShape, RouteShape, RoutedTile, fbt_model};
use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, LaunchConfig1D, simt::LaunchConfig2D,
};

impl FbtModel {
    pub(super) fn moe(
        &mut self,
        layer: usize,
        events: &mut Option<&mut Vec<(Stage, CudaEvent)>>,
    ) -> CudaResult<()> {
        let s = &mut self.scratch;
        let weights = &self.weights.layers[layer];
        let rows = self.rows as u32;
        let route = RouteShape {
            rows,
            experts: EXPERTS as u32,
            top_k: 3,
        };
        let moe = MoeShape {
            row_tiles: 1,
            ..MoeShape::fbt()
        };
        matmul(
            self.synth.as_ref(),
            &self.module,
            &self.stream,
            &s.normalized,
            &weights.router,
            &mut s.route_scores,
            self.rows,
            EXPERTS,
            WIDTH,
        )?;
        mark(events, &self.stream, Stage::MoeRouter)?;
        let launch = self
            .module
            .prepare_routetopk(LaunchConfig1D::new(rows, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .routetopk(
                &self.stream,
                &launch,
                &s.route_scores,
                &mut s.experts,
                &mut s.route_weights,
                &mut s.margins,
                route,
            )
            .map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_route_counts(LaunchConfig1D::new(EXPERTS as u32, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .route_counts(
                &self.stream,
                &launch,
                &s.experts,
                &mut s.counts,
                &mut s.prefixes,
                rows * 3,
            )
            .map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_route_layout(LaunchConfig1D::new(1, 32, 0))
            .map_err(cuda_error)?;
        self.module
            .route_layout(
                &self.stream,
                &launch,
                &s.counts,
                &mut s.offsets,
                &mut s.tiles,
                EXPERTS as u32,
                s.tile_capacity as u32,
            )
            .map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_route_pack(LaunchConfig1D::new(rows * 3, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .route_pack(
                &self.stream,
                &launch,
                &s.normalized,
                &s.experts,
                &s.offsets,
                &s.prefixes,
                &mut s.mapping,
                &mut s.packed,
                rows * 3,
            )
            .map_err(cuda_error)?;
        mark(events, &self.stream, Stage::MoeRoute)?;
        experts(
            &self.module,
            &self.stream,
            weights,
            &s.packed,
            &s.tiles,
            &mut s.projection,
            &mut s.activation,
            &mut s.routed,
            s.tile_capacity,
            rows * 3,
            self.rows <= 2048,
        )?;
        experts(
            &self.module,
            &self.stream,
            weights,
            &s.normalized,
            &s.shared_tiles,
            &mut s.shared_projection,
            &mut s.shared_activation,
            &mut s.shared,
            self.rows.div_ceil(64),
            rows,
            false,
        )?;
        mark(events, &self.stream, Stage::MoeExperts)?;
        let launch = self
            .module
            .prepare_combine_mhc(LaunchConfig1D::new((rows * 512).div_ceil(256), 256, 0))
            .map_err(cuda_error)?;
        self.module
            .combine_mhc(
                &self.stream,
                &launch,
                &s.route_weights,
                &s.mapping,
                &s.routed,
                &s.shared,
                &s.coefficients,
                &mut s.streams,
                route,
                moe,
            )
            .map_err(cuda_error)?;
        mark(events, &self.stream, Stage::MoeCombine)
    }
}

fn experts(
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    weights: &Layer,
    input: &DeviceBuffer<u16>,
    tiles: &DeviceBuffer<RoutedTile>,
    projection: &mut DeviceBuffer<u16>,
    activation: &mut DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    capacity: usize,
    rows: u32,
    compact: bool,
) -> CudaResult<()> {
    expert_project(
        module,
        stream,
        input,
        &weights.gate,
        tiles,
        projection,
        capacity,
        0,
        compact,
    )?;
    let launch = module
        .prepare_routed_activate(LaunchConfig1D::new((rows * 216).div_ceil(256), 256, 0))
        .map_err(cuda_error)?;
    module
        .routed_activate(stream, &launch, projection, activation, rows)
        .map_err(cuda_error)?;
    expert_project(
        module,
        stream,
        activation,
        &weights.down,
        tiles,
        output,
        capacity,
        1,
        compact,
    )
}

fn expert_project(
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    weights: &DeviceBuffer<u16>,
    tiles: &DeviceBuffer<RoutedTile>,
    output: &mut DeviceBuffer<u16>,
    capacity: usize,
    mode: u32,
    compact: bool,
) -> CudaResult<()> {
    let columns: u32 = if mode == 0 { 432 } else { 512 };
    if compact {
        let launch = module
            .prepare_routed_compact(LaunchConfig2D::new(
                (columns.div_ceil(16), capacity as u32 * 4),
                (64, 1),
                0,
            ))
            .map_err(cuda_error)?;
        module
            .routed_compact(
                stream,
                &launch,
                input,
                weights,
                tiles,
                cuda_host::RowWidth::new(output, columns),
                mode,
            )
            .map_err(cuda_error)
    } else {
        let launch = module
            .prepare_routed_project(LaunchConfig2D::new(
                (columns.div_ceil(16), capacity as u32),
                (256, 1),
                0,
            ))
            .map_err(cuda_error)?;
        module
            .routed_project(
                stream,
                &launch,
                input,
                weights,
                tiles,
                cuda_host::RowWidth::new(output, columns),
                mode,
            )
            .map_err(cuda_error)
    }
}

pub(super) fn check() -> CudaResult<()> {
    const ROWS: usize = 400;
    const EXPERTS: usize = super::EXPERTS;
    const ASSIGNMENTS: usize = ROWS * 3;
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    let module = crate::capture::load_model(&context)?;
    let (experts, routes) = check_topk(&module, &stream, ROWS, EXPERTS)?;
    let input = (0..ROWS * 512)
        .map(|index| 0x3000 + ((index / 512) % 16) as u16 * 64 + (index % 4) as u16)
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&stream, &input).map_err(cuda_error)?;
    let mut counts = DeviceBuffer::zeroed(&stream, EXPERTS).map_err(cuda_error)?;
    let mut prefixes =
        DeviceBuffer::zeroed(&stream, EXPERTS * ASSIGNMENTS.div_ceil(1024)).map_err(cuda_error)?;
    let mut offsets = DeviceBuffer::zeroed(&stream, EXPERTS).map_err(cuda_error)?;
    let capacity = ASSIGNMENTS.div_ceil(16) + EXPERTS;
    let mut tiles = DeviceBuffer::<RoutedTile>::zeroed(&stream, capacity).map_err(cuda_error)?;
    let mut mapping = DeviceBuffer::zeroed(&stream, ASSIGNMENTS).map_err(cuda_error)?;
    let mut packed = DeviceBuffer::<u16>::zeroed(&stream, ASSIGNMENTS * 512).map_err(cuda_error)?;
    let launch = module
        .prepare_route_counts(LaunchConfig1D::new(EXPERTS as u32, 256, 0))
        .map_err(cuda_error)?;
    module
        .route_counts(
            &stream,
            &launch,
            &experts,
            &mut counts,
            &mut prefixes,
            ASSIGNMENTS as u32,
        )
        .map_err(cuda_error)?;
    let launch = module
        .prepare_route_layout(LaunchConfig1D::new(1, 32, 0))
        .map_err(cuda_error)?;
    module
        .route_layout(
            &stream,
            &launch,
            &counts,
            &mut offsets,
            &mut tiles,
            EXPERTS as u32,
            capacity as u32,
        )
        .map_err(cuda_error)?;
    let launch = module
        .prepare_route_pack(LaunchConfig1D::new(ASSIGNMENTS as u32, 256, 0))
        .map_err(cuda_error)?;
    module
        .route_pack(
            &stream,
            &launch,
            &input,
            &experts,
            &offsets,
            &prefixes,
            &mut mapping,
            &mut packed,
            ASSIGNMENTS as u32,
        )
        .map_err(cuda_error)?;
    let counts = read_prefix(&counts, &stream, EXPERTS)?;
    let _tile_layout = read_prefix(&tiles, &stream, capacity)?;
    let mapping = read_prefix(&mapping, &stream, ASSIGNMENTS)?;
    let packed_host = read_prefix(&packed, &stream, ASSIGNMENTS * 512)?;
    verify_pack(&routes, &counts, &mapping, &packed_host, ROWS)?;
    moe(&module, &stream, &packed, &tiles, capacity)?;
    sampling(&module, &stream)?;
    context.check_err().map_err(cuda_error)?;
    println!(
        "ROUTE_PARITY ok=true experts=625 assignments=1200 high-ids=622,623,624 ties=stable \
         chunk-boundary=1024"
    );
    Ok(())
}

fn check_topk(
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    rows: usize,
    experts_count: usize,
) -> CudaResult<(DeviceBuffer<u32>, Vec<u32>)> {
    let assignments = rows * 3;
    let mut scores = vec![0_u16; rows * experts_count];
    for row in (0..rows).step_by(2) {
        for (expert, value) in [(624, 0x4000), (623, 0x3c00), (622, 0x3800)] {
            scores[row * experts_count + expert] = value;
        }
    }
    let scores = DeviceBuffer::from_host(stream, &scores).map_err(cuda_error)?;
    let mut experts = DeviceBuffer::zeroed(stream, assignments).map_err(cuda_error)?;
    let mut weights = DeviceBuffer::<u16>::zeroed(stream, assignments).map_err(cuda_error)?;
    let mut margins = DeviceBuffer::<f32>::zeroed(stream, rows).map_err(cuda_error)?;
    let shape = RouteShape {
        rows: rows as u32,
        experts: experts_count as u32,
        top_k: 3,
    };
    let launch = module
        .prepare_routetopk(LaunchConfig1D::new(rows as u32, 256, 0))
        .map_err(cuda_error)?;
    module
        .routetopk(
            stream,
            &launch,
            &scores,
            &mut experts,
            &mut weights,
            &mut margins,
            shape,
        )
        .map_err(cuda_error)?;
    let routes = read_prefix(&experts, stream, assignments)?;
    for (row, route) in routes.chunks_exact(3).enumerate() {
        let expected = if row % 2 == 0 {
            [624, 623, 622]
        } else {
            [0, 1, 2]
        };
        if route != expected {
            return Err(format!("625-expert route mismatch at row {row}: {route:?}"));
        }
    }
    Ok((experts, routes))
}

fn verify_pack(
    routes: &[u32],
    counts: &[u32],
    mapping: &[u32],
    packed_host: &[u16],
    rows: usize,
) -> CudaResult<()> {
    const EXPERTS: usize = super::EXPERTS;
    let mut offsets = vec![0_usize; EXPERTS];
    let mut ranks = vec![0_usize; EXPERTS];
    let mut total = 0;
    for expert in 0..EXPERTS {
        let expected = if [0, 1, 2, 622, 623, 624].contains(&expert) {
            rows / 2
        } else {
            0
        };
        if counts[expert] as usize != expected {
            return Err(format!("expert {expert} count mismatch"));
        }
        offsets[expert] = total;
        total += expected;
    }
    for (assignment, &expert) in routes.iter().enumerate() {
        let expert = expert as usize;
        let target = offsets[expert] + ranks[expert];
        ranks[expert] += 1;
        if mapping[assignment] as usize != target
            || packed_host[target * 512..(target + 1) * 512]
                .iter()
                .enumerate()
                .any(|(column, &v)| {
                    v != 0x3000 + ((assignment / 3) % 16) as u16 * 64 + (column % 4) as u16
                })
        {
            return Err(format!(
                "stable expert packing mismatch at assignment {assignment}"
            ));
        }
    }
    Ok(())
}

fn moe(
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    tiles: &DeviceBuffer<RoutedTile>,
    capacity: usize,
) -> CudaResult<()> {
    let rows = input.len() / 512;
    let mut gate = vec![0_u16; 626 * 512 * 432];
    let mut down = vec![0_u16; 626 * 216 * 512];
    for expert in [0, 1, 2, 3, 623, 624, 625] {
        for inner in 0..512 {
            for column in 0..432 {
                gate[expert * 512 * 432 + inner * 432 + column] = if column < 216 {
                    0x1800 + (expert % 3) as u16 * 1024
                } else {
                    0x1400 + (inner % 3) as u16 * 1024
                };
            }
        }
        for inner in 0..216 {
            for column in 0..512 {
                down[expert * 216 * 512 + inner * 512 + column] =
                    0x1800 + (column % 3) as u16 * 1024;
            }
        }
    }
    let gate = DeviceBuffer::from_host(stream, &gate).map_err(cuda_error)?;
    let down = DeviceBuffer::from_host(stream, &down).map_err(cuda_error)?;
    let mut reference_activation = DeviceBuffer::zeroed(stream, rows * 224).map_err(cuda_error)?;
    let mut reference_output = DeviceBuffer::zeroed(stream, rows * 512).map_err(cuda_error)?;
    let mut projection = DeviceBuffer::zeroed(stream, rows * 432).map_err(cuda_error)?;
    let mut activation = DeviceBuffer::zeroed(stream, rows * 224).map_err(cuda_error)?;
    let mut output = DeviceBuffer::zeroed(stream, rows * 512).map_err(cuda_error)?;
    let shape = MoeShape {
        row_tiles: 4,
        ..MoeShape::fbt()
    };
    let launch = module
        .prepare_routed_gate(LaunchConfig2D::new(
            (216_u32.div_ceil(16), capacity as u32 * 4),
            (16, 16),
            0,
        ))
        .map_err(cuda_error)?;
    module
        .routed_gate(
            stream,
            &launch,
            input,
            &gate,
            tiles,
            cuda_host::RowWidth::new(&mut reference_activation, 224),
            shape,
        )
        .map_err(cuda_error)?;
    let launch = module
        .prepare_routed_down(LaunchConfig2D::new((32, capacity as u32 * 4), (16, 16), 0))
        .map_err(cuda_error)?;
    module
        .routed_down(
            stream,
            &launch,
            &reference_activation,
            &down,
            tiles,
            cuda_host::RowWidth::new(&mut reference_output, 512),
            shape,
        )
        .map_err(cuda_error)?;
    expert_project(
        module,
        stream,
        input,
        &gate,
        tiles,
        &mut projection,
        capacity,
        0,
        true,
    )?;
    let launch = module
        .prepare_routed_activate(LaunchConfig1D::new(
            (rows as u32 * 216).div_ceil(256),
            256,
            0,
        ))
        .map_err(cuda_error)?;
    module
        .routed_activate(stream, &launch, &projection, &mut activation, rows as u32)
        .map_err(cuda_error)?;
    expert_project(
        module,
        stream,
        &activation,
        &down,
        tiles,
        &mut output,
        capacity,
        1,
        true,
    )?;
    let reference = read_prefix(&reference_output, stream, rows * 512)?;
    let actual = read_prefix(&output, stream, rows * 512)?;
    let mut maximum = 0.0_f32;
    for (&expected, &value) in reference.iter().zip(&actual) {
        let expected = widen(expected);
        let value = widen(value);
        let error = (expected - value).abs();
        if !value.is_finite() || error > 0.0001 + expected.abs() * 0.001 {
            return Err(format!("tensor-core MoE mismatch: {expected} vs {value}"));
        }
        maximum = maximum.max(error);
    }
    if reference.iter().all(|&value| value == 0) {
        return Err("MoE parity fixture did not exercise a nonzero branch".into());
    }
    let target = crate::target_name(stream.context())?;
    println!(
        "MOE_PARITY ok=true rows={rows} width=512 expert-width=216 target={target} \
         max-abs-error={maximum}"
    );
    Ok(())
}

fn widen(value: u16) -> f32 {
    let exponent = (value >> 10) & 31;
    let fraction = value & 1023;
    let sign = if value & 0x8000 == 0 { 1.0 } else { -1.0 };
    match exponent {
        0 => sign * f32::from(fraction) * 2.0_f32.powi(-24),
        31 => f32::from_bits(
            (u32::from(value & 0x8000) << 16) | 0x7f800000 | (u32::from(fraction) << 13),
        ),
        _ => sign * f32::from(1024 + fraction) * 2.0_f32.powi(i32::from(exponent) - 25),
    }
}

fn sampling(module: &fbt_model::LoadedModule, stream: &CudaStream) -> CudaResult<()> {
    let logits =
        DeviceBuffer::from_host(stream, &vec![0x6000_u16; 32 * VOCAB]).map_err(cuda_error)?;
    let mut sampled = DeviceBuffer::zeroed(stream, 32).map_err(cuda_error)?;
    let launch = module
        .prepare_sample(LaunchConfig1D::new(32, 256, 0))
        .map_err(cuda_error)?;
    let mut reference = Vec::new();
    for temperature in [0.8, 1.0e-8] {
        module
            .sample(
                stream,
                &launch,
                &logits,
                &mut sampled,
                32,
                temperature,
                17,
                0,
            )
            .map_err(cuda_error)?;
        let actual = read_prefix(&sampled, stream, 32)?;
        if reference.is_empty() {
            reference = actual;
        } else if actual != reference {
            return Err("uniform sampling changed at small positive temperature".into());
        }
    }
    if reference.iter().all(|&token| token == reference[0]) {
        return Err("uniform stochastic sampler collapsed".into());
    }
    module
        .sample(stream, &launch, &logits, &mut sampled, 32, 0.0, 17, 0)
        .map_err(cuda_error)?;
    if read_prefix(&sampled, stream, 32)?
        .iter()
        .any(|&token| token != 0)
    {
        return Err("greedy sampling tie break changed".into());
    }
    println!("SAMPLING_PARITY ok=true temperature=0,0.8,1e-8 finite-offset=512 stable-seed=17");
    Ok(())
}

pub(crate) fn matmul(
    synth: Option<&crate::gemm::Gemm>,
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    weights: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    columns: usize,
    inner: usize,
) -> CudaResult<()> {
    matmul_at(
        synth, module, stream, input, weights, output, rows, columns, inner, 0,
    )
}

pub(super) fn matmul_at(
    synth: Option<&crate::gemm::Gemm>,
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    weights: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    columns: usize,
    inner: usize,
    first: usize,
) -> CudaResult<()> {
    if first
        .checked_add(rows)
        .and_then(|v| v.checked_mul(inner))
        .is_none_or(|v| v > input.len())
    {
        return Err("matmul input range exceeds allocation".into());
    }
    // The measured recipe benefits the wide vocabulary readout. Retain the
    // CUDA-Oxide schedule for smaller projections, where synthesis regressed.
    if columns == VOCAB {
        if let Some(kernel) = synth {
            return kernel.range(stream, input, weights, output, rows, columns, inner, first);
        }
    }
    let shape = MatmulShape {
        rows: rows as u32,
        columns: columns as u32,
        inner: inner as u32,
        input_stride: inner as u32,
        weight_stride: columns as u32,
        output_stride: columns as u32,
        input_offset: (first * inner) as u64,
        weight_offset: 0,
        output_offset: 0,
    };
    let launch = module
        .prepare_matmul_turing(LaunchConfig2D::new(
            (columns.div_ceil(16) as u32, rows.div_ceil(64) as u32),
            (256, 1),
            0,
        ))
        .map_err(cuda_error)?;
    module
        .matmul_turing(
            stream,
            &launch,
            input,
            weights,
            cuda_host::RowWidth::new(output, columns as u32),
            shape,
        )
        .map_err(cuda_error)
}
