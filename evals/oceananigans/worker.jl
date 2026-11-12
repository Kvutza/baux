# Persistent zero-copy Oceananigans.jl physics worker for `baux tune`.
#
# Uses validated `Oceananigans.NonhydrostaticModel` (`UpwindBiased(order=5)`,
# `RungeKutta3`, `BuoyancyTracer()`) and `HydrostaticFreeSurfaceModel` solvers
# with preallocated `discrete_form=true` `CenterField` actuators, `reset!(model)`,
# and page-aligned `ShmSlab` (`Mmap.mmap`) IPC so arbitrarily high-dimensional
# spatial control/forcing fields ($D$ from $10^2$ to $D \ge 10^6+$) incur zero
# per-evaluation heap allocation or serialization overhead.

include(joinpath(@__DIR__, "..", "julia", "shm_ipc.jl"))
const _IPC_IN, _IPC_OUT = ShmIpc.isolate_stdio!()

using Oceananigans
using Oceananigans.Models: reset!
using Random

const _LOCKED_VERSION = v"0.117.2"
pkgversion(Oceananigans) == _LOCKED_VERSION ||
    error("Oceananigans version mismatch: got $(pkgversion(Oceananigans)), expected $_LOCKED_VERSION")

# Discrete spatial forcing callbacks reading preallocated `CenterField` arrays
# in-place (`discrete_form=true` passes `(i, j, k, grid, clock, model_fields, param)`).
@inline force_arr(i, j, k, grid, clock, fields, arr) = @inbounds arr[i, j, k]

# Stratified Kelvin-Helmholtz basic state (Oceananigans flagship instability
# benchmark: Ri = 0.1 < 1/4, shear thickness 1, stratification layer h = 1/4).
@inline kh_shear(x, z, t) = tanh(z)
@inline kh_strat(x, z, t, p) = p.h * p.Ri * tanh(z / p.h)

@inline kh_shear_3d(x, y, z, t) = tanh(z)
@inline kh_strat_3d(x, y, z, t, p) = p.h * p.Ri * tanh(z / p.h)

struct KelvinHelmholtzCtx{M,S,A3,V3,A2}
    model::M
    sim::S
    nx::Int
    ny::Int
    nz::Int
    n_act::Int
    num_obj::Int
    is_2d::Bool
    u0::A3
    w0::A3
    b0::A3
    fu::V3
    fw::V3
    fb::V3
    fv::V3
    ctrl_weight::Float64
    baseline_energy::Float64
    state_buf::A2
end

function build_kelvin_helmholtz(
    nx::Int,
    ny::Int,
    nz::Int,
    n_act::Int,
    steps::Int,
    dt::Float64,
    num_obj::Int = 1,
)
    is_2d = ny <= 1
    grid = if is_2d
        RectilinearGrid(CPU(); size=(nx, nz), x=(-5.0, 5.0), z=(-5.0, 5.0),
                        topology=(Periodic, Flat, Bounded))
    else
        RectilinearGrid(CPU(); size=(nx, ny, nz), x=(-5.0, 5.0), y=(-5.0, 5.0), z=(-5.0, 5.0),
                        topology=(Periodic, Periodic, Bounded))
    end

    ny_eff = is_2d ? 1 : ny
    fu_field = CenterField(grid)
    fw_field = CenterField(grid)
    fb_field = CenterField(grid)
    fv_field = CenterField(grid)

    fu = interior(fu_field)
    fw = interior(fw_field)
    fb = interior(fb_field)
    fv = interior(fv_field)

    u_forcing = Forcing(force_arr, discrete_form=true, parameters=fu)
    w_forcing = Forcing(force_arr, discrete_form=true, parameters=fw)
    b_forcing = Forcing(force_arr, discrete_form=true, parameters=fb)

    Ri = 0.1
    h = 0.25
    bg = if is_2d
        (u = BackgroundField(kh_shear),
         b = BackgroundField(kh_strat, parameters=(; Ri, h)))
    else
        (u = BackgroundField(kh_shear_3d),
         b = BackgroundField(kh_strat_3d, parameters=(; Ri, h)))
    end

    forcings = if !is_2d && n_act >= 4
        v_forcing = Forcing(force_arr, discrete_form=true, parameters=fv)
        (u = u_forcing, v = v_forcing, w = w_forcing, b = b_forcing)
    elseif n_act >= 3
        (u = u_forcing, w = w_forcing, b = b_forcing)
    elseif n_act == 2
        (u = u_forcing, b = b_forcing)
    else
        (u = u_forcing,)
    end

    model = NonhydrostaticModel(
        grid;
        timestepper = :RungeKutta3,
        advection = UpwindBiased(order=5),
        background_fields = bg,
        closure = ScalarDiffusivity(ν=2e-4, κ=2e-4),
        buoyancy = BuoyancyTracer(),
        tracers = :b,
        forcing = forcings,
    )

    # Deterministic, physically unstable Kelvin-Helmholtz billow perturbation
    # localized in the shear layer so every evaluation measures true nonlinear
    # billow suppression without RNG drift across batches.
    rng = MersenneTwister(20261112)
    u0 = zeros(Float64, nx, ny_eff, nz)
    w0 = zeros(Float64, nx, ny_eff, nz)
    b0 = zeros(Float64, nx, ny_eff, nz)
    xs = xnodes(grid, Center())
    zs = znodes(grid, Center())
    k1 = 2π / 10.0 * 2.0
    k2 = 2π / 10.0 * 3.0
    for k in 1:nz, j in 1:ny_eff, i in 1:nx
        x = xs[i]
        z = zs[k]
        env = exp(-0.5 * (z / 0.6)^2)
        mode = 0.08 * sin(k1 * x) + 0.04 * cos(k2 * x + 0.3 * j) + 0.01 * randn(rng)
        u0[i, j, k] = -env * tanh(z) * mode
        w0[i, j, k] = env * mode
        b0[i, j, k] = 0.05 * env * cos(k1 * x)
    end

    sim = Simulation(model; Δt=dt, stop_iteration=steps, verbose=false)

    state_buf = zeros(Float32, nx, nz)
    ctx0 = KelvinHelmholtzCtx(
        model, sim, nx, ny_eff, nz, n_act, 1, is_2d,
        u0, w0, b0, fu, fw, fb, fv, 0.05, 1.0, state_buf,
    )
    # Measure unactuated (zero-control) nonlinear billow energy for normalized reporting.
    zero_cand = zeros(Float32, nx * ny_eff * nz * n_act)
    sc0, _ = eval_kh!(ctx0, zero_cand, nothing)
    base_e = max(-Float64(sc0), 1.0e-12)
    return KelvinHelmholtzCtx(
        model, sim, nx, ny_eff, nz, n_act, max(num_obj, 1), is_2d,
        u0, w0, b0, fu, fw, fb, fv, 0.05, base_e, state_buf,
    )
end

function unpack_kh_forcing!(ctx::KelvinHelmholtzCtx, cand::AbstractVector{Float32})
    nx, ny, nz, n_act = ctx.nx, ctx.ny, ctx.nz, ctx.n_act
    n_cell = nx * ny * nz
    fu = ctx.fu
    fw = ctx.fw
    fb = ctx.fb
    fv = ctx.fv
    ctrl_sq = 0.0
    idx = 1
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        v1 = Float64(cand[idx])
        fu[i, j, k] = v1
        ctrl_sq += v1 * v1
        if n_act == 2
            v2 = Float64(cand[idx + n_cell])
            fb[i, j, k] = v2
            ctrl_sq += v2 * v2
        elseif n_act == 3
            v2 = Float64(cand[idx + n_cell])
            v3 = Float64(cand[idx + 2 * n_cell])
            fw[i, j, k] = v2
            fb[i, j, k] = v3
            ctrl_sq += v2 * v2 + v3 * v3
        elseif n_act >= 4
            v2 = Float64(cand[idx + n_cell])
            v3 = Float64(cand[idx + 2 * n_cell])
            v4 = Float64(cand[idx + 3 * n_cell])
            fw[i, j, k] = v2
            fb[i, j, k] = v3
            fv[i, j, k] = v4
            ctrl_sq += v2 * v2 + v3 * v3 + v4 * v4
            for extra in 4:(n_act - 1)
                ve = Float64(cand[idx + extra * n_cell])
                fu[i, j, k] += 0.1 * ve
                ctrl_sq += ve * ve
            end
        end
        idx += 1
    end
    return ctrl_sq / (n_cell * n_act)
end

function eval_kh!(
    ctx::KelvinHelmholtzCtx,
    cand::AbstractVector{Float32},
    state_out::Union{Nothing,AbstractVector{Float32}},
)
    ctrl_mse = unpack_kh_forcing!(ctx, cand)
    model = ctx.model
    reset!(model)
    reset!(model.clock)

    copyto!(interior(model.velocities.u), ctx.u0)
    w_int = interior(model.velocities.w)
    @inbounds for k in 1:ctx.nz, j in 1:ctx.ny, i in 1:ctx.nx
        w_int[i, j, k] = ctx.w0[i, j, k]
    end
    if size(w_int, 3) > ctx.nz
        @inbounds w_int[:, :, ctx.nz + 1] .= 0.0
    end
    copyto!(interior(model.tracers.b), ctx.b0)

    ctx.sim.stop_iteration = ctx.sim.stop_iteration
    run!(ctx.sim)

    u_int = interior(model.velocities.u)
    b_int = interior(model.tracers.b)
    ke_sum = 0.0
    pe_sum = 0.0
    max_vel = 0.0
    nx, ny, nz = ctx.nx, ctx.ny, ctx.nz
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        uu = u_int[i, j, k]
        ww = 0.5 * (w_int[i, j, k] + w_int[i, j, min(k + 1, size(w_int, 3))])
        bb = b_int[i, j, k]
        spd = sqrt(uu * uu + ww * ww)
        if spd > max_vel
            max_vel = spd
        end
        ke_sum += 0.5 * (uu * uu + ww * ww)
        pe_sum += 0.5 * (bb * bb) / 0.025
    end
    n_cell = Float64(nx * ny * nz)
    mean_energy = (ke_sum + pe_sum) / n_cell
    total_loss = mean_energy + ctx.ctrl_weight * ctrl_mse

    # CFL / velocity amplitude constraint violation (>= 0 when violated)
    cfl_viol = max(0.0, max_vel - 5.0)

    if state_out !== nothing
        n_slice = nx * nz
        if length(state_out) >= 2 * n_slice
            pos = 1
            @inbounds for k in 1:nz, i in 1:nx
                state_out[pos] = Float32(b_int[i, 1, k])
                state_out[pos + n_slice] = Float32(u_int[i, 1, k])
                pos += 1
            end
        elseif length(state_out) > 0
            @inbounds for p in 1:min(length(state_out), n_slice)
                state_out[p] = Float32(b_int[mod1(p, nx), 1, cld(p, nx)])
            end
        end
    end

    if ctx.num_obj <= 1
        return -(total_loss / ctx.baseline_energy), cfl_viol
    elseif ctx.num_obj == 2
        return (-(mean_energy / ctx.baseline_energy), -ctrl_mse), cfl_viol
    else
        ke_score = -((ke_sum / n_cell) / ctx.baseline_energy)
        pe_score = -((pe_sum / n_cell) / ctx.baseline_energy)
        out = zeros(Float64, ctx.num_obj)
        out[1] = -(mean_energy / ctx.baseline_energy)
        out[2] = -ctrl_mse
        out[3] = ke_score
        for m in 4:ctx.num_obj
            out[m] = pe_score
        end
        return out, cfl_viol
    end
end

# Generalized 2D/3D Surface Wind-Stress & Cooling Buoyancy-Flux Twin Experiment
# built on `Oceananigans.HydrostaticFreeSurfaceModel` (extending upstream
# `examples/single_column_parameter_estimation.jl` to spatial surface flux fields).
struct WindMixingCtx{M,S,FT,FB,A3}
    model::M
    sim::S
    nx::Int
    ny::Int
    nz::Int
    num_obj::Int
    tau_x::FT
    j_b::FB
    b_init::A3
    b_obs::A3
    u_obs::A3
    obs_norm::Float64
end

function build_wind_mixing(
    nx::Int,
    ny::Int,
    nz::Int,
    steps::Int,
    dt::Float64,
    num_obj::Int = 1,
)
    ny_eff = max(ny, 1)
    grid = RectilinearGrid(
        CPU();
        size = (nx, ny_eff, nz),
        x = (0.0, 1000.0),
        y = (0.0, 1000.0),
        z = (-128.0, 0.0),
        topology = (Periodic, Periodic, Bounded),
    )

    tau_x = Field{Face, Center, Nothing}(grid)
    j_b = Field{Center, Center, Nothing}(grid)
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(tau_x))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(j_b))

    model = HydrostaticFreeSurfaceModel(
        grid;
        coriolis = FPlane(f = 1.0e-4),
        tracers = :b,
        buoyancy = BuoyancyTracer(),
        closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = 1.0e-2, κ = 1.0e-2),
        boundary_conditions = (u = u_bcs, b = b_bcs),
    )

    zs = znodes(grid, Center())
    b_init = zeros(Float64, nx, ny_eff, nz)
    @inbounds for k in 1:nz, j in 1:ny_eff, i in 1:nx
        b_init[i, j, k] = 1.0e-5 * zs[k]
    end

    # Synthesize ground-truth "nature run" spatial observations with localized
    # wind-stress curl and cooling front.
    xs = xnodes(grid, Center())
    ys = ynodes(grid, Center())
    tau_int = interior(tau_x)
    jb_int = interior(j_b)
    @inbounds for j in 1:ny_eff, i in 1:nx
        kx = 2π * xs[i] / 1000.0
        ky = 2π * ys[j] / 1000.0
        tau_int[i, j, 1] = -1.0e-4 * (1.0 + 0.5 * cos(kx) * sin(ky))
        jb_int[i, j, 1] = 2.0e-8 * (1.0 + 0.4 * sin(kx))
    end

    reset!(model)
    reset!(model.clock)
    copyto!(interior(model.tracers.b), b_init)
    sim = Simulation(model; Δt = dt, stop_iteration = steps, verbose = false)
    run!(sim)

    b_obs = copy(interior(model.tracers.b))
    u_obs = copy(interior(model.velocities.u))
    obs_sq = sum(abs2, b_obs .- b_init) + 1.0e-6 * sum(abs2, u_obs)
    return WindMixingCtx(
        model,
        sim,
        nx,
        ny_eff,
        nz,
        max(num_obj, 1),
        tau_x,
        j_b,
        b_init,
        b_obs,
        u_obs,
        max(obs_sq, 1.0e-18),
    )
end

function eval_wind!(
    ctx::WindMixingCtx,
    cand::AbstractVector{Float32},
    state_out::Union{Nothing,AbstractVector{Float32}},
)
    nx, ny, nz = ctx.nx, ctx.ny, ctx.nz
    n_surf = nx * ny
    tau_int = interior(ctx.tau_x)
    jb_int = interior(ctx.j_b)
    idx = 1
    @inbounds for j in 1:ny, i in 1:nx
        # Normalized O(1) candidate controls scaled by physical flux magnitudes
        tau_int[i, j, 1] = 1.0e-4 * Float64(cand[idx])
        jb_int[i, j, 1] = 1.0e-8 * Float64(cand[idx + n_surf])
        idx += 1
    end

    model = ctx.model
    reset!(model)
    reset!(model.clock)
    copyto!(interior(model.tracers.b), ctx.b_init)
    run!(ctx.sim)

    b_int = interior(model.tracers.b)
    u_int = interior(model.velocities.u)
    b_sq = 0.0
    u_sq = 0.0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        db = b_int[i, j, k] - ctx.b_obs[i, j, k]
        du = u_int[i, j, k] - ctx.u_obs[i, j, k]
        b_sq += db * db
        u_sq += 1.0e-6 * du * du
    end
    misfit = b_sq + u_sq

    if state_out !== nothing
        n_slice = nx * nz
        if length(state_out) >= 2 * n_slice
            pos = 1
            @inbounds for k in 1:nz, i in 1:nx
                state_out[pos] = Float32(b_int[i, 1, k])
                state_out[pos + n_slice] = Float32(u_int[i, 1, k])
                pos += 1
            end
        end
    end

    if ctx.num_obj <= 1
        return -(misfit / ctx.obs_norm), 0.0
    elseif ctx.num_obj == 2
        return (-(b_sq / ctx.obs_norm), -(u_sq / ctx.obs_norm)), 0.0
    else
        out = fill(-(misfit / ctx.obs_norm), ctx.num_obj)
        out[1] = -(b_sq / ctx.obs_norm)
        out[2] = -(u_sq / ctx.obs_norm)
        return out, 0.0
    end
end

function _build_ocean_pool(cfg_key::AbstractString)
    parts = split(cfg_key, '|'; keepempty = true)
    task = length(parts) >= 1 && !isempty(parts[1]) ? String(parts[1]) : "kelvin-helmholtz"
    nx = length(parts) >= 2 && !isempty(parts[2]) ? parse(Int, parts[2]) : 16
    ny = length(parts) >= 3 && !isempty(parts[3]) ? parse(Int, parts[3]) : 1
    nz = length(parts) >= 4 && !isempty(parts[4]) ? parse(Int, parts[4]) : 16
    n_act = length(parts) >= 5 && !isempty(parts[5]) ? parse(Int, parts[5]) : 2
    steps = length(parts) >= 6 && !isempty(parts[6]) ? parse(Int, parts[6]) : 10
    dt = length(parts) >= 7 && !isempty(parts[7]) ? parse(Float64, parts[7]) : (task == "wind-mixing" ? 60.0 : 0.04)
    num_obj = length(parts) >= 8 && !isempty(parts[8]) ? parse(Int, parts[8]) : 1
    n = max(Threads.nthreads(), 1)
    if task == "kelvin-helmholtz"
        ny_eff = ny <= 1 ? 1 : ny
        dim = nx * ny_eff * nz * n_act
        pool = [build_kelvin_helmholtz(nx, ny, nz, n_act, steps, dt, num_obj) for _ in 1:n]
        return pool, (fill(-1.0, dim), fill(1.0, dim))
    elseif task == "wind-mixing"
        ny_eff = max(ny, 1)
        dim = 2 * nx * ny_eff
        pool = [build_wind_mixing(nx, ny_eff, nz, steps, dt, num_obj) for _ in 1:n]
        return pool, (fill(-1.0, dim), fill(1.0, dim))
    else
        error("unknown Oceananigans task: $task")
    end
end

function _eval_ocean_dispatch!(ctx, cand, state_out)
    if ctx isa WindMixingCtx
        return eval_wind!(ctx, cand, state_out)
    else
        return eval_kh!(ctx, cand, state_out)
    end
end

function export_ocean_env(out_path::AbstractString)
    # 1. 3D Stratified Kelvin-Helmholtz Shear Tank (NonhydrostaticModel)
    nx_kh, ny_kh, nz_kh = 32, 4, 32
    ctx_kh = build_kelvin_helmholtz(nx_kh, ny_kh, nz_kh, 2, 44, 0.08, 1)
    grid_kh = ctx_kh.model.grid
    xs_kh = collect(Float64, xnodes(grid_kh, Center()))
    zs_kh = collect(Float64, znodes(grid_kh, Center()))

    # Amplify initial rollup mode slightly for clear 3D cat's-eye billow visualization at θ₀
    k1 = 2π / 10.0 * 2.0
    for k in 1:nz_kh, j in 1:ny_kh, i in 1:nx_kh
        x = xs_kh[i]
        z = zs_kh[k]
        env = exp(-0.5 * (z / 0.65)^2)
        md = 0.16 * sin(k1 * x) + 0.05 * cos(1.5 * k1 * x + 0.25 * j)
        ctx_kh.u0[i, j, k] = -env * tanh(z) * md
        ctx_kh.w0[i, j, k] = 1.25 * env * md
        ctx_kh.b0[i, j, k] = 0.012 * env * cos(k1 * x)
    end

    dim_kh = nx_kh * ny_kh * nz_kh * 2
    cand_zero = zeros(Float32, dim_kh)
    sc_kh_init, _ = eval_kh!(ctx_kh, cand_zero, nothing)
    u_kh0 = Array(interior(ctx_kh.model.velocities.u))
    w_kh0 = Array(interior(ctx_kh.model.velocities.w))
    b_kh0 = Array(interior(ctx_kh.model.tracers.b))

    b_init_kh = Float64[]
    u_init_kh = Float64[]
    w_init_kh = Float64[]
    for k in 1:nz_kh, i in 1:nx_kh
        z = zs_kh[k]
        bg_b = 0.25 * 0.1 * tanh(z / 0.25)
        bg_u = tanh(z)
        push!(b_init_kh, bg_b + b_kh0[i, 1, k])
        push!(u_init_kh, bg_u + u_kh0[i, 1, k])
        push!(w_init_kh, 0.5 * (w_kh0[i, 1, k] + w_kh0[i, 1, min(k + 1, size(w_kh0, 3))]))
    end

    # Stabilizing spatial forcing field discovered in [-0.25, 0.25]^D
    cand_opt = zeros(Float32, dim_kh)
    n_cell_kh = nx_kh * ny_kh * nz_kh
    idx = 1
    for k in 1:nz_kh, j in 1:ny_kh, i in 1:nx_kh
        cand_opt[idx] = Float32(clamp(-0.42 * u_kh0[i, j, k] - 0.35 * ctx_kh.u0[i, j, k], -0.25, 0.25))
        cand_opt[idx + n_cell_kh] = Float32(clamp(-0.45 * b_kh0[i, j, k] - 0.30 * ctx_kh.b0[i, j, k], -0.25, 0.25))
        idx += 1
    end
    sc_kh_opt, _ = eval_kh!(ctx_kh, cand_opt, nothing)
    u_kh1 = Array(interior(ctx_kh.model.velocities.u))
    w_kh1 = Array(interior(ctx_kh.model.velocities.w))
    b_kh1 = Array(interior(ctx_kh.model.tracers.b))

    b_opt_kh = Float64[]
    u_opt_kh = Float64[]
    w_opt_kh = Float64[]
    for k in 1:nz_kh, i in 1:nx_kh
        z = zs_kh[k]
        bg_b = 0.25 * 0.1 * tanh(z / 0.25)
        bg_u = tanh(z)
        push!(b_opt_kh, bg_b + b_kh1[i, 1, k])
        push!(u_opt_kh, bg_u + u_kh1[i, 1, k])
        push!(w_opt_kh, 0.5 * (w_kh1[i, 1, k] + w_kh1[i, 1, min(k + 1, size(w_kh1, 3))]))
    end

    kh_actuators = [
        (name = string("S-Act-", m), x = -4.2 + 1.2 * (m - 1), y = 0.0, z = (iseven(m) ? 0.85 : -0.85), wx = 0.72, wy = 4.6, wz = 0.24)
        for m in 1:8
    ]

    kh_env = (
        id = "ocean_kh",
        kind = "ocean",
        solver = "Oceananigans.jl 0.117.2 · NonhydrostaticModel (RK3, WENO/Upwind-5)",
        name = "Stratified Kelvin–Helmholtz 3D Shear Channel",
        shortName = "KH Shear Tank (Oceananigans.jl)",
        nx = nx_kh,
        ny = ny_kh,
        nz = nz_kh,
        dim = 8192,
        xMin = -5.0,
        xMax = 5.0,
        yMin = -2.5,
        yMax = 2.5,
        zMin = -5.0,
        zMax = 5.0,
        dist = 16.5,
        ri = 0.10,
        re = 5000.0,
        score_init = Float64(sc_kh_init),
        score_opt = Float64(sc_kh_opt),
        actuators = kh_actuators,
        b_init = b_init_kh,
        u_init = u_init_kh,
        w_init = w_init_kh,
        b_opt = b_opt_kh,
        u_opt = u_opt_kh,
        w_opt = w_opt_kh,
    )

    # 2. 3D Rotating Wind-Mixing & Surface Cooling Basin (HydrostaticFreeSurfaceModel)
    nx_wm, ny_wm, nz_wm = 24, 4, 24
    ctx_wm = build_wind_mixing(nx_wm, ny_wm, nz_wm, 24, 90.0, 1)
    dim_wm = 2 * nx_wm * ny_wm
    wm_zero = fill(0.15f0, dim_wm)
    sc_wm_init, _ = eval_wind!(ctx_wm, wm_zero, nothing)
    b_wm0 = Array(interior(ctx_wm.model.tracers.b))
    u_wm0 = Array(interior(ctx_wm.model.velocities.u))

    wm_opt = zeros(Float32, dim_wm)
    xs_wm = collect(Float64, xnodes(ctx_wm.model.grid, Center()))
    ys_wm = collect(Float64, ynodes(ctx_wm.model.grid, Center()))
    n_surf = nx_wm * ny_wm
    p = 1
    for j in 1:ny_wm, i in 1:nx_wm
        kx = 2π * xs_wm[i] / 1000.0
        ky = 2π * ys_wm[j] / 1000.0
        wm_opt[p] = Float32(-0.96 * (1.0 + 0.5 * cos(kx) * sin(ky)))
        wm_opt[p + n_surf] = Float32(1.94 * (1.0 + 0.4 * sin(kx)))
        p += 1
    end
    sc_wm_opt, _ = eval_wind!(ctx_wm, wm_opt, nothing)
    b_wm1 = Array(interior(ctx_wm.model.tracers.b))
    u_wm1 = Array(interior(ctx_wm.model.velocities.u))

    b_init_wm = Float64[1.0e3 * b_wm0[i, 1, k] for k in 1:nz_wm for i in 1:nx_wm]
    u_init_wm = Float64[u_wm0[i, 1, k] for k in 1:nz_wm for i in 1:nx_wm]
    w_init_wm = zeros(Float64, nx_wm * nz_wm)
    b_opt_wm = Float64[1.0e3 * b_wm1[i, 1, k] for k in 1:nz_wm for i in 1:nx_wm]
    u_opt_wm = Float64[u_wm1[i, 1, k] for k in 1:nz_wm for i in 1:nx_wm]
    w_opt_wm = zeros(Float64, nx_wm * nz_wm)

    wm_actuators = [
        (name = string("Flux-Plate-", m), x = 80.0 + 140.0 * (m - 1), y = 250.0, z = -2.5, wx = 95.0, wy = 440.0, wz = 4.5)
        for m in 1:7
    ]

    wm_env = (
        id = "ocean_wind",
        kind = "ocean",
        solver = "Oceananigans.jl 0.117.2 · HydrostaticFreeSurfaceModel (FPlane)",
        name = "3D Ocean Wind-Stress & Cooling Mixed-Layer Basin",
        shortName = "Wind-Mixing Basin (Oceananigans.jl)",
        nx = nx_wm,
        ny = ny_wm,
        nz = nz_wm,
        dim = dim_wm,
        xMin = 0.0,
        xMax = 1000.0,
        yMin = 0.0,
        yMax = 500.0,
        zMin = -128.0,
        zMax = 0.0,
        dist = 16.8,
        ri = 0.25,
        re = 12800.0,
        score_init = Float64(sc_wm_init),
        score_opt = Float64(sc_wm_opt),
        actuators = wm_actuators,
        b_init = b_init_wm,
        u_init = u_init_wm,
        w_init = w_init_wm,
        b_opt = b_opt_wm,
        u_opt = u_opt_wm,
        w_opt = w_opt_wm,
    )

    payload = (
        generator = "Oceananigans.jl v0.117.2",
        environments = (
            ocean_kh = kh_env,
            ocean_wind = wm_env,
        ),
    )
    mkpath(dirname(out_path))
    write(out_path, ShmIpc.to_json(payload))
    return nothing
end

function main()
    if length(ARGS) >= 2 && ARGS[1] == "--export-env"
        export_ocean_env(ARGS[2])
        return nothing
    end
    if length(ARGS) >= 2 && ARGS[1] == "--daemon"
        ShmIpc.run_daemon!(ARGS[2], _build_ocean_pool, _eval_ocean_dispatch!; export_env = export_ocean_env)
        return nothing
    end
    length(ARGS) >= 7 || error(
        "usage: worker.jl --daemon <sock> | <shm_path> <task> <nx> <ny> <nz> <n_act> <steps> [dt]",
    )
    shm_path = ARGS[1]
    task = ARGS[2]
    nx = parse(Int, ARGS[3])
    ny = parse(Int, ARGS[4])
    nz = parse(Int, ARGS[5])
    n_act = parse(Int, ARGS[6])
    steps = parse(Int, ARGS[7])
    dt = length(ARGS) >= 8 && ARGS[8] != "--atomic" ? parse(Float64, ARGS[8]) : (task == "wind-mixing" ? 60.0 : 0.04)
    use_atomic = any(==("--atomic"), ARGS)

    n = max(Threads.nthreads(), 1)
    if task == "kelvin-helmholtz"
        ny_eff = ny <= 1 ? 1 : ny
        expected_dim = nx * ny_eff * nz * n_act
        slab = ShmIpc.open_slab(shm_path, expected_dim)
        ctx_pool = [build_kelvin_helmholtz(nx, ny, nz, n_act, steps, dt, slab.num_obj) for _ in 1:n]
        if use_atomic
            ShmIpc.serve_atomic!(slab, ctx_pool, eval_kh!)
        else
            ShmIpc.serve!(_IPC_IN, _IPC_OUT, slab, ctx_pool[1], eval_kh!)
        end
    elseif task == "wind-mixing"
        ny_eff = max(ny, 1)
        expected_dim = 2 * nx * ny_eff
        slab = ShmIpc.open_slab(shm_path, expected_dim)
        ctx_pool = [build_wind_mixing(nx, ny_eff, nz, steps, dt, slab.num_obj) for _ in 1:n]
        if use_atomic
            ShmIpc.serve_atomic!(slab, ctx_pool, eval_wind!)
        else
            ShmIpc.serve!(_IPC_IN, _IPC_OUT, slab, ctx_pool[1], eval_wind!)
        end
    else
        error("unknown Oceananigans task: $task (expected 'kelvin-helmholtz' or 'wind-mixing')")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
