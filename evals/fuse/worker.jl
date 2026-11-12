# Persistent zero-copy FUSE.jl whole-facility & stationary-plasma tokamak worker
# for `baux tune`.
#
# Uses General Atomics' official `FUSE.jl` (`v1.2.1`) `case_parameters`,
# `SimulationParameters.parameters_from_opt!`, `IMAS.ObjectiveFunctionsLibrary`,
# and `IMAS.ConstraintFunctionsLibrary` over page-aligned `ShmSlab` (`Mmap.mmap`)
# IPC with zero HDF5/CSV disk churn per rollout and clean solver exception
# isolation.

include(joinpath(@__DIR__, "..", "julia", "shm_ipc.jl"))
const _IPC_IN, _IPC_OUT = ShmIpc.isolate_stdio!()

using FUSE
const IMAS = FUSE.IMAS

const _LOCKED_VERSION = v"1.2.1"
pkgversion(FUSE) == _LOCKED_VERSION ||
    error("FUSE version mismatch: got $(pkgversion(FUSE)), expected $_LOCKED_VERSION")

IMAS.update_ObjectiveFunctionsLibrary!()
IMAS.update_ConstraintFunctionsLibrary!()

function load_case(case_name::AbstractString)
    up = uppercase(case_name)
    if up == "ARC"
        return FUSE.case_parameters(:ARC)
    elseif up == "FPP"
        return FUSE.case_parameters(:FPP)
    elseif up == "SPARC"
        return FUSE.case_parameters(:SPARC; init_from = :scalars)
    elseif up == "ITER"
        return FUSE.case_parameters(:ITER; init_from = :scalars, boundary_from = :scalars)
    elseif up == "D3D" || up == "D3D_HMODE"
        return FUSE.case_parameters(:D3D, :H_mode)
    elseif up == "KDEMO"
        return FUSE.case_parameters(:KDEMO)
    else
        return FUSE.case_parameters(Symbol(case_name))
    end
end

function attach_actuators!(ini, act, case_name::AbstractString)
    up = uppercase(case_name)
    # 1. Core equilibrium & geometry actuators (8-12 continuous knobs depending on HCD)
    if !ismissing(ini.equilibrium, :B0)
        b0 = Float64(ini.equilibrium.B0)
        lo, hi = b0 < 0 ? (1.15 * b0, 0.85 * b0) : (0.85 * b0, 1.15 * b0)
        ini.equilibrium.B0 = ini.equilibrium.B0 ↔ [min(lo, hi), max(lo, hi)]
    end
    if !ismissing(ini.equilibrium, :R0) && ini.equilibrium.boundary_from == :scalars
        r0 = Float64(ini.equilibrium.R0)
        ini.equilibrium.R0 = ini.equilibrium.R0 ↔ [0.90 * r0, 1.12 * r0]
    end
    if !ismissing(ini.equilibrium, :κ) && ini.equilibrium.boundary_from == :scalars
        kap = Float64(ini.equilibrium.κ)
        ini.equilibrium.κ = ini.equilibrium.κ ↔ [max(1.3, 0.90 * kap), 1.08 * kap]
    end
    if !ismissing(ini.equilibrium, :δ) && ini.equilibrium.boundary_from == :scalars
        del = Float64(ini.equilibrium.δ)
        ini.equilibrium.δ = ini.equilibrium.δ ↔ [max(0.20, del - 0.12), min(0.75, del + 0.12)]
    end
    if !ismissing(ini.equilibrium, :ip)
        ip0 = Float64(ini.equilibrium.ip)
        ini.equilibrium.ip = ini.equilibrium.ip ↔ [0.80 * ip0, 1.20 * ip0]
    end

    # 2. Core kinetic & impurity profile actuators
    if !ismissing(ini.core_profiles, :ne_value)
        ne0 = Float64(ini.core_profiles.ne_value)
        ini.core_profiles.ne_value = ini.core_profiles.ne_value ↔ [0.70 * ne0, 1.15 * ne0]
    end
    if !ismissing(ini.core_profiles, :Te_core)
        te0 = Float64(ini.core_profiles.Te_core)
        ini.core_profiles.Te_core = ini.core_profiles.Te_core ↔ [0.70 * te0, 1.35 * te0]
    end
    if !ismissing(ini.core_profiles, :zeff)
        z0 = Float64(ini.core_profiles.zeff)
        ini.core_profiles.zeff = ini.core_profiles.zeff ↔ [max(1.15, 0.80 * z0), min(3.2, 1.40 * z0)]
    end

    # 3. Auxiliary heating & current-drive actuators present on this machine
    for k in eachindex(ini.ic_antenna)
        if !ismissing(ini.ic_antenna[k], :power_launched)
            p0 = max(Float64(ini.ic_antenna[k].power_launched), 1.0e6)
            ini.ic_antenna[k].power_launched = ini.ic_antenna[k].power_launched ↔ [0.25 * p0, 1.50 * p0]
        end
    end
    for k in eachindex(ini.ec_launcher)
        if !ismissing(ini.ec_launcher[k], :power_launched)
            p0 = max(Float64(ini.ec_launcher[k].power_launched), 0.5e6)
            ini.ec_launcher[k].power_launched = ini.ec_launcher[k].power_launched ↔ [0.20 * p0, 1.50 * p0]
        end
    end
    for k in eachindex(ini.nb_unit)
        if !ismissing(ini.nb_unit[k], :power_launched)
            p0 = max(Float64(ini.nb_unit[k].power_launched), 0.5e6)
            ini.nb_unit[k].power_launched = ini.nb_unit[k].power_launched ↔ [0.20 * p0, 1.50 * p0]
        end
    end

    # Ensure engineering sizing checks report smooth constraint violations rather than throwing
    act.ActorHFSsizing.error_on_performance = false
    act.ActorHFSsizing.error_on_technology = false
    if up == "SPARC"
        act.ActorVerticalStability.model = false
    end
    return nothing
end

mutable struct FuseCtx
    base_ini::FUSE.ParametersAllInits
    base_act::FUSE.ParametersAllActors
    workflow::String
    objective_fns::Vector{IMAS.ObjectiveFunction}
    constraints::Vector{IMAS.ConstraintFunction}
    bnds::Matrix{Float64}
    x_buf::Vector{Float64}
    s_buf::Vector{Float64}
    warmed::Bool
    penalty_weight::Float64
    eval_count::Int
end

function build_fuse_ctx(
    case_name::AbstractString,
    workflow::AbstractString,
    obj_csv::AbstractString,
    penalty_weight::Float64,
    cstr_csv::AbstractString = "",
)
    ini, act = load_case(case_name)
    attach_actuators!(ini, act, case_name)
    opts = FUSE.SimulationParameters.opt_parameters(ini)
    isempty(opts) && error("FUSE case $case_name produced 0 opt_parameters")

    OFL = IMAS.ObjectiveFunctionsLibrary
    CFL = IMAS.ConstraintFunctionsLibrary

    # Register max_βn objective if requested and not yet in stock library
    if !haskey(OFL, :max_βn)
        IMAS.ObjectiveFunction(
            :max_βn,
            "-",
            dd -> dd.equilibrium.time_slice[].global_quantities.beta_normal,
            Inf,
        )
    end

    obj_fns = IMAS.ObjectiveFunction[]
    for raw in split(obj_csv, ',')
        s = strip(raw)
        isempty(s) && continue
        obj_sym = Symbol(s)
        haskey(OFL, obj_sym) || error(
            "unknown IMAS ObjectiveFunction :$obj_sym; available: $(collect(keys(OFL)))",
        )
        push!(obj_fns, OFL[obj_sym])
    end
    isempty(obj_fns) && error("FUSE worker requires at least one IMAS ObjectiveFunction")

    cstrs = if !isempty(strip(cstr_csv))
        out = IMAS.ConstraintFunction[]
        for raw in split(cstr_csv, ',')
            s = strip(raw)
            isempty(s) && continue
            sym = Symbol(s)
            haskey(CFL, sym) || error(
                "unknown IMAS ConstraintFunction :$sym; available: $(collect(keys(CFL)))",
            )
            push!(out, CFL[sym])
        end
        out
    elseif workflow == "whole-facility"
        IMAS.ConstraintFunction[
            CFL[:power_electric_net],
            CFL[:max_tf_coil_j],
            CFL[:max_oh_coil_j],
            CFL[:max_tf_coil_stress],
            CFL[:max_oh_coil_stress],
        ]
    elseif workflow == "stationary-plasma"
        IMAS.ConstraintFunction[
            CFL[:min_q95],
            CFL[:max_beta_normal],
        ]
    elseif workflow == "equilibrium" || workflow == "pf-control"
        IMAS.ConstraintFunction[
            CFL[:min_q95],
        ]
    elseif workflow == "dynamic-plasma"
        IMAS.ConstraintFunction[
            CFL[:min_q95],
            CFL[:max_beta_normal],
        ]
    else
        error("unknown FUSE workflow '$workflow' (expected 'whole-facility', 'stationary-plasma', 'equilibrium', 'dynamic-plasma', or 'pf-control')")
    end

    bnds = FUSE.SimulationParameters.float_bounds(opts)
    return FuseCtx(
        ini,
        act,
        String(workflow),
        obj_fns,
        cstrs,
        bnds,
        zeros(Float64, length(opts)),
        zeros(Float64, length(obj_fns)),
        false,
        penalty_weight,
        0,
    )
end

function write_dd_state!(dd::IMAS.DD, out::AbstractVector{Float32})
    fill!(out, 0.0f0)
    length(out) < 8 && return nothing
    eqt = dd.equilibrium.time_slice[]
    out[1] = Float32(eqt.boundary.geometric_axis.r)
    out[2] = Float32(eqt.global_quantities.ip / 1.0e6)
    out[3] = Float32(eqt.global_quantities.q_95)
    out[4] = Float32(eqt.global_quantities.beta_normal)
    cp1d = dd.core_profiles.profiles_1d[]
    out[5] = Float32(IMAS.fusion_power(cp1d) / 1.0e6)
    out[6] = !isempty(dd.balance_of_plant) && !ismissing(dd.balance_of_plant, :power_electric_net) ?
        Float32(IMAS.@ddtime(dd.balance_of_plant.power_electric_net) / 1.0e6) : 0.0f0
    out[7] = !isempty(dd.costing) && !ismissing(dd.costing.cost_direct_capital, :cost) ?
        Float32(dd.costing.cost_direct_capital.cost / 1.0e3) : 0.0f0
    out[8] = !isempty(dd.build) && !ismissing(dd.build.oh, :flattop_duration) ?
        Float32(dd.build.oh.flattop_duration / 3600.0) : 0.0f0

    # Pack radial profiles (Te [keV] and ne [10^20 m^-3]) if space permits
    te = cp1d.electrons.temperature
    ne = cp1d.electrons.density_thermal
    n_prof = min(length(te), div(length(out) - 8, 2))
    @inbounds for i in 1:n_prof
        out[8 + i] = Float32(te[i] / 1.0e3)
        out[8 + n_prof + i] = Float32(ne[i] / 1.0e20)
    end
    return nothing
end

function eval_fuse!(
    ctx::FuseCtx,
    cand::AbstractVector{Float32},
    state_out::Union{Nothing,AbstractVector{Float32}},
)
    x = ctx.x_buf
    bnds = ctx.bnds
    if !ctx.warmed
        # During pre-READY JIT warmup, `ShmIpc.serve!` passes an all-zero vector;
        # evaluate at the physical domain midpoint so FUSE compiles every actor cleanly!
        ctx.warmed = true
        @inbounds for j in eachindex(x)
            x[j] = 0.5 * (bnds[1, j] + bnds[2, j])
        end
    else
        @inbounds for j in eachindex(x)
            x[j] = clamp(Float64(cand[j]), bnds[1, j], bnds[2, j])
        end
    end

    ini = deepcopy(ctx.base_ini)
    act = deepcopy(ctx.base_act)
    FUSE.SimulationParameters.parameters_from_opt!(ini, x; clip = true)

    dd = IMAS.dd()
    FUSE.actor_logging(dd, false)
    FUSE.init(dd, ini, act)
    ini.general.dd = missing

    if ctx.workflow == "whole-facility"
        FUSE.ActorWholeFacility(dd, act)
    elseif ctx.workflow == "stationary-plasma"
        FUSE.ActorStationaryPlasma(dd, act)
        FUSE.ActorPlasmaLimits(dd, act)
    elseif ctx.workflow == "dynamic-plasma"
        FUSE.ActorDynamicPlasma(dd, act; Δt = 0.25, Nt = 2, evolve_pf_active = true)
        FUSE.ActorPlasmaLimits(dd, act)
    elseif ctx.workflow == "pf-control"
        FUSE.ActorEquilibrium(dd, act; model = :TEQUILA, ip_from = :pulse_schedule)
        if isempty(dd.pf_active.coil)
            FUSE.ActorPFdesign(dd, act; model = :uniform)
        end
        FUSE.ActorPFactive(dd, act; update_equilibrium = true)
        FUSE.ActorPlasmaLimits(dd, act)
    else
        FUSE.ActorEquilibrium(dd, act; ip_from = :pulse_schedule)
    end

    # IMAS ObjectiveFunction always returns a minimization quantity `f_min`;
    # BAUX maximizes `score_m = -f_min_m - penalty * sum(g_i^2)`.
    viol_l1 = 0.0
    viol_sq = 0.0
    for cfn in ctx.constraints
        v = try
            FUSE.nan2inf(cfn(dd))
        catch
            0.0
        end
        if isfinite(v) && v > 0.0
            viol_l1 += v
            viol_sq += v * v
        elseif !isfinite(v)
            viol_l1 += 100.0
            viol_sq += 1.0e4
        end
    end

    if state_out !== nothing
        write_dd_state!(dd, state_out)
    end

    ctx.eval_count += 1

    if length(ctx.objective_fns) == 1
        f_min = FUSE.nan2inf(ctx.objective_fns[1](dd))
        if !isfinite(f_min)
            return -500.0, max(viol_l1, 1.0e6)
        end
        return -f_min - ctx.penalty_weight * viol_sq, viol_l1
    end

    any_bad = false
    s_out = ctx.s_buf
    @inbounds for m in eachindex(ctx.objective_fns)
        fm = try
            FUSE.nan2inf(ctx.objective_fns[m](dd))
        catch
            Inf
        end
        if isfinite(fm)
            s_out[m] = -fm - ctx.penalty_weight * viol_sq
        else
            s_out[m] = -500.0
            any_bad = true
        end
    end
    return s_out, (any_bad ? max(viol_l1, 1.0e6) : viol_l1)
end

function dump_bounds(case_name::AbstractString)
    ini, act = load_case(case_name)
    attach_actuators!(ini, act, case_name)
    opts = FUSE.SimulationParameters.opt_parameters(ini)
    bnds = FUSE.SimulationParameters.float_bounds(opts)
    # Emit compact tab-separated specification on `_IPC_OUT` (`fd 1`) for `--bounds` query
    println(_IPC_OUT, "FUSE_BOUNDS\t", size(bnds, 2))
    for j in 1:size(bnds, 2)
        println(_IPC_OUT, bnds[1, j], "\t", bnds[2, j])
    end
    flush(_IPC_OUT)
    return nothing
end

function _build_fuse_pool(cfg_key::AbstractString)
    parts = split(cfg_key, '|'; keepempty = true)
    case_name = length(parts) >= 1 && !isempty(parts[1]) ? String(parts[1]) : "SPARC"
    workflow = length(parts) >= 2 && !isempty(parts[2]) ? String(parts[2]) : "stationary-plasma"
    obj_name = length(parts) >= 3 && !isempty(parts[3]) ? String(parts[3]) : "max_fusion"
    penalty = length(parts) >= 4 && !isempty(parts[4]) ? parse(Float64, parts[4]) : 25.0
    cstr_csv = length(parts) >= 5 ? String(parts[5]) : ""
    n = max(Threads.nthreads(), 1)
    ctx1 = build_fuse_ctx(case_name, workflow, obj_name, penalty, cstr_csv)
    try
        eval_fuse!(ctx1, zeros(Float32, length(ctx1.x_buf)), nothing)
    catch
        ctx1.warmed = true
    end
    pool = [i == 1 ? ctx1 : deepcopy(ctx1) for i in 1:n]
    return pool, (ctx1.bnds[1, :], ctx1.bnds[2, :])
end

function _clean_poly_rz(rs::AbstractVector, zs::AbstractVector)
    pts = Vector{Float64}[]
    n = min(length(rs), length(zs))
    for i in 1:n
        r = Float64(rs[i])
        z = Float64(zs[i])
        (isfinite(r) && isfinite(z)) || continue
        if isempty(pts) || hypot(r - pts[end][1], z - pts[end][2]) > 1.0e-5
            push!(pts, [r, z])
        end
    end
    if length(pts) >= 3 && hypot(pts[end][1] - pts[1][1], pts[end][2] - pts[1][2]) <= 1.0e-4
        pop!(pts)
    end
    length(pts) < 3 && return pts
    area2 = 0.0
    m = length(pts)
    for i in 1:m
        p1 = pts[i]
        p2 = pts[mod1(i + 1, m)]
        area2 += p1[1] * p2[2] - p2[1] * p1[2]
    end
    if area2 < 0.0
        reverse!(pts)
    end
    best_i = 1
    best_score = -Inf
    for i in 1:m
        sc = pts[i][1] - 0.08 * abs(pts[i][2])
        if sc > best_score
            best_score = sc
            best_i = i
        end
    end
    return best_i == 1 ? pts : vcat(pts[best_i:end], pts[1:(best_i - 1)])
end

function _resample_closed_rz(pts::Vector{Vector{Float64}}, n_out::Int)
    m = length(pts)
    m < 3 && return pts
    cum = zeros(Float64, m + 1)
    for i in 1:m
        p1 = pts[i]
        p2 = pts[mod1(i + 1, m)]
        cum[i + 1] = cum[i] + hypot(p2[1] - p1[1], p2[2] - p1[2])
    end
    perim = cum[end]
    perim <= 1.0e-9 && return pts
    out = Vector{Vector{Float64}}(undef, n_out)
    seg = 1
    for k in 1:n_out
        s = perim * (k - 1) / n_out
        while seg < m && cum[seg + 1] < s
            seg += 1
        end
        ds = max(cum[seg + 1] - cum[seg], 1.0e-12)
        t = clamp((s - cum[seg]) / ds, 0.0, 1.0)
        p1 = pts[seg]
        p2 = pts[mod1(seg + 1, m)]
        out[k] = [round(p1[1] + t * (p2[1] - p1[1]); digits = 4), round(p1[2] + t * (p2[2] - p1[2]); digits = 4)]
    end
    return out
end

function _offset_poly_rz(pts::Vector{Vector{Float64}}, r0::Float64, dr::Float64)
    out = Vector{Vector{Float64}}(undef, length(pts))
    for i in eachindex(pts)
        r, z = pts[i][1], pts[i][2]
        vr = r - r0
        vz = z
        nrm = max(hypot(vr, vz), 1.0e-6)
        out[i] = [round(max(0.08, r + dr * (vr / nrm)); digits = 4), round(z + dr * (vz / nrm); digits = 4)]
    end
    return out
end

function _find_layer_poly(dd::IMAS.DD, names::Vector{String})
    for target in names
        for lyr in dd.build.layer
            lname = lowercase(String(lyr.name))
            if lname == target && !ismissing(lyr, :outline) && !ismissing(lyr.outline, :r) && length(lyr.outline.r) >= 4
                return _clean_poly_rz(lyr.outline.r, lyr.outline.z)
            end
        end
    end
    return Vector{Float64}[]
end

function _extract_one_tokamak(
    id_str::String,
    full_name::String,
    short_name::String,
    ini::FUSE.ParametersAllInits,
    act::FUSE.ParametersAllActors;
    run_pf::Bool = false,
)
    dd = IMAS.dd()
    FUSE.actor_logging(dd, false)
    FUSE.init(dd, ini, act)
    if isempty(dd.equilibrium.time_slice) || ismissing(dd.equilibrium.time_slice[].boundary.outline, :r)
        FUSE.ActorEquilibrium(dd, act; model = :TEQUILA, ip_from = :pulse_schedule)
    end
    if run_pf
        try
            if isempty(dd.pf_active.coil)
                FUSE.ActorPFdesign(dd, act; model = :uniform)
            end
            FUSE.ActorPFactive(dd, act; update_equilibrium = true)
        catch
        end
    end

    eqt = dd.equilibrium.time_slice[]
    lcfs_raw = _clean_poly_rz(eqt.boundary.outline.r, eqt.boundary.outline.z)
    lcfs_pts = _resample_closed_rz(lcfs_raw, 72)
    r_vals = [p[1] for p in lcfs_pts]
    z_vals = [p[2] for p in lcfs_pts]

    r0 = (!ismissing(eqt.boundary, :geometric_axis) && !ismissing(eqt.boundary.geometric_axis, :r)) ?
        Float64(eqt.boundary.geometric_axis.r) : 0.5 * (minimum(r_vals) + maximum(r_vals))
    z0 = (!ismissing(eqt.boundary, :geometric_axis) && !ismissing(eqt.boundary.geometric_axis, :z)) ?
        Float64(eqt.boundary.geometric_axis.z) : 0.5 * (minimum(z_vals) + maximum(z_vals))
    amin = !ismissing(eqt.boundary, :minor_radius) ?
        Float64(eqt.boundary.minor_radius) : 0.5 * (maximum(r_vals) - minimum(r_vals))
    b0 = (!ismissing(dd.equilibrium.vacuum_toroidal_field, :b0) && !isempty(dd.equilibrium.vacuum_toroidal_field.b0)) ?
        abs(Float64(dd.equilibrium.vacuum_toroidal_field.b0[1])) : 5.3
    ip_ma = !ismissing(eqt.global_quantities, :ip) ? abs(Float64(eqt.global_quantities.ip)) / 1.0e6 : 5.0
    q95 = (!ismissing(eqt.global_quantities, :q_95) && isfinite(eqt.global_quantities.q_95)) ?
        abs(Float64(eqt.global_quantities.q_95)) : 3.5
    beta_n = (!ismissing(eqt.global_quantities, :beta_normal) && isfinite(eqt.global_quantities.beta_normal)) ?
        abs(Float64(eqt.global_quantities.beta_normal)) : 1.8
    p_fus = if !isempty(dd.core_profiles.profiles_1d)
        try
            Float64(IMAS.fusion_power(dd.core_profiles.profiles_1d[])) / 1.0e6
        catch
            0.0
        end
    else
        0.0
    end

    # First wall: prefer detailed ODS limiter contour when present, else build layer
    fw_raw = if !isempty(dd.wall.description_2d) && !isempty(dd.wall.description_2d[1].limiter.unit)
        u1 = dd.wall.description_2d[1].limiter.unit[1].outline
        _clean_poly_rz(u1.r, u1.z)
    else
        Vector{Float64}[]
    end
    if length(fw_raw) < 6
        fw_raw = _find_layer_poly(dd, ["hfs first wall", "hfs wall", "lfs first wall", "lfs wall"])
    end
    fw_pts = length(fw_raw) > 96 ? _resample_closed_rz(fw_raw, 96) : fw_raw

    vv_out_raw = _find_layer_poly(dd, ["hfs vacuum vessel outer", "lfs vacuum vessel outer", "hfs vessel"])
    vv_in_raw = _find_layer_poly(dd, ["hfs vacuum vessel inner", "lfs vacuum vessel inner", "hfs wall"])
    vv_out, vv_in = if length(vv_out_raw) >= 6 && length(vv_in_raw) >= 6
        (_resample_closed_rz(vv_out_raw, 80), _resample_closed_rz(vv_in_raw, 80))
    else
        base_fw = _resample_closed_rz(fw_pts, 80)
        thk = max(0.08 * amin, 0.06)
        (_offset_poly_rz(base_fw, r0, 2.0 * thk), _offset_poly_rz(base_fw, r0, 0.9 * thk))
    end

    tf_out_raw = _find_layer_poly(dd, ["hfs tf", "lfs tf"])
    tf_in_raw = _find_layer_poly(
        dd,
        ["hfs gap outer wall tf", "hfs gap tf blanket tank", "hfs gap tf shield", "hfs gap oh coils"],
    )
    tf_out, tf_in = if length(tf_out_raw) >= 6 && length(tf_in_raw) >= 6
        (_resample_closed_rz(tf_out_raw, 64), _resample_closed_rz(tf_in_raw, 64))
    else
        (Vector{Float64}[], Vector{Float64}[])
    end

    n_tf = if !ismissing(dd.build, :tf) && !ismissing(dd.build.tf, :coils_n)
        Int(dd.build.tf.coils_n)
    else
        18
    end

    ax_r = (!ismissing(eqt.global_quantities, :magnetic_axis) && !ismissing(eqt.global_quantities.magnetic_axis, :r)) ?
        Float64(eqt.global_quantities.magnetic_axis.r) : r0
    ax_z = (!ismissing(eqt.global_quantities, :magnetic_axis) && !ismissing(eqt.global_quantities.magnetic_axis, :z)) ?
        Float64(eqt.global_quantities.magnetic_axis.z) : z0

    rx, zx = if !isempty(eqt.boundary.x_point) && !ismissing(eqt.boundary.x_point[1], :r)
        (Float64(eqt.boundary.x_point[1].r), Float64(eqt.boundary.x_point[1].z))
    else
        i_minz = argmin([p[2] for p in lcfs_pts])
        (lcfs_pts[i_minz][1], lcfs_pts[i_minz][2])
    end

    coils = NamedTuple[]
    for c in dd.pf_active.coil
        isempty(c.element) && continue
        el = c.element[1]
        rc, zc, wc, hc = if !ismissing(el.geometry, :rectangle) && !ismissing(el.geometry.rectangle, :r)
            rec = el.geometry.rectangle
            (Float64(rec.r), Float64(rec.z), Float64(rec.width), Float64(rec.height))
        elseif !ismissing(el.geometry, :outline) && !ismissing(el.geometry.outline, :r) && !isempty(el.geometry.outline.r)
            ors = el.geometry.outline.r
            ozs = el.geometry.outline.z
            (0.5 * (minimum(ors) + maximum(ors)), 0.5 * (minimum(ozs) + maximum(ozs)), max(0.04, maximum(ors) - minimum(ors)), max(0.04, maximum(ozs) - minimum(ozs)))
        else
            (0.0, 0.0, 0.0, 0.0)
        end
        rc <= 0.05 && continue
        cur_ma = if !ismissing(c, :current) && !ismissing(c.current, :data) && !isempty(c.current.data)
            Float64(c.current.data[1]) / 1.0e6
        else
            0.0
        end
        push!(
            coils,
            (
                name = ismissing(c, :name) ? "PF" : String(c.name),
                r = round(rc; digits = 4),
                z = round(zc; digits = 4),
                w = round(max(wc, 0.04); digits = 4),
                h = round(max(hc, 0.04); digits = 4),
                current_ma = round(cur_ma; digits = 4),
            ),
        )
    end

    all_r = [p[1] for p in vv_out]
    all_z = [p[2] for p in vv_out]
    if !isempty(tf_out)
        append!(all_r, [p[1] for p in tf_out])
        append!(all_z, [p[2] for p in tf_out])
    end
    for c in coils
        push!(all_r, c.r - 0.5 * c.w, c.r + 0.5 * c.w)
        push!(all_z, c.z - 0.5 * c.h, c.z + 0.5 * c.h)
    end
    r_min = round(max(0.0, minimum(all_r) - 0.15 * amin); digits = 2)
    r_max = round(maximum(all_r) + 0.15 * amin; digits = 2)
    z_ext = round(max(abs(minimum(all_z)), abs(maximum(all_z))) + 0.15 * amin; digits = 2)
    span = max(r_max, 2.0 * z_ext)

    return (
        id = id_str,
        kind = "tokamak",
        solver = "FUSE.jl 1.2.1 · IMAS + TEQUILA Equilibrium",
        name = full_name,
        shortName = short_name,
        R0 = round(r0; digits = 3),
        Z0 = round(z0; digits = 3),
        a = round(amin; digits = 3),
        B0 = round(b0; digits = 2),
        ipNominal = round(ip_ma; digits = 2),
        q95 = round(q95; digits = 2),
        betaN = round(beta_n; digits = 2),
        pFusMW = round(p_fus; digits = 2),
        axisR = round(ax_r; digits = 4),
        axisZ = round(ax_z; digits = 4),
        rx = round(rx; digits = 4),
        zx = round(zx; digits = 4),
        dist = round(3.35 * span; digits = 2),
        rMin = r_min,
        rMax = r_max,
        zMin = -z_ext,
        zMax = z_ext,
        n_tf = n_tf,
        tf_outer = tf_out,
        tf_inner = tf_in,
        vessel_outer = vv_out,
        vessel_inner = vv_in,
        first_wall = fw_pts,
        lcfs = lcfs_pts,
        coils = coils,
    )
end

function export_fuse_env(out_path::AbstractString)
    ini_sparc, act_sparc = FUSE.case_parameters(:SPARC; init_from = :scalars)
    sparc = _extract_one_tokamak(
        "sparc",
        "SPARC (Commonwealth Fusion Systems)",
        "SPARC · CFS (FUSE.jl)",
        ini_sparc,
        act_sparc;
        run_pf = true,
    )

    ini_arc, act_arc = FUSE.case_parameters(:ARC)
    arc = _extract_one_tokamak(
        "arc",
        "ARCv3a (MIT PSFC / CFS Compact Pilot Plant)",
        "ARC · MIT/CFS (FUSE.jl)",
        ini_arc,
        act_arc;
        run_pf = true,
    )

    ini_iter, act_iter = FUSE.case_parameters(:ITER; init_from = :ods)
    iter_tok = _extract_one_tokamak(
        "iter",
        "ITER (Q = 10 Burning Plasma Baseline)",
        "ITER · Cadarache (FUSE.jl)",
        ini_iter,
        act_iter;
        run_pf = false,
    )

    ini_d3d, act_d3d = FUSE.case_parameters(:D3D, :H_mode)
    d3d = _extract_one_tokamak(
        "d3d",
        "DIII-D National Fusion Facility (H-mode)",
        "DIII-D · GA (FUSE.jl)",
        ini_d3d,
        act_d3d;
        run_pf = false,
    )

    payload = (
        generator = "FUSE.jl v1.2.1",
        machines = (
            sparc = sparc,
            arc = arc,
            iter = iter_tok,
            d3d = d3d,
        ),
    )
    mkpath(dirname(out_path))
    write(out_path, ShmIpc.to_json(payload))
    return nothing
end

function main()
    if length(ARGS) >= 2 && ARGS[1] == "--export-env"
        export_fuse_env(ARGS[2])
        return nothing
    end
    if length(ARGS) >= 2 && ARGS[1] == "--daemon"
        ShmIpc.run_daemon!(
            ARGS[2],
            _build_fuse_pool,
            eval_fuse!;
            penalty_score = -500.0,
            export_env = export_fuse_env,
        )
        return nothing
    end
    length(ARGS) >= 2 || error(
        "usage: worker.jl --bounds <case> | --daemon <sock> | <shm_path> <case> <workflow> <objective> [penalty]",
    )
    if ARGS[1] == "--bounds"
        dump_bounds(ARGS[2])
        return nothing
    end
    length(ARGS) >= 4 || error(
        "usage: worker.jl <shm_path> <case> <workflow> <objective> [penalty]",
    )
    shm_path = ARGS[1]
    case_name = ARGS[2]
    workflow = ARGS[3]
    obj_name = ARGS[4]
    penalty = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 25.0
    cstr_csv = length(ARGS) >= 6 ? ARGS[6] : ""
    use_atomic = any(==("--atomic"), ARGS)

    ctx1 = build_fuse_ctx(case_name, workflow, obj_name, penalty, cstr_csv)
    slab = ShmIpc.open_slab(shm_path, length(ctx1.x_buf))
    if use_atomic
        ctx1.warmed = true
        ctx_pool = [i == 1 ? ctx1 : deepcopy(ctx1) for i in 1:max(Threads.nthreads(), 1)]
        ShmIpc.serve_atomic!(slab, ctx_pool, eval_fuse!)
    else
        ShmIpc.serve!(_IPC_IN, _IPC_OUT, slab, ctx1, eval_fuse!)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
