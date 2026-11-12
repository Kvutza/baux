#!/usr/bin/env julia
# SPDX-License-Identifier: CC0-1.0
# Minimal academic journal figure (AMS / JMLR style) — data and axes only; prose belongs in \caption{}.

using CairoMakie
using JSON3

const INK       = colorant"#111111"
const INK_AXIS  = colorant"#27272A"
const GRID_LINE = colorant"#ECEEF2"
const RULE_MID  = colorant"#CBD5E1"

const COL_ZERO  = colorant"#111111"
const COL_WASS  = colorant"#1E3A8A"
const COL_EUCL  = colorant"#B45309"
const COL_HYP   = colorant"#0F766E"
const COL_CONF  = colorant"#64748B"

function clean_axis!(ax::Axis)
    ax.topspinevisible = false
    ax.rightspinevisible = false
    ax.leftspinecolor = INK_AXIS
    ax.bottomspinecolor = INK_AXIS
    ax.spinewidth = 0.8
    ax.xtickcolor = INK_AXIS
    ax.ytickcolor = INK_AXIS
    ax.xtickwidth = 0.8
    ax.ytickwidth = 0.8
    ax.xticksize = 3.5
    ax.yticksize = 3.5
    ax.xgridcolor = GRID_LINE
    ax.ygridcolor = GRID_LINE
    ax.xgridwidth = 0.6
    ax.ygridwidth = 0.6
    ax.xlabelcolor = INK
    ax.ylabelcolor = INK
    ax.xticklabelcolor = INK_AXIS
    ax.yticklabelcolor = INK_AXIS
    ax.titlesize = 11.0
    ax.titlealign = :left
    ax.titlegap = 5.0
    ax.xlabelsize = 10.5
    ax.ylabelsize = 10.5
    ax.xticklabelsize = 9.5
    ax.yticklabelsize = 9.5
    return ax
end

function jitter_offsets(n::Int, amp::Float64 = 0.15)
    n <= 1 && return zeros(Float64, n)
    return [amp * sin(1.7 * i + 0.4) for i in 1:n]
end

function render_suite_b(json_path::String, out_prefix::String)
    data = JSON3.read(read(json_path, String))
    cfgs = data.configs
    rounds = Int(data.rounds)
    n_seeds = Int(data.n_paired_seeds)
    xs = collect(1.0:Float64(rounds))

    serif  = Makie.texfont(:text)
    bold   = Makie.texfont(:bold)
    italic = Makie.texfont(:italic)

    set_theme!(Theme(
        fonts = (; regular = serif, bold = bold, italic = italic, bold_italic = italic),
        font = serif,
        backgroundcolor = :white,
    ))

    fig = Figure(size = (820, 255), figure_padding = (10, 14, 10, 10))

    curves = [
        ("turbo_zero_r10_ellipsoid_ball", "TuRBO-0 (R=10)", COL_ZERO, :solid,   2.0),
        ("turbo_enn_r1_wasserstein",      "Wasserstein",    COL_WASS, :solid,   1.7),
        ("turbo_enn_r1_euclidean",        "Euclidean",      COL_EUCL, :dash,    1.6),
        ("turbo_enn_r1_hyperbolic",       "Hyperbolic",     COL_HYP,  :dashdot, 1.6),
        ("turbo_enn_r1_conformal",        "Conformal",      COL_CONF, :dot,     1.6),
    ]

    # (a) Full horizon log-log
    ax_a = Axis(
        fig[1, 1],
        title = "(a)",
        xlabel = L"Round $t$",
        ylabel = "Tracking error",
        xscale = log10,
        yscale = log10,
        xticks = ([1, 10, 100, 1000, 10000], [L"10^0", L"10^1", L"10^2", L"10^3", L"10^4"]),
        yticks = ([1.3, 2.0, 5.0, 10.0, 30.0, 100.0], ["1.3", "2", "5", "10", "30", "100"]),
    )
    clean_axis!(ax_a)
    xlims!(ax_a, 0.9, 11500.0)
    ylims!(ax_a, 1.18, 105.0)

    for (cid, _, col, ls, lw) in reverse(curves)
        c = cfgs[Symbol(cid)]
        m = Float64.(c.mean)
        s = Float64.(c.sem)
        band!(ax_a, xs, max.(m .- s, 1.20), m .+ s, color = (col, cid == "turbo_zero_r10_ellipsoid_ball" ? 0.13 : 0.08))
        lines!(ax_a, xs, m, color = col, linestyle = ls, linewidth = lw)
    end
    scatter!(ax_a, [100.0], [20.59], marker = :utriangle, markersize = 7.5, color = :white, strokecolor = INK, strokewidth = 1.2)
    text!(ax_a, 128.0, 20.59, text = "PPO", fontsize = 8.8, color = INK_AXIS, align = (:left, :center))

    # (b) Tail refinement
    ax_b = Axis(
        fig[1, 2],
        title = "(b)",
        xlabel = L"Round $t$",
        ylabel = "Tracking error",
        xscale = log10,
        xticks = ([100, 300, 1000, 3000, 10000], [L"10^2", L"3{\times}10^2", L"10^3", L"3{\times}10^3", L"10^4"]),
        yticks = 1.3:0.2:2.1,
    )
    clean_axis!(ax_b)
    xlims!(ax_b, 95.0, 10800.0)
    ylims!(ax_b, 1.26, 2.10)

    idx_z = 100:rounds
    xz = xs[idx_z]
    for (cid, _, col, ls, lw) in reverse(curves)
        c = cfgs[Symbol(cid)]
        m = Float64.(c.mean)[idx_z]
        ps = Float64.(c.paired_sem)[idx_z]
        band!(ax_b, xz, m .- ps, m .+ ps, color = (col, cid == "turbo_zero_r10_ellipsoid_ball" ? 0.15 : 0.09))
        lines!(ax_b, xz, m, color = col, linestyle = ls, linewidth = lw)
    end

    handles = [LineElement(color = col, linestyle = ls, linewidth = lw) for (_, _, col, ls, lw) in curves]
    labels  = [lbl for (_, lbl, _, _, _) in curves]
    axislegend(
        ax_b, handles, labels,
        position = :rt,
        framevisible = false,
        backgroundcolor = (:white, 0.85),
        labelsize = 8.8,
        rowgap = 1.5,
        patchsize = (18, 8),
        padding = (4, 4, 2, 2),
    )

    # (c) Paired differences at t = 10^4
    row_specs = [
        ("turbo_enn_r1_conformal",        "Conformal",   COL_CONF, false),
        ("turbo_enn_r1_hyperbolic",       "Hyperbolic",  COL_HYP,  false),
        ("turbo_enn_r1_euclidean",        "Euclidean",   COL_EUCL, false),
        ("turbo_enn_r1_wasserstein",      "Wasserstein", COL_WASS, false),
        ("turbo_zero_r10_origin_only",    "Origin",      COL_ZERO, true),
        ("turbo_zero_r10_nearest_center", "Nearest",     COL_ZERO, true),
        ("turbo_zero_r10_box_contain",    "Box",         COL_ZERO, true),
        ("turbo_zero_r10_ellipsoid_ball", "Ellipsoid",   COL_ZERO, true),
    ]
    nrows = length(row_specs)

    ax_c = Axis(
        fig[1, 3],
        title = "(c)",
        xlabel = L"$\Delta$ error at $t = 10^4$",
        yticks = (collect(1:nrows), [r[2] for r in row_specs]),
        xticks = [-0.2, 0.0, 0.2, 0.4],
    )
    clean_axis!(ax_c)
    xlims!(ax_c, -0.24, 0.44)
    ylims!(ax_c, 0.4, nrows + 0.5)

    hlines!(ax_c, [4.5], color = RULE_MID, linewidth = 0.75)
    vlines!(ax_c, [0.0], color = INK_AXIS, linewidth = 0.9, linestyle = :dash)

    jit = jitter_offsets(n_seeds, 0.15)
    for (idx, (cid, _, col, is_zero)) in enumerate(row_specs)
        c = cfgs[Symbol(cid)]
        diffs = Float64.(c.diff_vs_ref_seeds)
        dm = Float64(c.diff_mean)
        ds = Float64(c.diff_sem)
        ci = Float64(c.diff_ci95)

        y_pts = is_zero ? fill(Float64(idx), n_seeds) : (Float64(idx) .+ jit)
        scatter!(
            ax_c, diffs, y_pts,
            markersize = is_zero ? 5.0 : 5.8,
            color = (col, 0.25),
            strokecolor = (col, 0.8),
            strokewidth = 0.75,
        )
        if !is_zero
            lines!(ax_c, [dm - ci, dm + ci], [idx, idx], color = col, linewidth = 0.95)
            lines!(ax_c, [dm - ci, dm - ci], [idx - 0.11, idx + 0.11], color = col, linewidth = 0.95)
            lines!(ax_c, [dm + ci, dm + ci], [idx - 0.11, idx + 0.11], color = col, linewidth = 0.95)
            lines!(ax_c, [dm - ds, dm + ds], [idx, idx], color = col, linewidth = 2.8)
        end
        scatter!(ax_c, [dm], [idx], marker = :diamond, markersize = 7.5, color = :white, strokecolor = col, strokewidth = 1.4)
    end

    colsize!(fig.layout, 1, Relative(0.32))
    colsize!(fig.layout, 2, Relative(0.36))
    colsize!(fig.layout, 3, Relative(0.32))
    colgap!(fig.layout, 14)

    save(out_prefix * ".pdf", fig)
    save(out_prefix * ".svg", fig)
    save(out_prefix * ".png", fig; px_per_unit = 3)
    println("Saved: ", out_prefix, ".{pdf,svg,png}")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("Usage: plot_suite_b.jl <plot_data.json> <out_prefix>")
    render_suite_b(ARGS[1], ARGS[2])
end
