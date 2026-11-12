# Zero-copy page-aligned mmap worker IPC shared between Rust `baux::unified_mem::ShmSlab`
# and persistent Julia physics workers (`Oceananigans.jl` and `FUSE.jl`).
#
# File-descriptor isolation runs immediately upon inclusion so any package
# `println` or C/Fortran `printf` on fd 1 routes to `stderr` rather than
# corrupting the binary doorbell pipe.

module ShmIpc

using Mmap

const _Sockets = Base.require(Base.PkgId(Base.UUID("6462fe0b-24de-5631-8697-dd941f90decc"), "Sockets"))

const SHM_MAGIC = UInt64(0x4241_5558_5348_4d31)
const READY_TAG = UInt64(0x5244_5931_0000_0001)

const SHM_DOORBELL_CMD_SLOT = 16
const SHM_DOORBELL_SEQ_SLOT = 17
const SHM_DOORBELL_CNT_SLOT = 18
const SHM_ACK_SEQ_SLOT = 32
const SHM_ACK_STATUS_SLOT = 33
const SHM_ACK_US_SLOT = 34
const SHM_ACK_THREADS_SLOT = 35

const SHM_CMD_NOP = UInt64(0)
const SHM_CMD_EVAL = UInt64(1)
const SHM_CMD_DETACH = UInt64(2)
const SHM_CMD_SHUTDOWN = UInt64(3)
const SHM_CMD_SNAP = UInt64(4)

@inline function atomic_load_slot(hdr::Ptr{UInt64}, slot::Int)::UInt64
    return Core.Intrinsics.atomic_pointerref(hdr + slot * 8, :acquire)
end

@inline function atomic_store_slot!(hdr::Ptr{UInt64}, slot::Int, val::UInt64)
    Core.Intrinsics.atomic_pointerset(hdr + slot * 8, val, :release)
    return nothing
end

function isolate_stdio!()
    raw_fd = ccall(:dup, Cint, (Cint,), 1)
    raw_fd < 0 && error("dup(1) failed in ShmIpc")
    rc = ccall(:dup2, Cint, (Cint, Cint), 2, 1)
    rc < 0 && error("dup2(2, 1) failed in ShmIpc")
    redirect_stdout(stderr)
    ipc_in = stdin
    ipc_out = fdio(raw_fd, true)
    return ipc_in, ipc_out
end

struct SlabView
    bytes::Vector{UInt8}
    max_batch::Int
    num_dim::Int
    num_obj::Int
    state_len::Int
    cands::Matrix{Float32}
    scores::Matrix{Float64}
    viols::Vector{Float64}
    state::Vector{Float32}
end

function open_slab(path::AbstractString, expected_dim::Integer)::SlabView
    sz = filesize(path)
    sz < 16384 && error("ShmSlab file too small: $path ($sz bytes)")
    io = open(path, "r+")
    bytes = Mmap.mmap(io, Vector{UInt8}, sz)
    close(io)

    base = pointer(bytes)
    hdr = unsafe_wrap(Array, Ptr{UInt64}(base), 10; own=false)
    magic = ltoh(hdr[1])
    ver = ltoh(hdr[2])
    magic == SHM_MAGIC || error("ShmSlab magic mismatch: $( repr(magic) )")
    ver == 1 || error("ShmSlab unsupported version: $ver")

    max_batch = Int(ltoh(hdr[3]))
    num_dim = Int(ltoh(hdr[4]))
    state_len = Int(ltoh(hdr[5]))
    cand_off = Int(ltoh(hdr[6]))
    score_off = Int(ltoh(hdr[7]))
    viol_off = Int(ltoh(hdr[8]))
    state_off = Int(ltoh(hdr[9]))
    raw_obj = ltoh(hdr[10])
    num_obj = raw_obj == 0 ? 1 : Int(raw_obj)

    expected_dim > 0 && num_dim != Int(expected_dim) &&
        error("ShmSlab num_dim mismatch: slab=$num_dim expected=$expected_dim")

    cands = unsafe_wrap(Array, Ptr{Float32}(base + cand_off), (num_dim, max_batch); own=false)
    scores = unsafe_wrap(Array, Ptr{Float64}(base + score_off), (num_obj, max_batch); own=false)
    viols = unsafe_wrap(Array, Ptr{Float64}(base + viol_off), max_batch; own=false)
    state = unsafe_wrap(Array, Ptr{Float32}(base + state_off), state_len; own=false)

    return SlabView(bytes, max_batch, num_dim, num_obj, state_len, cands, scores, viols, state)
end

@inline function write_scores!(
    col::AbstractVector{Float64},
    sc,
    penalty_score::Float64,
)
    if sc isa Number
        v = isfinite(sc) ? Float64(sc) : penalty_score
        fill!(col, v)
    else
        m_len = length(col)
        s_len = length(sc)
        @inbounds for m in 1:m_len
            raw = m <= s_len ? sc[m] : penalty_score
            col[m] = isfinite(raw) ? Float64(raw) : penalty_score
        end
    end
    return nothing
end

function serve!(
    ipc_in::IO,
    ipc_out::IO,
    slab::SlabView,
    ctx,
    eval_one!::Function;
    penalty_score::Float64 = -1.0e9,
)
    raw = slab.bytes
    GC.@preserve raw begin
        # Pre-warm LLVM + KernelAbstractions JIT before Round 1 starts.
        fill!(@view(slab.cands[:, 1]), 0.0f0)
        try
            _s0, _v0 = eval_one!(
                ctx,
                @view(slab.cands[:, 1]),
                slab.state_len > 0 ? slab.state : nothing,
            )
            write_scores!(@view(slab.scores[:, 1]), _s0, penalty_score)
            slab.viols[1] = Float64(_v0)
        catch err
            @warn "ShmIpc warmup rollout caught solver exception (continuing)" exception =
                (err, catch_backtrace())
            fill!(@view(slab.scores[:, 1]), penalty_score)
            slab.viols[1] = 1.0e6
        end

        ready = Ref((htol(READY_TAG), htol(UInt64(slab.num_dim))))
        write(ipc_out, ready)
        flush(ipc_out)

        req = Ref((UInt32(0), UInt32(0)))
        ack = Ref((UInt32(0), UInt32(0), UInt64(0)))

        while !eof(ipc_in)
            try
                read!(ipc_in, req)
            catch err
                err isa EOFError && break
                rethrow(err)
            end
            cmd_raw, cnt_raw = req[]
            cmd = ltoh(cmd_raw)
            cnt = Int(ltoh(cnt_raw))
            cmd == 0 && break

            t0 = time_ns()
            status = UInt32(0)
            if cnt < 1 || cnt > slab.max_batch
                status = UInt32(2)
            elseif cmd == 1
                @inbounds for b in 1:cnt
                    try
                        sc, vl = eval_one!(ctx, @view(slab.cands[:, b]), nothing)
                        write_scores!(@view(slab.scores[:, b]), sc, penalty_score)
                        slab.viols[b] = isfinite(vl) ? Float64(vl) : 1.0e6
                    catch err
                        @warn "worker eval_one! caught non-convergent solver step ($(typeof(err)))" maxlog=1
                        fill!(@view(slab.scores[:, b]), penalty_score)
                        slab.viols[b] = 1.0e6
                    end
                end
            elseif cmd == 2
                try
                    st_out = slab.state_len > 0 ? slab.state : nothing
                    sc, vl = eval_one!(ctx, @view(slab.cands[:, 1]), st_out)
                    write_scores!(@view(slab.scores[:, 1]), sc, penalty_score)
                    slab.viols[1] = isfinite(vl) ? Float64(vl) : 1.0e6
                catch err
                    @warn "worker snapshot eval_one! caught solver error" exception=(err, catch_backtrace()) maxlog=3
                    fill!(@view(slab.scores[:, 1]), penalty_score)
                    slab.viols[1] = 1.0e6
                    status = UInt32(1)
                end
            else
                status = UInt32(3)
            end

            Core.Intrinsics.atomic_fence(:release)
            dt_us = UInt64(div(time_ns() - t0, 1000))
            ack[] = (htol(status), htol(UInt32(cnt)), htol(dt_us))
            write(ipc_out, ack)
            flush(ipc_out)
        end
    end
    return nothing
end

@inline function _eval_batch!(slab::SlabView, ctx_pool::Vector, eval_one!::Function, cnt::Int, penalty_score::Float64)
    nctx = length(ctx_pool)
    if cnt <= 1 || nctx <= 1 || Threads.nthreads() <= 1
        ctx = ctx_pool[1]
        @inbounds for b in 1:cnt
            try
                sc, vl = eval_one!(ctx, @view(slab.cands[:, b]), nothing)
                write_scores!(@view(slab.scores[:, b]), sc, penalty_score)
                slab.viols[b] = isfinite(vl) ? Float64(vl) : 1.0e6
            catch err
                println(stderr, "[ShmIpc] solver non-convergence: ", sprint(showerror, err))
                flush(stderr)
                fill!(@view(slab.scores[:, b]), penalty_score)
                slab.viols[b] = 1.0e6
            end
        end
        return UInt64(0)
    end
    Threads.@threads :static for b in 1:cnt
        tid = clamp(Threads.threadid(), 1, nctx)
        ctx = @inbounds ctx_pool[tid]
        try
            sc, vl = eval_one!(ctx, @view(slab.cands[:, b]), nothing)
            write_scores!(@view(slab.scores[:, b]), sc, penalty_score)
            slab.viols[b] = isfinite(vl) ? Float64(vl) : 1.0e6
        catch err
            fill!(@view(slab.scores[:, b]), penalty_score)
            slab.viols[b] = 1.0e6
        end
    end
    return UInt64(0)
end

function serve_atomic!(
    slab::SlabView,
    ctx_pool::Vector,
    eval_one!::Function;
    penalty_score::Float64 = -1.0e9,
    slab_path::AbstractString = "",
)
    raw = slab.bytes
    GC.@preserve raw begin
        hdr = Ptr{UInt64}(pointer(raw))
        last_seq = atomic_load_slot(hdr, SHM_ACK_SEQ_SLOT)
        atomic_store_slot!(hdr, SHM_ACK_THREADS_SLOT, UInt64(Threads.nthreads()))
        spins = 0
        while true
            seq = atomic_load_slot(hdr, SHM_DOORBELL_SEQ_SLOT)
            if seq > last_seq
                spins = 0
                last_seq = seq
                cmd = atomic_load_slot(hdr, SHM_DOORBELL_CMD_SLOT)
                cnt = Int(atomic_load_slot(hdr, SHM_DOORBELL_CNT_SLOT))
                if cmd == SHM_CMD_DETACH || cmd == SHM_CMD_SHUTDOWN
                    atomic_store_slot!(hdr, SHM_ACK_STATUS_SLOT, UInt64(0))
                    atomic_store_slot!(hdr, SHM_ACK_US_SLOT, UInt64(0))
                    atomic_store_slot!(hdr, SHM_ACK_SEQ_SLOT, seq)
                    return cmd
                end
                t0 = time_ns()
                status = UInt64(0)
                if cmd == SHM_CMD_EVAL
                    if cnt < 1 || cnt > slab.max_batch
                        status = UInt64(2)
                    else
                        status = _eval_batch!(slab, ctx_pool, eval_one!, cnt, penalty_score)
                    end
                elseif cmd == SHM_CMD_SNAP
                    try
                        st_out = slab.state_len > 0 ? slab.state : nothing
                        sc, vl = eval_one!(ctx_pool[1], @view(slab.cands[:, 1]), st_out)
                        write_scores!(@view(slab.scores[:, 1]), sc, penalty_score)
                        slab.viols[1] = isfinite(vl) ? Float64(vl) : 1.0e6
                    catch err
                        println(stderr, "[ShmIpc] snapshot non-convergence: ", sprint(showerror, err))
                        flush(stderr)
                        fill!(@view(slab.scores[:, 1]), penalty_score)
                        slab.viols[1] = 1.0e6
                        if slab.state_len > 0
                            fill!(slab.state, 0.0f0)
                        end
                    end
                else
                    status = UInt64(3)
                end
                Threads.atomic_fence()
                dt_us = UInt64(div(time_ns() - t0, 1000))
                atomic_store_slot!(hdr, SHM_ACK_STATUS_SLOT, status)
                atomic_store_slot!(hdr, SHM_ACK_US_SLOT, dt_us)
                atomic_store_slot!(hdr, SHM_ACK_THREADS_SLOT, UInt64(Threads.nthreads()))
                atomic_store_slot!(hdr, SHM_ACK_SEQ_SLOT, seq)
            else
                spins += 1
                if spins < 512
                    GC.safepoint()
                elseif spins < 2048
                    yield()
                else
                    if mod(spins, 1024) == 0 && !isempty(slab_path) && !isfile(slab_path)
                        return SHM_CMD_DETACH
                    end
                    sleep(0.0005)
                end
            end
        end
    end
end

function write_json(io::IO, s::AbstractString)
    write(io, '"')
    for c in s
        if c == '"'
            write(io, "\\\"")
        elseif c == '\\'
            write(io, "\\\\")
        elseif c == '\n'
            write(io, "\\n")
        elseif c == '\r'
            write(io, "\\r")
        elseif c == '\t'
            write(io, "\\t")
        elseif UInt32(c) < 0x20
            write(io, ' ')
        else
            write(io, c)
        end
    end
    write(io, '"')
    return nothing
end

write_json(io::IO, s::Symbol) = write_json(io, String(s))
write_json(io::IO, b::Bool) = (write(io, b ? "true" : "false"); nothing)
write_json(io::IO, ::Nothing) = (write(io, "null"); nothing)
write_json(io::IO, ::Missing) = (write(io, "null"); nothing)
write_json(io::IO, n::Integer) = (print(io, n); nothing)

function write_json(io::IO, x::AbstractFloat)
    v = isfinite(x) ? round(Float64(x); digits = 5) : 0.0
    if v == -0.0
        v = 0.0
    end
    print(io, v)
    return nothing
end

function write_json(io::IO, xs::Union{AbstractVector,Tuple})
    write(io, '[')
    first = true
    for item in xs
        first || write(io, ',')
        first = false
        write_json(io, item)
    end
    write(io, ']')
    return nothing
end

function write_json(io::IO, nt::NamedTuple)
    write(io, '{')
    first = true
    for k in keys(nt)
        first || write(io, ',')
        first = false
        write_json(io, String(k))
        write(io, ':')
        write_json(io, getfield(nt, k))
    end
    write(io, '}')
    return nothing
end

function write_json(io::IO, d::AbstractDict)
    write(io, '{')
    first = true
    for k in sort!(collect(keys(d)); by = string)
        first || write(io, ',')
        first = false
        write_json(io, string(k))
        write(io, ':')
        write_json(io, d[k])
    end
    write(io, '}')
    return nothing
end

function to_json(x)::String
    io = IOBuffer()
    write_json(io, x)
    return String(take!(io))
end

function run_daemon!(
    sock_path::AbstractString,
    build_ctx_pool::Function,
    eval_one!::Function;
    penalty_score::Float64 = -1.0e9,
    export_env::Union{Nothing,Function} = nothing,
)
    rm(sock_path; force=true)
    mkpath(dirname(sock_path))
    server = _Sockets.listen(sock_path)
    ctx_cache = Dict{String,Any}()
    try
        while true
            sock = _Sockets.accept(server)
            try
                while isopen(sock)
                    line = readline(sock; keep=false)
                    isempty(line) && break
                    parts = split(line, '\t'; limit=3)
                    op = parts[1]
                    if op == "PING"
                        write(sock, "PONG\n")
                        try flush(sock) catch end
                        break
                    elseif op == "SHUTDOWN"
                        write(sock, "BYE\n")
                        try flush(sock) catch end
                        return nothing
                    elseif op == "ENV" && length(parts) >= 2 && export_env !== nothing
                        export_env(parts[2])
                        write(sock, "OK\n")
                        try flush(sock) catch end
                        break
                    elseif op == "BOUNDS" && length(parts) >= 2
                        cfg_key = parts[2]
                        _, bounds = get!(ctx_cache, cfg_key) do
                            build_ctx_pool(cfg_key)
                        end
                        low, high = bounds
                        write(sock, string("BOUNDS\t", join(low, ","), "\t", join(high, ","), "\n"))
                        try flush(sock) catch end
                        break
                    elseif op == "OPEN" && length(parts) >= 2
                        slab_path = parts[2]
                        cfg_key = length(parts) >= 3 ? parts[3] : ""
                        ctx_pool, bounds = get!(ctx_cache, cfg_key) do
                            build_ctx_pool(cfg_key)
                        end
                        low, _ = bounds
                        slab = open_slab(slab_path, length(low))
                        write(
                            sock,
                            string("OK\t", length(low), "\t", slab.state_len, "\t", slab.num_obj, "\n"),
                        )
                        flush(sock)
                        cmd = serve_atomic!(
                            slab,
                            ctx_pool,
                            eval_one!;
                            penalty_score = penalty_score,
                            slab_path = slab_path,
                        )
                        if cmd == SHM_CMD_SHUTDOWN
                            return nothing
                        end
                        break
                    else
                        write(sock, "ERR\tunknown command\n")
                        try flush(sock) catch end
                        break
                    end
                end
            catch err
                if !(err isa Base.IOError)
                    @warn "daemon client error" exception=(err, catch_backtrace()) maxlog=5
                end
            finally
                try close(sock) catch end
            end
        end
    finally
        try close(server) catch end
        rm(sock_path; force=true)
    end
end

end # module ShmIpc
