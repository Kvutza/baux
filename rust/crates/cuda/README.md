# CUDA-Oxide

## GPU artifact capture

Set `BAUX_GPU_EVIDENCE` to a directory before invoking an existing CUDA
command to retain loader evidence. This enables evidence I/O at module creation,
not in the kernel launch loop. Without it, model loading uses the original
generated loader. Evidence write failures propagate to the caller.

All twelve current `fbt_model` construction sites use `src/capture.rs`: the
three model constructors, prefill, context cache, embedding/feedback checks,
precision/mHC/routing/weights checks, and synthesized GEMM's reference module.
The helper follows the pinned non-generic loader's first matching module bundle
and Cubin-before-PTX selection. It binds the unchanged matching image with the
generated `from_module` API; arbitrary matching symbols are insufficient.

| Path | Recorded bytes | Remaining gap |
| --- | --- | --- |
| Embedded cubin/PTX | Exact image submitted to `cuModuleLoadData`, including the terminator cuda-core otherwise adds internally; target and available bundle compile policy | Original producer identity/complete flags and driver version are unreported; PTX JIT/native relocation output is unavailable |
| Embedded NVVM IR/LTO IR | Original selected IR, bundle target/policy, successful compilation/link stage with no output image | Pinned loader keeps architecture routing and its materialized image private; its original loading behavior is preserved |
| Generated GEMM | Exact generated PTX, ordered `ptxas` arguments, unchanged cubin output, and separately the terminated cubin submitted to the driver | `ptxas` executable identity/version and driver relocation output are unreported |

GEMM temporary filenames include a process-local sequence so concurrent recipes
in the same process do not overwrite one another. Capture records are content
identities, not proofs or execution traces. They do not establish which graph
nodes execute, replay arguments, buffer identities/contents, launch geometry,
or coverage of the separate trials/Muon/PTX-probe loaders. Capturing the opaque
IR loader's actual output requires an upstream observer at materialization/load;
reimplementing its private compatibility policy would risk changing behavior.

Implementation references are the pinned
[cuda-host embedded loader](https://github.com/NVlabs/cuda-oxide/blob/6abfaa091e29a6275c1943895bfbc97efa306e98/crates/cuda-host/src/embedded.rs)
and [generated bindings](https://github.com/NVlabs/cuda-oxide/blob/6abfaa091e29a6275c1943895bfbc97efa306e98/crates/cuda-macros/src/cuda_module/mod.rs),
plus local `cuda-core 0.3.1` `src/simt/module.rs`'s image terminator behavior.

Validation on 2026-10-23: rustfmt parsing/format checks passed for all ten changed
Rust files; `./baux check` passed for the new capture and GEMM code. The full
changed-file naming scan reports only the existing `initialize_mhc4` names in
`model.rs` and `model/weights.rs`. `./tools/buck2w build //:cuda-wheel` rejected
the macOS/ARM host as incompatible with its Linux/x86_64 requirement. No CUDA
typecheck, hardware load, benchmark, or native verification was performed.

## Native module observer and instruction coverage

The existing Modal runner `ptx` action also performs a small production
`model-check` fixture when `BAUX_GPU_EVIDENCE` is set. It enables the generated
GEMM route with `BAUX_SYNTH_GEMM=32`, subscribes to CUPTI module-load callbacks,
copies each callback-owned image before returning, and retains its SHA-256 and
`nvdisasm` output. `sandbox.txt` preserves the remote handle; `modal-native.tar.gz`
contains the evidence. An existing destination is rejected. The observer is
opt-in and does not intercept launches or replace the driver's loader.

The [CUPTI module resource contract](https://docs.nvidia.com/cupti/13.1.0/api/structCUpti__ModuleResourceData.html)
exposes module ID, cubin size and image pointer. This can observe the image hidden
behind CUDA-Oxide's private IR loader. It does not prove device relocation,
executed graph-node coverage, launch arguments, instruction decoding or ISA
semantics. A loaded module may include kernels the finite fixture never calls.

Validation on 2026-10-23 and 2026-10-24 (artifact timestamps 2026-10-24 UTC): Linux
CUDA release build, C observer compilation, finite `model-check`, native
disassembly, and the 4,096-element/one-iteration PTX probe passed on Tesla T4,
and were repeated on the shared-Gaussian revision `06b8004f014dd0d2ff940acfafe386e487cb66b6`
(host executable SHA-256 `b8f39239f257e5a340ef249b481c9df7d6be3c397af16d582939e65adb9b8d08`).
The first Linux build exposed seven old module-split import/visibility errors;
the correction only restores the splitmix64 alias, FP16 helper imports and shared
constants. The model fixture reported exact FP16/MOE/routing/sampling/generation
comparisons and successful mHC graph reuse. Under `compute-sanitizer` (version
2025.3.0.0, sandbox `sb-JbznMdlFSLMMSBbFEoZOC7`), both `memcheck` and `synccheck`
passed with zero errors (`status=0`) on `parity` (36 cases, 51,437 neighbor
samples) and `resident` (UCB/Thompson/Pareto acquisitions and BF16 resident
search), and `synccheck` with `BAUX_SYNTH_GEMM=32` passed with zero errors
(`status=0`) across the complete `model-check` suite (`4096` rows, executions
`5/7/11`, generation unroll `1/2`, full model and causal mHC parity). In the
preceding `retry-005` run (`sb-x32QTweiDjQ8c3BaJrIEww`), `memcheck` on
`model-check` completed through `MODEL rows=4096 executions=5,7,11` with zero
reported errors before reaching its 900 s command timeout (`status=124`, cgroup
`memory.peak` 2.57 GiB, `oom=0`). These checks establish neither all-path
coverage nor native ISA semantics.

The pinned Rust Modal SDK still fails before allocation because it sends an
unspecified SDK client type. The approved temporary official SDK 1.6.1 fallback
rebuilt the existing recipe in the user-selected `kvutza` workspace. It preserved
CUDA 13.0.1, LLVM 21, Rust nightly-2026-08-28 and CUDA-Oxide revision 6abfaa0.
Image `im-2E2WoWnwbZ81zkSkRXDLMb` ran sandbox `sb-aOXdqxcEEMDET0aSds3rSL`, which
was terminated after successful evidence download. Driver KMD was 610.57.04,
CUDA UMD 13.3; `ptxas` was 13.0.88 and `nvdisasm` 13.0.85. The actual loader
selected embedded PTX, so this run demonstrates observation of driver-JIT output;
the separate NVVM/LTO input paths remain unexercised.

| Captured module | SHA-256 | Native bytes | Text sections | Static instructions | Opcode/modifier forms |
| --- | --- | ---: | ---: | ---: | ---: |
| Production device module | `cca27b1d617254ada082a8c695bf1243a99dd6eb0fce9629708ab41f0d30701a` | 800432 | 100 | 39024 | 196 |
| Generated GEMM | `9efa587a73ce8c7de0105f83d7edef281539a4b31ed4e3db9544eaeca771847f` | 7712 | 1 | 280 | 37 |

The production module includes the model, trials, Muon and diffusion sections;
these counts describe loaded code, not executed coverage. Four model loads had
identical cubin hashes. The observed GEMM image exactly matched its recorded
assembler output. Full model forms include `HMMA.1688.F32`, `BAR.SYNC`,
`WARPSYNC`, `BSSY/BSYNC`, `BRX`, relative calls/returns, FP64 arithmetic, directed
rounding, FTZ conversions, SFU approximations and shared/local/global memory.
None currently has a checked sm_75 decoding-and-execution proof in this project.
Counting mnemonics is insufficient: operands, modifiers, control bits, ABI and
reachable call/branch targets also need checking.

Evidence is under
`results/cuda/native-20261024-kvutza-001/retry-002/`: `source-change.txt` binds the
initial archive revision and exact three-file correction; `modal-native.tar.gz`
preserves build logs, lockfile, host executable hash, PTX/CBOR recipes, native
images and disassembly. `evidence/baux-gpu-evidence/*.inventory.json` records
every text section and form, with zero unmatched instruction lines. This census
uses vendor disassembly and is not a checked decoder.

Before this run, the two available `results/ptx/muon[-ws]-t4-20261008.sass` files
were identical 26-kernel calibration captures, not full-model evidence. Their
SHA-256 is `d123bbb4cf2be31d4b749a0db72519b148d0453e421c5c92037b370f499ec643`.

### SASS-guided spill, division, and FP64 elimination (`native-20261024-kvutza-005`)

Inspecting the captured T4 `sm_75` ELF `.cubin` (`module-364-0-21.cubin`) and
disassembly (`module-364-0-21.cubin.sass`) identified five concrete codegen
bottlenecks across nine kernels:
1. Array-literal `for` loops (`core::array::IntoIter`) in `device/src/fbt_route.rs`
   (`attention_dot`, `attention_norm`, `multihead_norm`) spilled iterator state
   to thread-local stack memory (`8` `LDL`/`STL` instructions in `pisa_attention`
   and `6` in `pisa_multihead`).
2. Checked slice indexing (`qkv[...]`, `blocks[...]`, `kv[...]`, `scores_input[...]`,
   `input[...]`, `weights[...]`) and runtime integer divisions/modulos by
   `shape.block_tokens` and `shape.head_width` (`64`, enforced by `PisaShape::validate`)
   in `device/src/fbt_route.rs` and `device/src/fbt.rs` emitted `BPT.TRAP` panic
   branches (`BSSY`/`BSYNC` divergence barriers) and `MUFU.RCP` reciprocal
   Newton-Raphson sequences, inflating `pisa_attention` to `37` GPRs (`928` insns)
   and `pisa_multihead` to `38` GPRs (`1,088` insns). Replacing checked indexing
   under validated launch bounds with `get_unchecked` and runtime `/ 64`, `% 64`
   with `>> 6`, `& 63` reduced `pisa_attention` to `21` GPRs (`592` insns, `-36.2%`)
   and `pisa_multihead` to `29` GPRs (`648` insns, `-40.4%`).
3. Local stack arrays in `device/src/fbt.rs` (`mhc_predict`: `dots: [f32; 3]`
   and serial `transport: [f32; 16]`) emitted `38` `LDL`/`STL` instructions (`80 B`
   stack frame); replacing `dots` with scalar accumulators and reusing shared memory
   `raw[4..20]` in the serial branch removed all `38` spills (`0 B` frame, `46 -> 41` GPRs).
4. `dense_normal` in `device/src/bf16.rs` performed `f64` uniform conversions
   (`I2F.F64.U64` + `DMUL` + `F2F.F32.F64`) and diverged odd/even lanes across
   separate `.cos()` and `.sin()` calls; switching to 24-bit `f32` uniform
   scaling and a phase-shifted single `.cos()` call cut `FP64` instructions by
   `47%–75%` across `write_bf16`, `distance_bf16`, `distance2_bf16`, and
   `reference_bf16`.
5. `A_LOAD` in `rust/crates/ptx-synth/src/gemm.rs` emitted `div.u32` and `rem.u32`
   by `32`; replacing with `shr.u32` and `and.b32` eliminated the remaining
   `MUFU.RCP` sequence in `baux_gemm_k32`.

Validated on Tesla T4 (`sb-sXeImPCHaC9AHTqCNqrHsR`, host executable SHA-256
`504a63672d124cace994361c53ca8782cb7e86b85536195058a3ff6fd5cd9bc3`, loaded
production module SHA-256
`53d6fa2cee8ad45fe2543586619b4c2c82197085ae5fe1aa1ab9987a98bde60e`, `778,160`
bytes vs `800,432` baseline, generated GEMM module SHA-256
`31cb4fa11a916e18e6ca9e806ef0b34cdba01d8b3119983a98fa3a8ffb007bab`; all
`parity`, `resident`, `model-check`, and `compute-sanitizer` `memcheck`/`synccheck`
checks passed with zero errors):

| Kernel | Base Insn | Opt Insn | Base GPR | Opt GPR | Spills (`LDL`+`STL`) | `FP64` | `MUFU.RCP` | `BPT.TRAP` | Paired T4 Speedup |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| `pisa_attention` | 928 | 592 | 37 | 21 | `8 -> 0` | `0 -> 0` | `8 -> 5` | `9 -> 2` | `1.966x` (`29.325 -> 14.915 ms` in 4K `prefill` attention, `1.797x` full `prefill` device `32.900 -> 18.306 ms`) |
| `pisa_multihead` | 1088 | 648 | 38 | 29 | `6 -> 0` | `0 -> 0` | `7 -> 4` | `9 -> 1` | `1.165x` (`53.478 -> 45.886 ms` `wave_pisa_ms` in 4K `wave-fixture`; `1.149x` in 65K `context` `2.633 -> 2.291 ms`) |
| `mhc_predict` | 2016 | 2024 | 46 | 41 | `38 -> 0` | `0 -> 0` | `20 -> 20` | `1 -> 1` | `1.038x` (`5.414 -> 5.218 ms` `wave_mhc_ms`; `1.113x` end-to-end 4K `wave_total_ms` `659.24 -> 592.27 ms`) |
| `routed_gate` | 384 | 360 | 26 | 25 | `0 -> 0` | `0 -> 0` | `0 -> 0` | `4 -> 1` | `1.054x` (`0.606 -> 0.575 ms` routing in 4K `prefill`) |
| `write_bf16` | 856 | 736 | 53 | 51 | `0 -> 0` | `19 -> 10` | `6 -> 6` | `0 -> 0` | `1.440x–1.559x` (`0.0568 -> 0.0364 ms` at 262K Gaussian `bf16-bench`) |
| `distance_bf16` | 784 | 672 | 48 | 48 | `0 -> 0` | `19 -> 10` | `4 -> 4` | `0 -> 0` | paired parity with `distance2_bf16` |
| `distance2_bf16` | 1336 | 1112 | 56 | 56 | `0 -> 0` | `37 -> 19` | `6 -> 6` | `0 -> 0` | `1.046x` score (`0.9349 -> 0.8942 ms`), `1.065x` total `gpu_ms` at 262K |
| `reference_bf16` | 896 | 664 | 38 | 38 | `0 -> 0` | `24 -> 6` | `7 -> 7` | `1 -> 1` | `-25.9%` static instructions, `-75.0%` `FP64` |
| `baux_gemm_k32` | 280 | 280 | 53 | 55 | `0 -> 0` | `0 -> 0` | `1 -> 0` | `0 -> 0` | `0` `MUFU.RCP` in synthesized `ptx-synth` GEMM |
| **Module total (101 kernels, Wave 2)** | **39304** | **38040** | — | — | **`62 -> 10`** | **`398 -> 344`** | **`302 -> 295`** | **`181 -> 161`** | `-22,272` cubin bytes, `-64.3%` stack frame bytes (`224 -> 80 B`) |

### Wave 3: 1M-context forward-pass kernel & tree-closure optimization (`native-20261024-kvutza-006`)

Using `./baux research kernel-scorecard`, `kernel-diff`, and `kernel-slice` to inspect the 1M-token (`1,048,576` context = `16,384` leaves, `11` tree levels) forward-pass path across both CUDA (`sm_75`) and Apple Silicon (`G16`) exposed four 1M-context scaling bottlenecks:
1. **`pisa_select` and `pisa_shadow_refine` (`device/src/fbt.rs`)**: `60%` of `pisa_select` (`738` of `1,249` SASS lines) was a thread-0 serial selection loop (`if tid == 0`) over shared memory (`SCORES`, `CANDIDATES`, `CHOSEN`, `EXPANDED`) with `2` `BAR.SYNC` barriers per tree level (`22` barriers per query at 1M context), `3` `MUFU.RCP` divisions, and `2` `BPT.TRAP` branches. Replacing the shared-memory serial loop with in-register 16-lane butterfly warp reductions (`warp::shuffle_xor_f32` / `warp::shuffle_xor` over offsets `8, 4, 2, 1`) and `warp::shuffle_f32_sync(..., tid >> 1)` child expansion reduced shared memory to a single `QUERY` array and `1` `BAR.SYNC` barrier across the entire kernel (`0` `MUFU.RCP`, `0` `BPT.TRAP`, `8 -> 2` `BSYNC`), while preserving exact `[u32; 8]` block parity with `baux::semantics::context::probe::Fixture::blocks`.
2. **`attention_pos` (`device/src/fbt_route.rs`) and tree builders (`pisaleaves`, `pisaupper`, `cache_leaves`, `rope`, `rms`, `mhc_mix`, `mhc_update`, `combine_routes`, `combine_mhc`)**: Replacing runtime power-of-two divisions/modulos (`row / shape.context`, `row % shape.context`, `((absolute / shape.visibility) + 1) * shape.visibility - 1`) with bitshifts and bitmasks (`row >> shape.context.trailing_zeros()`, `row & (shape.context - 1)`, `absolute | shape.visibility.saturating_sub(1)`) eliminated all remaining `BPT.TRAP`s in `pisa_multihead` (`9 -> 0`) and `pisaleaves` (`384 -> 72` static instructions, `-81.2%`, `29 -> 13` GPRs).
3. **1M-context subtree repair launch closure (`backends/cuda/src/context.rs`)**: Routing bounded leaf repairs (`end - first <= 64`) through the single-block `cache_ancestors` closure instead of `14` separate per-level `cache_parents` launches reduced 1M-context `repair-ms` from `0.0940 ms` to `0.0554 ms` (`1.698x` speedup, `-41.1%`).
4. **Apple M4 (`Target::G16`) `fbt_pisa1_select_attention_q4` (`fbt_pisa1.metal`), `pisa_index` (`pisa_index.metal`), and `context_parents` (`context_tree.metal`)**: Replacing per-level `selected_blocks` threadgroup stores/loads and `22` `simdgroup_barrier`s across 11 tree levels with in-register `simd_shuffle` eliminated `100%` of stack frame spills in `fbt_pisa1_select_attention_q4` on Apple M4 (`frame_spill_bytes: 2,304 -> 0 B`, static AGX instructions `3,643 -> 3,260`, or `3,157` with `PISA_CACHE`), vectorized the 64-dim tree dot product in `pisa_index.metal` with `float4`/`half4` loads (`0 B` spill), and parallelized `context_parents` across `512` threads (`8` parent nodes per step).

Validated on Tesla T4 (`sb-OZ1y8u6K7kEv5n08EcVi1R`, host executable SHA-256 `d640d8106350dbc3fa2d2e426349cf516fbca90889286868a2ea178b68c7226a`, production module SHA-256 `4f1b24335762a8f7ffcae2cc97573b13d8818439ae69d8d4bada6a19fc6a6442`, `781,488` bytes, `37,920` static instructions vs `39,024` baseline, `80 B` stack frame vs `224 B` baseline, `10` spill ops vs `62` baseline, `282` `MUFU.RCP` vs `301` baseline, `145` `BPT.TRAP` vs `181` baseline, `970` `BSYNC` vs `1,030` baseline; all `parity`, `resident`, `model-check`, 1M-context `context` probes, and `compute-sanitizer` `memcheck`/`synccheck` passed with zero errors):

| Workload / Stage (Tesla T4 Paired A/B) | Baseline Median | Wave 3 Optimized Median | Speedup | Delta |
| --- | ---: | ---: | ---: | ---: |
| **1M Context (`1,048,576` tokens, `q=128`)** `median-query-device-ms` (`pisa_select` + `pisa_multihead`) | `0.9881 ms` | `0.6977 ms` | **`1.416x`** | `-29.39%` |
| **1M Context (`1,048,576` tokens, `q=128`)** `repair-ms` (14-level subtree closure) | `0.0942 ms` | `0.0566 ms` | **`1.663x`** | `-39.88%` |
| **1M Context (`1,048,576` tokens, `q=512`)** `median-query-device-ms` | `2.8458 ms` | `2.4039 ms` | **`1.184x`** | `-15.53%` |
| **1M Context (`1,048,576` tokens, `q=512`)** `repair-ms` | `0.0940 ms` | `0.0554 ms` | **`1.698x`** | `-41.10%` |
| **1M Forward Wave (`seq=1,048,576, chunk=4,096, visits=2`)** `wave_1m_index_ms` (`rope`+`split`+`tree`+11-level `pisa_select`) | `16.146 ms` | `13.731 ms` | **`1.176x`** | `-14.96%` |
| **1M Forward Wave (`seq=1,048,576, chunk=4,096, visits=2`)** `wave_1m_pisa_ms` (`pisa_multihead` across 11 visits) | `60.203 ms` | `50.104 ms` | **`1.202x`** | `-16.78%` |
| **1M Forward Wave (`seq=1,048,576, chunk=4,096, visits=2`)** `wave_1m_attention_ms` | `93.805 ms` | `81.176 ms` | **`1.156x`** | `-13.46%` |
| **1M Forward Wave (`seq=1,048,576, chunk=4,096, visits=2`)** `wave_1m_mhc_ms` | `5.811 ms` | `5.393 ms` | **`1.077x`** | `-7.19%` |
| **1M Forward Wave (`seq=1,048,576, chunk=4,096, visits=2`)** `wave_1m_total_ms` (complete 11-visit 4K chunk at 1M ctx) | `770.008 ms` | `685.240 ms` | **`1.124x`** | `-11.01%` (`-84.8 ms`/chunk) |
| **4K Prefill (`rows=4,096`)** `median_selection_ms` (`pisa_select`) | `0.520 ms` | `0.367 ms` | **`1.417x`** | `-29.42%` |
| **4K Prefill (`rows=4,096`)** `median_attention_ms` (`pisa_attention`) | `35.387 ms` | `17.095 ms` | **`2.070x`** | `-51.69%` |
| **4K Prefill (`rows=4,096`)** `median_rms_ms` / `median_residual_ms` / `median_routing_ms` | `0.100 / 1.319 / 0.728 ms` | `0.080 / 1.128 / 0.625 ms` | **`1.250x / 1.169x / 1.165x`** | `-20.0% / -14.5% / -14.2%` |
| **4K Prefill (`rows=4,096`)** `median_device_ms` (full prefill) | `39.698 ms` | `20.825 ms` | **`1.906x`** | `-47.54%` |

The actual `device/src/fbt.rs` has 57 kernel declarations. Its source requires
at least the following semantics, regardless of what the probe corpus covers:

| Source operation | Required obligation |
| --- | --- |
| FP16 `mma.sync.aligned.m16n8k8` with FP32 accumulation | Tensor fragment ownership, warp participation, accumulation/rounding and native tensor instruction correspondence |
| INT8 `mma.sync.aligned.m8n8k16` | Signed integer fragment mapping and accumulation semantics |
| Shared reductions and staging | Per-block allocation, barrier convergence, visibility and race freedom |
| Warp shuffle/vote/reduction | Active masks, lane mappings and behavior of inactive participants |
| FP16 packing, exp/log/sqrt/reciprocal | Exact conversions, exceptional values, rounding, approximation bounds and compiler contraction policy |
| Global addressing and branch-heavy routing | Pointer provenance, widths, signedness, control flow and actual parameter ABI |

`fbt_math::gaussian_normal` and `gaussian_tail` also use unbounded rejection
loops with wrapping counters. The production integer mixer and counter-word
functions (`baux::semantics::gaussian::{mix32, bits, normal_bounds}`) are now
shared with host semantics, extracted by Charon/Aeneas, and proved in
`BauxResearch.Semantics.Sampler` (`mixMatch`, `bitsRaw`, `bitsMatch`,
`boundsMatch`, `normalReach`, `otherMatch`, `tailReach`) to refine the
kernel-checked `BauxResearch.Semantics.Gaussian` permutation and conditional
`< 2^32` coset reachability theorems, while `backends/cuda/src/model/weights.rs`
enforces `normal_bounds` on the uploaded Ziggurat table before device upload.
Universal device loop termination still requires proving the floating-point
acceptance/log predicates, Ziggurat table generation, and native `sm_75`
compilation correspondence. A statistical termination claim does not establish
this deterministic total-correctness obligation.

For decoding, [CuAssembler](https://github.com/cloudcores/CuAssembler) supports
sm_75 encodings but delegates program semantics to the user and uses NVIDIA
disassembly; its README warns that disassembly can omit modifiers. It is useful
for studying encoding fields, not a checked decoder. [DocumentSASS](https://github.com/0xD0GF00D/DocumentSASS)
extracts encoding descriptions from NVIDIA tools; it also does not prove
semantics. [Y's validator target gate](https://github.com/NadezhdaaYSU/Y/blob/main/tools/ptxas_tval/domain.py)
accepts only sm_89, so it cannot certify this T4 workload unchanged.

The required checker must bind original cubin section bytes, architecture and
relocations to decoded instruction words; account for every operand modifier,
predicate and control field; then prove each reachable kernel against its
specified launch and memory contract. Unknown fields, indirect targets or
unsupported instruction modes must remain explicit failures. Re-encoding
agreement and device tests help find mistakes but do not replace that proof.

## Execution measurements

BAUX CUDA kernels written in Rust with CUDA-Oxide. T4 (`sm_75`) and A100
(`sm_80`) have hardware-validated stage measurements. The launcher also selects
H100 (`sm_90a`), but H100 hardware remains unmeasured. The tensor-core model
schedule is still the Turing-derived schedule: the A100 result establishes
compatibility and measured behavior, not an Ampere-optimized schedule. Complete
BO transaction evidence remains T4-only. CUDA is built separately from the CPU,
Metal, and OpenCL code.

The target is a subsecond BO pre-training round with a million-token context
window, including generation, scoring, evaluation, and the optimizer update.
Prompt occupancy and output length are independent workload counts; a million
new tokens per second is not required. The packed executor supports at most
64K. The streamed executor allocates bounded activations and distinct KV for
each recurrent visit. Capacity support alone does not establish latency or
learned long-context capability.

The `bo-generation-t4-v1` measurement is invalid as a BO/pre-training result:
it interpreted FP16 checkpoint bits as BF16, perturbed only the readout, and
used a throughput-only reward. Its command has been removed. The historical JSON
is retained alongside its invalidation notice under `results/cuda/`.

PTX synthesis can be selected for the checkpoint executor's vocabulary readout:

```sh
BAUX_SYNTH_GEMM=32 ./baux cuda generate-t4 --unroll 1 --out results/cuda/fresh-synth-run
```

The run checks boundary tiles against independent CPU FP16 arithmetic and
compares staging depths 8, 16, 32, and 64 with CUDA-Oxide on the same T4.
`gemm.json` records kernel comparisons; `result.json` and `tokens.json` record
generation. Default execution retains CUDA-Oxide until complete-generation
parity and timing justify a different schedule. The zero-output hierarchical
PTX stub has been removed.

T4 comparison on 2026-10-05: the replacement synthesized GEMM matched independent
CPU edge cases and CUDA-Oxide exactly on all four projection shapes. Selecting
it for readout produced identical 3,968-token outputs and accepted frontiers in
four alternating full-generation comparisons. CUDA-Oxide took 4.809–4.861 s;
synthesized readout took 4.766–4.838 s, a paired improvement of 0.4–0.9%.
Both used a 128-token zero-ID prompt and 21 repair waves at 4K. These are
execution measurements, not pre-training or text-quality results. Artifacts:
`results/cuda/generation-t4-synth-readout/{gemm,result,tokens}.json`.

The vocabulary readout now reuses an 8 MiB buffer in 512-row chunks. The former
full-context allocation would consume 16 GiB at 1,048,576 rows. On the T4,
chunking preserved all 3,968 generated tokens and 21 accepted frontiers; paired
generation times were 4.725/4.745 s versus 4.751/4.761 s with full logits.
Artifacts: `results/cuda/generation-t4-bounded-readout/`. This removes one memory
obstacle. Streamed execution now retains the accepted prefix within each
generation invocation and rebuilds KV at the start of the next invocation.

Measured on 2026-10-05: at 4K, 512-row streaming preserved all generated tokens
and accepted frontiers while reducing evaluated positions from 86,016 to
49,152 and device time from 4.122 s to 3.129 s. At 64K with a 64,512-token
zero-ID fixture prompt and 1,024 new tokens, streaming matched the packed
reference exactly: 69,632 versus 327,680 evaluated positions and 3.446 versus
15.001 s. Artifacts: `results/cuda/generation-t4-streamed-v1/` and
`results/cuda/generation-t4-streamed-64k-v1/`. These first measurements exclude
workspace initialization. Subsequent streamed wall timings include it;
checkpoint loading remains outside generation timing.

```sh
./baux cuda generate-t4 --context 65536 --prompt 64512 --chunk 512 --unroll 1 --out results/cuda/fresh-streamed-run
```

The benchmark compares streamed tokens and frontiers to packed execution through
64K. Zero and varied token-ID prompts are numerical fixtures, not a language
quality evaluation or a complete BO round.

The combined implementation also passed on the first 64,512 positions of the
checkpoint's actual pre-training corpus, followed by 1,024 newly generated
tokens. At 64K on T4, packed execution took 45.215 s and streaming took 4.415 s
device / 4.580 s wall including workspace initialization. Both produced exactly
the same tokens and 15 accepted frontiers. Evaluated positions fell from
983,040 to 78,848. This is one paired comparison, not a latency distribution.
Artifacts and decoded output: `results/cuda/generation-t4-corpus-64k/`;
source/data hashes and the exact prompt: the adjacent `-source/` directory.
The completion remains incoherent. This is execution progress, not a trained
long-context capability or a subsecond complete BO round.

The complete causal generator also executed at 1,048,576 context positions on
T4: 1,047,552 actual corpus tokens followed by 1,024 new tokens. With 512-row
chunks, it took 59.218 s device / 60.484 s wall, including workspace setup;
checkpoint load plus generation took 65.127 s. The first forward wave consumed
58.401 s; the remaining verification work consumed 0.817 s. Fourteen waves
evaluated 1,061,376 positions. A 1,024-row comparison took 54.885 s device and
produced identical tokens and accepted frontiers. This is chunk invariance,
not an independent packed reference at 1M. The completion remains incoherent.
Artifacts: `results/cuda/generation-t4-corpus-1m/`, with prompt provenance,
source hashes, command and log in the adjacent `-source/` directory. Fresh
context processing dominates this workload; subsecond BO remains unachieved.

The 4,096-row follow-up reduced device time to 52.025 s versus 60.956 s for
its paired 512-row reference (14.7% lower). All tokens and frontiers matched
both the paired reference and the saved baseline. Reusing constructor scratch
removed duplicate allocation and RoPE construction: wall time was 52.042 s,
with 49.367 s in the first wave and 2.659 s afterward. Larger repair chunks
increased evaluated positions to 1,101,824, so prefill and repair chunk sizes
should be tuned separately. Sampled middle-chunk time was 34.4% selected
attention, 33.5% MoE, 16.6% mHC, and 6.9% indexing. These percentages describe
one sampled chunk, not a whole-context trace. Artifacts:
`results/cuda/generation-t4-corpus-1m-chunk4096/`. Profiling was enabled in both
arms; this is a single paired comparison, not a latency distribution.

Prefill and repair chunk sizes can now differ. Set
`BAUX_CUDA_REPAIR_CHUNK=512` with `--chunk 4096` to retain large first-wave
batches and use smaller suffix batches. The effective repair size is capped
at the allocated chunk size. The default retains the same size for both.
This passed exact packed-token and frontier comparisons at 4K and 64K, and
streamed chunk invariance plus saved-baseline token equality at 1M.
At 1M the 4096/512 schedule took 51.220 s device / 51.338 s wall versus a
paired 512/512 reference's 61.679 s device. It evaluated 1,061,376 positions;
the first wave took 50.369 s and subsequent verification 0.850 s. This remains
a generation-only result. Artifact: `results/cuda/generation-t4-schedule-1m/`.

Set `BAUX_CUDA_PROMPT` to a JSON token-ID array when using `generate-t4` to
replace the zero-ID fixture. Its length must equal `--prompt`, and IDs must use
the checkpoint's 8,192-token vocabulary. The Rust runner uploads the array and
records `prompt_source=token_file`; it does not repeat or pad the input.
Set `BAUX_CUDA_PROFILE=1` to record stage timings for the first, middle and last
chunks of the first streamed wave. These are sampled chunk profiles, not sums
over the whole context. Profiling is disabled by default.

Inspect current coverage on any development host:

```sh
./baux cuda inspect
```

On a machine with the CUDA-Oxide toolchain installed, use the repository CLI:

```sh
./baux cuda setup
./baux cuda build
./baux cuda parity
./baux cuda resident
./baux cuda bench
./baux cuda sanitize
```

`setup` installs the pinned Linux x86-64 CUDA-Oxide toolchain on a fresh root
host with CUDA 13.0 and an NVIDIA runtime. Repository CUDA builds and runs use
the remaining `./baux cuda` commands.

The CUDA-Oxide revision is pinned in `Cargo.toml`; Rust and LLVM versions are in
`backends/cuda/rust-toolchain.toml` and `tools/cuda-setup`.

T4 verification on 2026-10-03: 36 parity cases passed, resident search passed,
Compute Sanitizer reported zero errors, and the 16,777,216-element perturbation
benchmark ran in 0.289 ms at 54.1 GiB/s. This measures the resident search
kernel, not the complete 64K generation and scoring loop.

The recurrent checkpoint executor and accepted-prefix generator use the same
CLI on the NVIDIA host:

```sh
./baux cuda model-check
./baux cuda model checkpoint.safetensors tokens.json fresh-output
./baux cuda generate checkpoint.safetensors prompt.json fresh-generation
```

JSON inputs contain vocabulary IDs in an array. `model` predicts one token per
input row; `generate` retains the prompt and generates the remaining positions.
Both accept context, recurrent visits, temperature, and seed after the output
directory. `generate` also accepts repair unroll (default 2). Packed context
must be a power of two from 4,096 to 65,536; inference visits range from one to four.
Output directories must be new. Artifacts include token IDs and measured timing;
`model` also saves normalized FP16 hidden states for numerical comparison.
`generate` accepts a final chunk size (zero selects packed execution); streamed
chunks are powers of two from 64 through 4,096. Streamed context capacity is
bounded at 1,048,576 positions. It includes the prompt and generated output.

The checkpoint format is `baux.fbt-pisa1-looped-mhc4-rope.v1`. The implementation
uses all 625 routed experts plus the shared expert. Repairs remain on-device
between passes in one submission batch; the host reads the accepted frontier
between batches. These commands do not yet run the Bayesian optimization loop.

T4 measurements on 2026-10-04: the seven-visit 4K analytical full-model fixture
used 496.120 ms of device time. The routed-attention slice's median was
50.847 ms device / 51.876 ms wall. The 3,968-token sampled fixture matched for
repair unroll 1 and 2. These measurements do not establish a 200 ms complete
round, 64K throughput, or checkpoint agreement with Metal.

Persistent PISA attention supports contexts through 1,048,576 tokens with a
separate KV cache and bounded query scratch:

```sh
./baux cuda context 1048576 128 7
./baux cuda sanitize context 1048576 128 3
```

Arguments are context capacity, query count, and timing repetitions. Query
count must be a multiple of four through 4,096; the numerical probe requires
at most half the context. The JSON report distinguishes cached attention from
complete model execution. It verifies prefix append, causal selection,
selected-support attention, and incremental tree repair against the Rust
reference shared with Metal. FP16 KV consumes 256 MiB per layer visit at 1M,
with approximately 4 MiB of tree summaries. The complete checkpoint executor
supports bounded streamed activations. Packed comparisons reach 64K; complete
1M generation has passed chunk-size invariance as measured above. See
[the context design and research](../docs/kernel-architecture-plan.md#million-token-context).

## Learned block diffusion

The new source accepts the distinct checkpoint format
`baux.fbt-pisa1-diffusion-mhc4-rope.v1`, including learned mask and index weights:

```sh
./baux cuda diffusion-check
./baux cuda diffusion CHECKPOINT PROMPT_JSON FRESH_OUTPUT 4096 2 0.8 17 2 128 true
./baux cuda sanitize diffusion-check
```

Arguments after the output directory are total context positions, recurrent
visits, temperature, seed, denoising steps, visibility block, and soft-input mode.
The context includes the prompt: 4,096 positions with a 128-token prompt produce
3,968 new tokens. The packed executor requires a power-of-two context from 4K
through 64K, aligned prompt/output blocks, and one to four recurrent visits.

Diffusion replaces exact autoregressive repairs with parallel denoising passes.
The kernels implement learned-mask input, confidence-weighted token interpolation,
block visibility, learned query projection, and sampled output confidence.
Selection is independent per query; Metal's grouped refinement and drift reuse
are not implemented here. The fixture compares sampled output with an analytical
uniform-weight model. It does not establish trained-checkpoint agreement.

The CUDA diffusion fixture passed on T4 on 2026-10-05. A trained diffusion
checkpoint and integration as a target-verified drafter remain outstanding.
Complete CUDA BO rounds still require integration.
