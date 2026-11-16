/* Web Worker hosting self-hosted Pyodide + baux_rust WASM + OxCaml + sync WebGPU RPC */
"use strict";

importScripts("./pyodide/pyodide.js");

let runtimePromise = null;
let pendingInterruptBuffer = null;
let interruptView = null;
let wantOpfsMount = true;
let opfsMounted = false;
let opfsHandle = null;

let webgpuWorker = null;
let webgpuReady = false;
let webgpuAdapterLabel = "";
let webgpuMaxBufferBytes = 0;
let webgpuMaxStorageBytes = 0;
let rpcSab = null;
let rpcCtrlI32 = null;
let rpcCtrlU32 = null;
let rpcCtrlF32 = null;
let rpcSeq = 0;

function buildCaps() {
  return {
    cpu_wasm_simd: true,
    webgpu: Boolean(webgpuReady),
    webgpu_adapter: webgpuAdapterLabel || "",
    opfs_mounted: Boolean(opfsMounted),
    coop_coep: Boolean(self.crossOriginIsolated),
  };
}

function applyInterruptBuffer(pyodide) {
  if (
    pendingInterruptBuffer &&
    typeof SharedArrayBuffer === "function" &&
    pendingInterruptBuffer instanceof SharedArrayBuffer &&
    pyodide &&
    typeof pyodide.setInterruptBuffer === "function"
  ) {
    interruptView = new Int32Array(pendingInterruptBuffer);
    pyodide.setInterruptBuffer(interruptView);
  }
}

function clearStaleInterruptSignal() {
  if (interruptView && typeof Atomics === "object") {
    Atomics.store(interruptView, 0, 0);
  }
}

async function flushOpfs() {
  if (opfsHandle && typeof opfsHandle.syncfs === "function") {
    try {
      await opfsHandle.syncfs();
    } catch (_) {
      // OPFS quota or transient flush failure should not mask REPL execution output.
    }
  }
}

function withCacheVer(url, spec) {
  if (spec && typeof spec.sha256 === "string" && spec.sha256.length >= 12) {
    return url + (url.includes("?") ? "&v=" : "?v=") + spec.sha256.slice(0, 12);
  }
  return url;
}

async function loadVerifiedScript(url, spec) {
  const resolvedUrl = withCacheVer(url, spec);
  if (
    spec &&
    typeof spec.size === "number" &&
    typeof spec.sha256 === "string" &&
    spec.sha256.length === 64
  ) {
    await fetchVerifiedBytes(resolvedUrl, spec.size, spec.sha256);
  }
  importScripts(resolvedUrl);
}

function checkedPosInt(val) {
  const n = Number(val);
  if (!Number.isSafeInteger(n) || n <= 0 || n > 0x7fffffff) {
    return 0;
  }
  return n >>> 0;
}

function checkedNonNegInt(val) {
  const n = Number(val);
  if (!Number.isSafeInteger(n) || n < 0 || n > 0x7fffffff) {
    return -1;
  }
  return n >>> 0;
}

function checkedMulInt(a, b) {
  const prod = a * b;
  if (!Number.isSafeInteger(prod) || prod <= 0 || prod > 0x0fffffff) {
    return 0;
  }
  return prod;
}

function align16Bytes(n) {
  return Math.ceil(n / 16) * 16;
}

function bindRpcSab(nextSab) {
  const rpc = BauxWebGpu.SAB_RPC;
  rpcSab = nextSab;
  rpcCtrlI32 = new Int32Array(nextSab, 0, rpc.HEADER_I32_LEN);
  rpcCtrlU32 = new Uint32Array(nextSab, 0, rpc.HEADER_I32_LEN);
  rpcCtrlF32 = new Float32Array(nextSab, 0, rpc.HEADER_I32_LEN);
}

function ensureSabCapacity(neededDataBytes) {
  const rpc = BauxWebGpu.SAB_RPC;
  const alignedNeeded = align16Bytes(neededDataBytes);
  const currentStaging = rpcSab ? rpcSab.byteLength - rpc.HEADER_BYTES : 0;
  if (rpcSab && alignedNeeded <= currentStaging) {
    return { ok: true, swappedSab: null };
  }
  const hardLimit =
    webgpuMaxBufferBytes > 0
      ? Math.min(rpc.MAX_STAGING_BYTES, webgpuMaxBufferBytes)
      : rpc.MAX_STAGING_BYTES;
  if (alignedNeeded > hardLimit) {
    return { ok: false, swappedSab: null };
  }
  const targetStaging = Math.min(
    hardLimit,
    Math.max(rpc.DEFAULT_STAGING_BYTES, alignedNeeded, currentStaging * 2)
  );
  try {
    const nextSab = new SharedArrayBuffer(rpc.HEADER_BYTES + targetStaging);
    bindRpcSab(nextSab);
    return { ok: true, swappedSab: nextSab };
  } catch (_) {
    return { ok: false, swappedSab: null };
  }
}

function validatePtrAndSlice(mod, ptr, elemCount, isFloat) {
  const bytePtr = ptr >>> 0;
  if (bytePtr === 0 || (bytePtr & 3) !== 0 || elemCount <= 0) {
    return null;
  }
  const byteLen = elemCount * 4;
  const heapU8 = mod.HEAPU8;
  if (!heapU8 || bytePtr + byteLen > heapU8.byteLength) {
    return null;
  }
  const wordOff = bytePtr >>> 2;
  const view = isFloat ? mod.HEAPF32 : mod.HEAPU32;
  if (!view || wordOff + elemCount > view.length) {
    return null;
  }
  return view.subarray(wordOff, wordOff + elemCount);
}

function syncWebGpuRpc(op, neededDataBytes, prepareFn, readbackFn) {
  const rpc = BauxWebGpu.SAB_RPC;
  if (!webgpuReady || !webgpuWorker || !rpcSab || !rpcCtrlI32) {
    return { rc: rpc.ERR_UNAVAILABLE, handle: 0 };
  }
  const cap = ensureSabCapacity(neededDataBytes);
  if (!cap.ok) {
    return { rc: rpc.ERR_STAGING_OVERFLOW, handle: 0 };
  }

  rpcSeq = (rpcSeq + 1) >>> 0;
  if (rpcSeq === 0) {
    rpcSeq = 1;
  }

  rpcCtrlI32[rpc.IDX_OP] = op | 0;
  rpcCtrlI32[rpc.IDX_RESULT_I32] = 0;
  rpcCtrlU32[rpc.IDX_RESULT_U32] = 0;
  if (!prepareFn(rpcSab, rpcCtrlI32, rpcCtrlU32, rpcCtrlF32)) {
    return { rc: rpc.ERR_INVALID_ARGS, handle: 0 };
  }

  Atomics.store(rpcCtrlU32, rpc.IDX_SEQ, rpcSeq);
  Atomics.store(rpcCtrlI32, rpc.IDX_STATE, rpc.STATE_PENDING);

  if (cap.swappedSab) {
    webgpuWorker.postMessage({ type: "dispatch", seq: rpcSeq, sab: cap.swappedSab });
  } else {
    webgpuWorker.postMessage({ type: "dispatch", seq: rpcSeq });
  }

  const deadline = Date.now() + 120000;
  while (true) {
    const curState = Atomics.load(rpcCtrlI32, rpc.IDX_STATE);
    if (curState === rpc.STATE_DONE || curState === rpc.STATE_ERROR) {
      break;
    }
    const remaining = deadline - Date.now();
    if (remaining <= 0) {
      webgpuReady = false;
      webgpuAdapterLabel = "";
      return { rc: rpc.ERR_TIMEOUT, handle: 0 };
    }
    const waitStatus = Atomics.wait(
      rpcCtrlI32,
      rpc.IDX_STATE,
      rpc.STATE_PENDING,
      Math.min(remaining, 10000)
    );
    if (!webgpuReady) {
      return { rc: rpc.ERR_UNAVAILABLE, handle: 0 };
    }
    if (waitStatus === "timed-out" && Date.now() >= deadline) {
      webgpuReady = false;
      webgpuAdapterLabel = "";
      return { rc: rpc.ERR_TIMEOUT, handle: 0 };
    }
  }

  const rc = Atomics.load(rpcCtrlI32, rpc.IDX_RESULT_I32) | 0;
  const handle = Atomics.load(rpcCtrlU32, rpc.IDX_RESULT_U32) >>> 0;
  Atomics.store(rpcCtrlI32, rpc.IDX_STATE, rpc.STATE_IDLE);
  if (rc !== rpc.ERR_OK) {
    return { rc, handle: 0 };
  }
  if (readbackFn && !readbackFn(rpcSab, rpcCtrlU32)) {
    return { rc: rpc.ERR_INVALID_ARGS, handle: 0 };
  }
  return { rc: rpc.ERR_OK, handle };
}

function registerWebGpuSymbols(pyodide) {
  const mod = pyodide._module;
  if (!mod) {
    throw new Error("Pyodide Emscripten _module unavailable");
  }
  const rpc = BauxWebGpu.SAB_RPC;

  function bauxgpu_available() {
    return webgpuReady ? 1 : 0;
  }
  bauxgpu_available.sig = "i";

  function bauxgpu_alloc(bytes) {
    if (!webgpuReady) {
      return 0;
    }
    const byteSize = checkedPosInt(bytes);
    if (byteSize === 0) {
      return 0;
    }
    const res = syncWebGpuRpc(
      rpc.OP_BUFFER_ALLOC,
      16,
      (_sab, _i32, u32) => {
        u32[rpc.IDX_P0] = byteSize;
        return true;
      },
      null
    );
    return res.rc === rpc.ERR_OK ? res.handle >>> 0 : 0;
  }
  bauxgpu_alloc.sig = "ii";

  function bauxgpu_free(handle) {
    const h = handle >>> 0;
    if (h === 0) {
      return rpc.ERR_INVALID_ARGS;
    }
    const res = syncWebGpuRpc(
      rpc.OP_BUFFER_FREE,
      16,
      (_sab, _i32, u32) => {
        u32[rpc.IDX_P0] = h;
        return true;
      },
      null
    );
    return res.rc | 0;
  }
  bauxgpu_free.sig = "ii";

  function bauxgpu_readf32(handle, outPtr, len) {
    const h = handle >>> 0;
    const elemCount = checkedPosInt(len);
    if (h === 0 || elemCount === 0) {
      return rpc.ERR_INVALID_ARGS;
    }
    if (!validatePtrAndSlice(mod, outPtr, elemCount, true)) {
      return rpc.ERR_INVALID_ARGS;
    }
    const outOff = rpc.HEADER_BYTES;
    const needed = align16Bytes(elemCount * 4);
    const res = syncWebGpuRpc(
      rpc.OP_BUFFER_READ_F32,
      needed,
      (_sab, _i32, u32) => {
        u32[rpc.IDX_P0] = h;
        u32[rpc.IDX_OFF0] = outOff;
        u32[rpc.IDX_LEN0] = elemCount;
        return true;
      },
      (sab) => {
        const dst = validatePtrAndSlice(mod, outPtr, elemCount, true);
        if (!dst) {
          return false;
        }
        dst.set(new Float32Array(sab, outOff, elemCount));
        return true;
      }
    );
    return res.rc | 0;
  }
  bauxgpu_readf32.sig = "iiii";

  function bauxgpu_exactdist(qRaw, nRaw, dRaw, queriesPtr, rowsPtr, invScale2Ptr, outDist2Ptr) {
    const q = checkedPosInt(qRaw);
    const n = checkedPosInt(nRaw);
    const d = checkedPosInt(dRaw);
    const qd = checkedMulInt(q, d);
    const nd = checkedMulInt(n, d);
    const qn = checkedMulInt(q, n);
    if (q === 0 || n === 0 || d === 0 || qd === 0 || nd === 0 || qn === 0) {
      return rpc.ERR_INVALID_ARGS;
    }
    const qIn = validatePtrAndSlice(mod, queriesPtr, qd, true);
    const rIn = validatePtrAndSlice(mod, rowsPtr, nd, true);
    const hasScale = (invScale2Ptr >>> 0) !== 0;
    const sIn = hasScale ? validatePtrAndSlice(mod, invScale2Ptr, d, true) : null;
    const oCheck = validatePtrAndSlice(mod, outDist2Ptr, qn, true);
    if (!qIn || !rIn || (hasScale && !sIn) || !oCheck) {
      return rpc.ERR_INVALID_ARGS;
    }

    const off0 = rpc.HEADER_BYTES;
    const off1 = off0 + align16Bytes(qd * 4);
    const off2 = off1 + align16Bytes(nd * 4);
    const off3 = off2 + (hasScale ? align16Bytes(d * 4) : 0);
    const needed = off3 + align16Bytes(qn * 4) - rpc.HEADER_BYTES;

    const res = syncWebGpuRpc(
      rpc.OP_EXACT_DIST,
      needed,
      (sab, _i32, u32) => {
        u32[rpc.IDX_P0] = q;
        u32[rpc.IDX_P1] = n;
        u32[rpc.IDX_P2] = d;
        u32[rpc.IDX_P3] = hasScale ? 1 : 0;
        u32[rpc.IDX_OFF0] = off0;
        u32[rpc.IDX_LEN0] = qd;
        u32[rpc.IDX_OFF1] = off1;
        u32[rpc.IDX_LEN1] = nd;
        u32[rpc.IDX_OFF2] = off2;
        u32[rpc.IDX_LEN2] = hasScale ? d : 0;
        u32[rpc.IDX_OFF3] = off3;
        u32[rpc.IDX_LEN3] = qn;
        new Float32Array(sab, off0, qd).set(qIn);
        new Float32Array(sab, off1, nd).set(rIn);
        if (hasScale) {
          new Float32Array(sab, off2, d).set(sIn);
        }
        return true;
      },
      (sab) => {
        const dst = validatePtrAndSlice(mod, outDist2Ptr, qn, true);
        if (!dst) {
          return false;
        }
        dst.set(new Float32Array(sab, off3, qn));
        return true;
      }
    );
    return res.rc | 0;
  }
  bauxgpu_exactdist.sig = "iiiiiiii";

  function bauxgpu_densedist2(qRaw, nRaw, dRaw, aPtr, bPtr, outDist2Ptr) {
    const q = checkedPosInt(qRaw);
    const n = checkedPosInt(nRaw);
    const d = checkedPosInt(dRaw);
    const qd = checkedMulInt(q, d);
    const nd = checkedMulInt(n, d);
    const qn = checkedMulInt(q, n);
    if (q === 0 || n === 0 || d === 0 || qd === 0 || nd === 0 || qn === 0) {
      return rpc.ERR_INVALID_ARGS;
    }
    const aIn = validatePtrAndSlice(mod, aPtr, qd, true);
    const bIn = validatePtrAndSlice(mod, bPtr, nd, true);
    const oCheck = validatePtrAndSlice(mod, outDist2Ptr, qn, true);
    if (!aIn || !bIn || !oCheck) {
      return rpc.ERR_INVALID_ARGS;
    }

    const off0 = rpc.HEADER_BYTES;
    const off1 = off0 + align16Bytes(qd * 4);
    const off2 = off1 + align16Bytes(nd * 4);
    const needed = off2 + align16Bytes(qn * 4) - rpc.HEADER_BYTES;

    const res = syncWebGpuRpc(
      rpc.OP_DENSE_DIST2,
      needed,
      (sab, _i32, u32) => {
        u32[rpc.IDX_P0] = q;
        u32[rpc.IDX_P1] = n;
        u32[rpc.IDX_P2] = d;
        u32[rpc.IDX_OFF0] = off0;
        u32[rpc.IDX_LEN0] = qd;
        u32[rpc.IDX_OFF1] = off1;
        u32[rpc.IDX_LEN1] = nd;
        u32[rpc.IDX_OFF2] = off2;
        u32[rpc.IDX_LEN2] = qn;
        new Float32Array(sab, off0, qd).set(aIn);
        new Float32Array(sab, off1, nd).set(bIn);
        return true;
      },
      (sab) => {
        const dst = validatePtrAndSlice(mod, outDist2Ptr, qn, true);
        if (!dst) {
          return false;
        }
        dst.set(new Float32Array(sab, off2, qn));
        return true;
      }
    );
    return res.rc | 0;
  }
  bauxgpu_densedist2.sig = "iiiiiii";

  function bauxgpu_knntopk(qRaw, nRaw, kRaw, skipRaw, dist2Ptr, outDistPtr, outIdxPtr) {
    const q = checkedPosInt(qRaw);
    const n = checkedPosInt(nRaw);
    const k = checkedPosInt(kRaw);
    const skip = checkedNonNegInt(skipRaw);
    const qn = checkedMulInt(q, n);
    const qk = checkedMulInt(q, k);
    if (q === 0 || n === 0 || k === 0 || skip < 0 || k > n || skip > n || qn === 0 || qk === 0) {
      return rpc.ERR_INVALID_ARGS;
    }
    const dIn = validatePtrAndSlice(mod, dist2Ptr, qn, true);
    const odCheck = validatePtrAndSlice(mod, outDistPtr, qk, true);
    const oiCheck = validatePtrAndSlice(mod, outIdxPtr, qk, false);
    if (!dIn || !odCheck || !oiCheck) {
      return rpc.ERR_INVALID_ARGS;
    }

    const off0 = rpc.HEADER_BYTES;
    const off1 = off0 + align16Bytes(qn * 4);
    const off2 = off1 + align16Bytes(qk * 4);
    const needed = off2 + align16Bytes(qk * 4) - rpc.HEADER_BYTES;

    const res = syncWebGpuRpc(
      rpc.OP_KNN_TOPK,
      needed,
      (sab, _i32, u32) => {
        u32[rpc.IDX_P0] = q;
        u32[rpc.IDX_P1] = n;
        u32[rpc.IDX_P2] = k;
        u32[rpc.IDX_P3] = skip >>> 0;
        u32[rpc.IDX_OFF0] = off0;
        u32[rpc.IDX_LEN0] = qn;
        u32[rpc.IDX_OFF1] = off1;
        u32[rpc.IDX_LEN1] = qk;
        u32[rpc.IDX_OFF2] = off2;
        u32[rpc.IDX_LEN2] = qk;
        new Float32Array(sab, off0, qn).set(dIn);
        return true;
      },
      (sab) => {
        const dstDist = validatePtrAndSlice(mod, outDistPtr, qk, true);
        const dstIdx = validatePtrAndSlice(mod, outIdxPtr, qk, false);
        if (!dstDist || !dstIdx) {
          return false;
        }
        dstDist.set(new Float32Array(sab, off1, qk));
        dstIdx.set(new Uint32Array(sab, off2, qk));
        return true;
      }
    );
    return res.rc | 0;
  }
  bauxgpu_knntopk.sig = "iiiiiiii";

  function bauxgpu_posterior(
    queriesRaw,
    nRaw,
    metricsRaw,
    inputKRaw,
    kRaw,
    skipRaw,
    epistemicScale,
    aleatoricScale,
    epsilon,
    observationNoiseRaw,
    distancesPtr,
    idsPtr,
    outcomesPtr,
    variancesPtr,
    scalesPtr,
    outSummaryPtr,
    outSelectedPtr,
    outWeightsPtr
  ) {
    const queries = checkedPosInt(queriesRaw);
    const n = checkedPosInt(nRaw);
    const metrics = checkedPosInt(metricsRaw);
    const inputK = checkedPosInt(inputKRaw);
    const k = checkedPosInt(kRaw);
    const skip = checkedNonNegInt(skipRaw);
    const qkIn = checkedMulInt(queries, inputK);
    const nm = checkedMulInt(n, metrics);
    const qm = checkedMulInt(queries, metrics);
    const sumLen = checkedMulInt(qm, 4);
    const qk = checkedMulInt(queries, k);
    const qkm = checkedMulInt(qk, metrics);
    if (
      queries === 0 ||
      n === 0 ||
      metrics === 0 ||
      inputK === 0 ||
      k === 0 ||
      skip < 0 ||
      skip + k > inputK ||
      qkIn === 0 ||
      nm === 0 ||
      sumLen === 0 ||
      qk === 0 ||
      qkm === 0
    ) {
      return rpc.ERR_INVALID_ARGS;
    }

    const distIn = validatePtrAndSlice(mod, distancesPtr, qkIn, true);
    const idsIn = validatePtrAndSlice(mod, idsPtr, qkIn, false);
    const outcIn = validatePtrAndSlice(mod, outcomesPtr, nm, true);
    const scaleIn = validatePtrAndSlice(mod, scalesPtr, metrics, true);
    const hasVar = (variancesPtr >>> 0) !== 0;
    const varIn = hasVar ? validatePtrAndSlice(mod, variancesPtr, nm, true) : null;
    const sumCheck = validatePtrAndSlice(mod, outSummaryPtr, sumLen, true);
    const selCheck = validatePtrAndSlice(mod, outSelectedPtr, qk, false);
    const wCheck = validatePtrAndSlice(mod, outWeightsPtr, qkm, true);
    if (
      !distIn ||
      !idsIn ||
      !outcIn ||
      !scaleIn ||
      (hasVar && !varIn) ||
      !sumCheck ||
      !selCheck ||
      !wCheck
    ) {
      return rpc.ERR_INVALID_ARGS;
    }

    const off0 = rpc.HEADER_BYTES;
    const off1 = off0 + align16Bytes(qkIn * 4);
    const off2 = off1 + align16Bytes(qkIn * 4);
    const off3 = off2 + align16Bytes(nm * 4);
    const off4 = off3 + align16Bytes(metrics * 4);
    const off5 = off4 + (hasVar ? align16Bytes(nm * 4) : 0);
    const off6 = off5 + align16Bytes(sumLen * 4);
    const off7 = off6 + align16Bytes(qk * 4);
    const needed = off7 + align16Bytes(qkm * 4) - rpc.HEADER_BYTES;

    const res = syncWebGpuRpc(
      rpc.OP_DEVICE_POSTERIOR,
      needed,
      (sab, _i32, u32, f32) => {
        u32[rpc.IDX_P0] = queries;
        u32[rpc.IDX_P1] = n;
        u32[rpc.IDX_P2] = metrics;
        u32[rpc.IDX_P3] = inputK;
        u32[rpc.IDX_P4] = k;
        u32[rpc.IDX_P5] = skip >>> 0;
        u32[rpc.IDX_P6] = observationNoiseRaw ? 1 : 0;
        u32[rpc.IDX_P7] = hasVar ? 1 : 0;
        f32[rpc.IDX_F0] = Number(epistemicScale);
        f32[rpc.IDX_F1] = Number(aleatoricScale);
        f32[rpc.IDX_F2] = Number(epsilon);
        u32[rpc.IDX_OFF0] = off0;
        u32[rpc.IDX_LEN0] = qkIn;
        u32[rpc.IDX_OFF1] = off1;
        u32[rpc.IDX_LEN1] = qkIn;
        u32[rpc.IDX_OFF2] = off2;
        u32[rpc.IDX_LEN2] = nm;
        u32[rpc.IDX_OFF3] = off3;
        u32[rpc.IDX_LEN3] = metrics;
        u32[rpc.IDX_OFF4] = off4;
        u32[rpc.IDX_LEN4] = hasVar ? nm : 0;
        u32[rpc.IDX_OFF5] = off5;
        u32[rpc.IDX_LEN5] = sumLen;
        u32[rpc.IDX_OFF6] = off6;
        u32[rpc.IDX_LEN6] = qk;
        u32[rpc.IDX_OFF7] = off7;
        u32[rpc.IDX_LEN7] = qkm;
        new Float32Array(sab, off0, qkIn).set(distIn);
        new Uint32Array(sab, off1, qkIn).set(idsIn);
        new Float32Array(sab, off2, nm).set(outcIn);
        new Float32Array(sab, off3, metrics).set(scaleIn);
        if (hasVar) {
          new Float32Array(sab, off4, nm).set(varIn);
        }
        return true;
      },
      (sab) => {
        const dstSum = validatePtrAndSlice(mod, outSummaryPtr, sumLen, true);
        const dstSel = validatePtrAndSlice(mod, outSelectedPtr, qk, false);
        const dstW = validatePtrAndSlice(mod, outWeightsPtr, qkm, true);
        if (!dstSum || !dstSel || !dstW) {
          return false;
        }
        dstSum.set(new Float32Array(sab, off5, sumLen));
        dstSel.set(new Uint32Array(sab, off6, qk));
        dstW.set(new Float32Array(sab, off7, qkm));
        return true;
      }
    );
    return res.rc | 0;
  }
  bauxgpu_posterior.sig = "iiiiiiifffiiiiiiiii";

  function bauxgpu_denseapply(mRaw, nRaw, kRaw, xPtr, wPtr, biasPtr, yPtr) {
    const m = checkedPosInt(mRaw);
    const n = checkedPosInt(nRaw);
    const k = checkedPosInt(kRaw);
    const mk = checkedMulInt(m, k);
    const nk = checkedMulInt(n, k);
    const mn = checkedMulInt(m, n);
    if (m === 0 || n === 0 || k === 0 || mk === 0 || nk === 0 || mn === 0) {
      return rpc.ERR_INVALID_ARGS;
    }
    const xIn = validatePtrAndSlice(mod, xPtr, mk, true);
    const wIn = validatePtrAndSlice(mod, wPtr, nk, true);
    const hasBias = (biasPtr >>> 0) !== 0;
    const bIn = hasBias ? validatePtrAndSlice(mod, biasPtr, n, true) : null;
    const yCheck = validatePtrAndSlice(mod, yPtr, mn, true);
    if (!xIn || !wIn || (hasBias && !bIn) || !yCheck) {
      return rpc.ERR_INVALID_ARGS;
    }

    const off0 = rpc.HEADER_BYTES;
    const off1 = off0 + align16Bytes(mk * 4);
    const off2 = off1 + align16Bytes(nk * 4);
    const off3 = off2 + (hasBias ? align16Bytes(n * 4) : 0);
    const needed = off3 + align16Bytes(mn * 4) - rpc.HEADER_BYTES;

    const res = syncWebGpuRpc(
      rpc.OP_DENSE_APPLY,
      needed,
      (sab, _i32, u32) => {
        u32[rpc.IDX_P0] = m;
        u32[rpc.IDX_P1] = n;
        u32[rpc.IDX_P2] = k;
        u32[rpc.IDX_P3] = hasBias ? 1 : 0;
        u32[rpc.IDX_OFF0] = off0;
        u32[rpc.IDX_LEN0] = mk;
        u32[rpc.IDX_OFF1] = off1;
        u32[rpc.IDX_LEN1] = nk;
        u32[rpc.IDX_OFF2] = off2;
        u32[rpc.IDX_LEN2] = hasBias ? n : 0;
        u32[rpc.IDX_OFF3] = off3;
        u32[rpc.IDX_LEN3] = mn;
        new Float32Array(sab, off0, mk).set(xIn);
        new Float32Array(sab, off1, nk).set(wIn);
        if (hasBias) {
          new Float32Array(sab, off2, n).set(bIn);
        }
        return true;
      },
      (sab) => {
        const dst = validatePtrAndSlice(mod, yPtr, mn, true);
        if (!dst) {
          return false;
        }
        dst.set(new Float32Array(sab, off3, mn));
        return true;
      }
    );
    return res.rc | 0;
  }
  bauxgpu_denseapply.sig = "iiiiiiii";

  function bauxgpu_weightselect(
    batchRaw,
    candidatesRaw,
    modeRaw,
    emitSoftmaxRaw,
    beta,
    temperature,
    minProb,
    muPtr,
    sePtr,
    zPtr,
    outScoresPtr,
    outProbsPtr,
    outBestIdxPtr,
    outBestScorePtr
  ) {
    const batch = checkedPosInt(batchRaw);
    const candidates = checkedPosInt(candidatesRaw);
    const mode = modeRaw >>> 0;
    const bc = checkedMulInt(batch, candidates);
    if (batch === 0 || candidates === 0 || mode > 2 || bc === 0) {
      return rpc.ERR_INVALID_ARGS;
    }
    const muIn = validatePtrAndSlice(mod, muPtr, bc, true);
    const seIn = validatePtrAndSlice(mod, sePtr, bc, true);
    const hasZ = mode === 1 && (zPtr >>> 0) !== 0;
    const zIn = hasZ ? validatePtrAndSlice(mod, zPtr, bc, true) : null;
    if (mode === 1 && !zIn) {
      return rpc.ERR_INVALID_ARGS;
    }
    const scoresCheck = validatePtrAndSlice(mod, outScoresPtr, bc, true);
    const probsCheck = validatePtrAndSlice(mod, outProbsPtr, bc, true);
    const idxCheck = validatePtrAndSlice(mod, outBestIdxPtr, batch, false);
    const bestCheck = validatePtrAndSlice(mod, outBestScorePtr, batch, true);
    if (!muIn || !seIn || !scoresCheck || !probsCheck || !idxCheck || !bestCheck) {
      return rpc.ERR_INVALID_ARGS;
    }

    const off0 = rpc.HEADER_BYTES;
    const off1 = off0 + align16Bytes(bc * 4);
    const off2 = off1 + align16Bytes(bc * 4);
    const off3 = off2 + (hasZ ? align16Bytes(bc * 4) : 0);
    const off4 = off3 + align16Bytes(bc * 4);
    const off5 = off4 + align16Bytes(bc * 4);
    const off6 = off5 + align16Bytes(batch * 4);
    const needed = off6 + align16Bytes(batch * 4) - rpc.HEADER_BYTES;

    const res = syncWebGpuRpc(
      rpc.OP_WEIGHT_SELECT,
      needed,
      (sab, _i32, u32, f32) => {
        u32[rpc.IDX_P0] = batch;
        u32[rpc.IDX_P1] = candidates;
        u32[rpc.IDX_P2] = mode;
        u32[rpc.IDX_P3] = emitSoftmaxRaw ? 1 : 0;
        u32[rpc.IDX_P4] = hasZ ? 1 : 0;
        f32[rpc.IDX_F0] = Number(beta);
        f32[rpc.IDX_F1] = Number(temperature);
        f32[rpc.IDX_F2] = Number(minProb);
        u32[rpc.IDX_OFF0] = off0;
        u32[rpc.IDX_LEN0] = bc;
        u32[rpc.IDX_OFF1] = off1;
        u32[rpc.IDX_LEN1] = bc;
        u32[rpc.IDX_OFF2] = off2;
        u32[rpc.IDX_LEN2] = hasZ ? bc : 0;
        u32[rpc.IDX_OFF3] = off3;
        u32[rpc.IDX_LEN3] = bc;
        u32[rpc.IDX_OFF4] = off4;
        u32[rpc.IDX_LEN4] = bc;
        u32[rpc.IDX_OFF5] = off5;
        u32[rpc.IDX_LEN5] = batch;
        u32[rpc.IDX_OFF6] = off6;
        u32[rpc.IDX_LEN6] = batch;
        new Float32Array(sab, off0, bc).set(muIn);
        new Float32Array(sab, off1, bc).set(seIn);
        if (hasZ) {
          new Float32Array(sab, off2, bc).set(zIn);
        }
        return true;
      },
      (sab) => {
        const dstScores = validatePtrAndSlice(mod, outScoresPtr, bc, true);
        const dstProbs = validatePtrAndSlice(mod, outProbsPtr, bc, true);
        const dstIdx = validatePtrAndSlice(mod, outBestIdxPtr, batch, false);
        const dstBest = validatePtrAndSlice(mod, outBestScorePtr, batch, true);
        if (!dstScores || !dstProbs || !dstIdx || !dstBest) {
          return false;
        }
        dstScores.set(new Float32Array(sab, off3, bc));
        dstProbs.set(new Float32Array(sab, off4, bc));
        dstIdx.set(new Uint32Array(sab, off5, batch));
        dstBest.set(new Float32Array(sab, off6, batch));
        return true;
      }
    );
    return res.rc | 0;
  }
  bauxgpu_weightselect.sig = "iiiiifffiiiiiii";

  const symbols = {
    bauxgpu_available,
    bauxgpu_alloc,
    bauxgpu_free,
    bauxgpu_readf32,
    bauxgpu_exactdist,
    bauxgpu_knntopk,
    bauxgpu_posterior,
    bauxgpu_denseapply,
    bauxgpu_densedist2,
    bauxgpu_weightselect,
  };

  if (typeof mod.mergeLibSymbols === "function") {
    mod.mergeLibSymbols(symbols, "bauxgpu");
  }
  mod.bauxgpu_available = bauxgpu_available;
  mod.bauxgpu_alloc = bauxgpu_alloc;
  mod.bauxgpu_free = bauxgpu_free;
  mod.bauxgpu_readf32 = bauxgpu_readf32;
  mod.bauxgpu_exactdist = bauxgpu_exactdist;
  mod.bauxgpu_knntopk = bauxgpu_knntopk;
  mod.bauxgpu_posterior = bauxgpu_posterior;
  mod.bauxgpu_denseapply = bauxgpu_denseapply;
  mod.bauxgpu_densedist2 = bauxgpu_densedist2;
  mod.bauxgpu_weightselect = bauxgpu_weightselect;
}

async function initWebGpuChannel(workerSpec) {
  const rpc = BauxWebGpu.SAB_RPC;
  webgpuReady = false;
  webgpuAdapterLabel = "";
  webgpuMaxBufferBytes = 0;
  webgpuMaxStorageBytes = 0;

  const canUseSubWorker =
    Boolean(self.crossOriginIsolated) &&
    typeof SharedArrayBuffer === "function" &&
    typeof Atomics === "object" &&
    typeof Atomics.wait === "function" &&
    typeof Worker === "function" &&
    typeof navigator !== "undefined" &&
    Boolean(navigator.gpu);

  if (!canUseSubWorker) {
    return;
  }

  try {
    if (
      workerSpec &&
      typeof workerSpec.size === "number" &&
      typeof workerSpec.sha256 === "string" &&
      workerSpec.sha256.length === 64
    ) {
      await fetchVerifiedBytes(
        withCacheVer("./" + workerSpec.path, workerSpec),
        workerSpec.size,
        workerSpec.sha256
      );
    }

    const initialSab = new SharedArrayBuffer(rpc.HEADER_BYTES + rpc.DEFAULT_STAGING_BYTES);
    bindRpcSab(initialSab);

    await new Promise((resolve) => {
      let settled = false;
      const done = () => {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve();
        }
      };

      const worker = new Worker(withCacheVer("./webgpu_worker.js", workerSpec));
      const timer = setTimeout(() => {
        webgpuReady = false;
        webgpuAdapterLabel = "";
        try {
          worker.terminate();
        } catch (_) {
          // Ignore worker terminate errors after init timeout.
        }
        webgpuWorker = null;
        done();
      }, 2500);

      worker.onmessage = (event) => {
        const msg = event.data || {};
        if (msg.type === "ready") {
          if (msg.available) {
            webgpuWorker = worker;
            webgpuReady = true;
            webgpuAdapterLabel = String(msg.adapterLabel || BauxWebGpu.formatAdapterLabel(msg.adapterInfo) || "");
            if (msg.limits && typeof msg.limits.maxBufferSize === "number") {
              webgpuMaxBufferBytes = msg.limits.maxBufferSize;
            }
            if (msg.limits && typeof msg.limits.maxStorageBufferBindingSize === "number") {
              webgpuMaxStorageBytes = msg.limits.maxStorageBufferBindingSize;
            }
          } else {
            webgpuReady = false;
            webgpuAdapterLabel = "";
            try {
              worker.terminate();
            } catch (_) {
              // Ignore teardown of inactive WebGPU sub-worker.
            }
            webgpuWorker = null;
          }
          done();
          return;
        }
        if (msg.type === "lost") {
          webgpuReady = false;
          webgpuAdapterLabel = "";
        }
      };

      worker.onerror = () => {
        webgpuReady = false;
        webgpuAdapterLabel = "";
        webgpuWorker = null;
        done();
      };

      worker.postMessage({ type: "init", sab: initialSab, eagerCompile: true });
    });
  } catch (_) {
    webgpuReady = false;
    webgpuAdapterLabel = "";
    webgpuWorker = null;
  }
}

async function mountWorkspaceFs(pyodide, allowOpfs) {
  opfsMounted = false;
  opfsHandle = null;
  pyodide.FS.mkdirTree("/workspace");
  if (
    !allowOpfs ||
    typeof navigator === "undefined" ||
    !navigator.storage ||
    typeof navigator.storage.getDirectory !== "function" ||
    typeof pyodide.mountNativeFS !== "function"
  ) {
    return;
  }
  try {
    const rootDir = await navigator.storage.getDirectory();
    const workspaceDir = await rootDir.getDirectoryHandle("workspace", { create: true });
    opfsHandle = await pyodide.mountNativeFS("/workspace", workspaceDir);
    opfsMounted = true;
  } catch (_) {
    // Graceful fallback to Emscripten MEMFS /workspace on restricted origins/browsers.
    pyodide.FS.mkdirTree("/workspace");
    opfsHandle = null;
    opfsMounted = false;
  }
}

function bytesToHex(buffer) {
  const bytes = new Uint8Array(buffer);
  const parts = new Array(bytes.length);
  for (let i = 0; i < bytes.length; i += 1) {
    parts[i] = bytes[i].toString(16).padStart(2, "0");
  }
  return parts.join("");
}

async function fetchJsonStrict(url, opts) {
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    throw new Error("HTTP " + resp.status + " fetching " + url);
  }
  return await resp.json();
}

async function fetchVerifiedBytes(url, expectedSize, expectedSha256) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error("HTTP " + resp.status + " fetching " + url);
  }
  const buf = await resp.arrayBuffer();
  if (buf.byteLength !== expectedSize) {
    throw new Error(
      "Size mismatch for " + url + ": got " + buf.byteLength + ", expected " + expectedSize
    );
  }
  const digest = await crypto.subtle.digest("SHA-256", buf);
  const actualHex = bytesToHex(digest);
  if (actualHex !== expectedSha256.toLowerCase()) {
    throw new Error(
      "SHA-256 mismatch for " + url + ": got " + actualHex + ", expected " + expectedSha256
    );
  }
  return buf;
}

const BOOTSTRAP_PY = `
import ast
import code
import io
import json
import math
import re
import rlcompleter
import sys
import time
import traceback
import types

_PKG_DIR = "/home/pyodide/baux_pkg"
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import numpy as np
import baux
from baux.benchmarks import synthetic


def _make_ns():
    return {
        "__name__": "__main__",
        "np": np,
        "numpy": np,
        "baux": baux,
        "synthetic": synthetic,
    }


_REPL_NS = _make_ns()
_CONSOLE = code.InteractiveConsole(locals=_REPL_NS)


def _reset_namespace():
    global _REPL_NS, _CONSOLE
    _REPL_NS = _make_ns()
    _CONSOLE = code.InteractiveConsole(locals=_REPL_NS)


def _json_scalar(val):
    if isinstance(val, (bool, int)):
        return int(val) if not isinstance(val, bool) else val
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _clean_float_list(seq, limit=128):
    out = []
    try:
        arr = np.asarray(seq, dtype=float).ravel().tolist()
    except Exception:
        return out
    for v in arr[:limit]:
        fv = _json_scalar(v)
        if fv is not None:
            out.append(round(fv, 6))
    return out


def _clean_pairs(seq, limit=128):
    out = []
    try:
        arr = np.asarray(seq, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return out
        for row in arr[:limit]:
            rx = _json_scalar(row[0])
            ry = _json_scalar(row[1])
            if rx is not None and ry is not None:
                out.append([round(rx, 5), round(ry, 5)])
    except Exception:
        return out
    return out


def _sanitize_curves(raw):
    if not isinstance(raw, dict):
        return None
    raw_series = raw.get("series")
    if not isinstance(raw_series, (list, tuple)) or not raw_series:
        return None
    series_out = []
    for item in raw_series[:6]:
        if not isinstance(item, dict):
            continue
        ys = _clean_float_list(item.get("y", []), limit=160)
        if not ys:
            continue
        xs = _clean_float_list(item.get("x", []), limit=160)
        if len(xs) != len(ys):
            xs = list(range(1, len(ys) + 1))
        series_out.append({
            "label": str(item.get("label", f"series {len(series_out) + 1}"))[:40],
            "x": xs,
            "y": ys,
            "dashed": bool(item.get("dashed", False)),
        })
    if not series_out:
        return None
    return {
        "kind": "curves",
        "title": str(raw.get("title", "Comparison"))[:80],
        "subtitle": str(raw.get("subtitle", ""))[:120],
        "x_label": str(raw.get("x_label", "Evaluations"))[:40],
        "y_label": str(raw.get("y_label", "Objective"))[:40],
        "log_y": bool(raw.get("log_y", False)),
        "series": series_out,
    }


def _extract_fig(raw):
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind", ""))
    if kind == "curves":
        return _sanitize_curves(raw)

    if kind == "tokamak":
        fg = raw.get("flux_grid") or {}
        nr = int(fg.get("nr", 0)) if isinstance(fg, dict) else 0
        nz = int(fg.get("nz", 0)) if isinstance(fg, dict) else 0
        if nr <= 1 or nz <= 1 or nr * nz > 2500:
            return None
        coils_out = []
        for c in (raw.get("coils") or [])[:40]:
            try:
                coils_out.append([round(float(x), 4) for x in c[:6]])
            except Exception:
                pass
        curves_sub = _sanitize_curves(raw.get("curves")) if isinstance(raw.get("curves"), dict) else None
        out = {
            "kind": "tokamak",
            "title": str(raw.get("title", "Tokamak Equilibrium"))[:80],
            "subtitle": str(raw.get("subtitle", ""))[:120],
            "r_range": _clean_float_list(raw.get("r_range", [0.4, 3.0]), limit=2),
            "z_range": _clean_float_list(raw.get("z_range", [-1.6, 1.6]), limit=2),
            "first_wall": _clean_pairs(raw.get("first_wall", []), limit=128),
            "lcfs": _clean_pairs(raw.get("lcfs", []), limit=128),
            "x_point": _clean_float_list(raw.get("x_point", []), limit=2),
            "coils": coils_out,
            "flux_grid": {
                "nr": nr,
                "nz": nz,
                "psi_init": _clean_float_list(fg.get("psi_init", []), limit=nr * nz),
                "psi_opt": _clean_float_list(fg.get("psi_opt", []), limit=nr * nz),
                "psi_init_target": _json_scalar(fg.get("psi_init_target")),
                "psi_opt_target": _json_scalar(fg.get("psi_opt_target")),
            },
        }
        if curves_sub is not None:
            out["curves"] = curves_sub
        return out

    if kind == "field2d":
        panels_out = []
        for p in (raw.get("panels") or [])[:2]:
            if not isinstance(p, dict):
                continue
            nx = int(p.get("nx", 0))
            nz = int(p.get("nz", 0))
            vals = _clean_float_list(p.get("vals", []), limit=1024)
            if nx >= 2 and nz >= 2 and len(vals) == nx * nz:
                panels_out.append({
                    "label": str(p.get("label", "field"))[:40],
                    "nx": nx,
                    "nz": nz,
                    "vals": vals,
                    "vmin": _json_scalar(p.get("vmin")),
                    "vmax": _json_scalar(p.get("vmax")),
                })
        if not panels_out:
            return None
        out = {
            "kind": "field2d",
            "title": str(raw.get("title", "2D Field"))[:80],
            "subtitle": str(raw.get("subtitle", ""))[:120],
            "panels": panels_out,
        }
        curves_sub = _sanitize_curves(raw.get("curves")) if isinstance(raw.get("curves"), dict) else None
        if curves_sub is not None:
            out["curves"] = curves_sub
        return out

    if kind == "band1d":
        xg = _clean_float_list(raw.get("x_grid", []), limit=160)
        mu = _clean_float_list(raw.get("mu", []), limit=160)
        if not xg or len(xg) != len(mu):
            return None
        out = {
            "kind": "band1d",
            "title": str(raw.get("title", "Surrogate Fit & Uncertainty"))[:80],
            "subtitle": str(raw.get("subtitle", ""))[:120],
            "x_grid": xg,
            "y_true": _clean_float_list(raw.get("y_true", []), limit=160),
            "mu": mu,
            "lo_2se": _clean_float_list(raw.get("lo_2se", []), limit=160),
            "hi_2se": _clean_float_list(raw.get("hi_2se", []), limit=160),
            "mw_lo": _clean_float_list(raw.get("mw_lo", []), limit=160),
            "mw_hi": _clean_float_list(raw.get("mw_hi", []), limit=160),
            "train_x": _clean_float_list(raw.get("train_x", []), limit=128),
            "train_y": _clean_float_list(raw.get("train_y", []), limit=128),
        }
        curves_sub = _sanitize_curves(raw.get("curves")) if isinstance(raw.get("curves"), dict) else None
        if curves_sub is not None:
            out["curves"] = curves_sub
        return out

    return None


def _summarize_state(ns):
    opt = None
    opt_name = None
    all_opts = []
    seen_ids = set()
    OptimizerState = getattr(baux, "OptimizerState", None)
    if OptimizerState is not None:
        for k, v in ns.items():
            if k.startswith("_") or not isinstance(v, OptimizerState):
                continue
            if int(getattr(v, "num_observations", 0)) <= 0 and k != "state":
                continue
            oid = id(v)
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            all_opts.append((k, v))
        if isinstance(ns.get("state"), OptimizerState):
            opt = ns["state"]
            opt_name = "state"
        elif all_opts:
            opt_name, opt = all_opts[-1]
        else:
            for k in reversed(list(ns.keys())):
                if not k.startswith("_") and isinstance(ns[k], OptimizerState):
                    opt = ns[k]
                    opt_name = k
                    break

    fig_out = _extract_fig(ns.get("fig"))
    if opt is None and fig_out is None:
        return None

    summary = {"fig": fig_out}
    if opt is not None:
        y_raw = getattr(opt, "y_obs", None)
        x_raw = getattr(opt, "x_obs", None)
        y_list = []
        best_trace = []
        recent_rows = []
        if y_raw is not None:
            ya = np.asarray(y_raw, dtype=float)
            if ya.ndim == 1:
                ya = ya[:, None]
            if ya.shape[0] > 0:
                prim = ya[:, 0] if ya.shape[1] == 1 else np.sum(ya, axis=1)
                running = -math.inf
                for v in prim.tolist():
                    fv = float(v)
                    if fv > running:
                        running = fv
                    y_list.append(fv if math.isfinite(fv) else None)
                    best_trace.append(running if math.isfinite(running) else None)
                if x_raw is not None:
                    xa = np.asarray(x_raw, dtype=float)
                    start = max(0, xa.shape[0] - 8)
                    for idx in range(start, xa.shape[0]):
                        row_x = [round(float(z), 4) for z in xa[idx, :6].tolist()]
                        recent_rows.append({
                            "obs": idx + 1,
                            "y": round(float(prim[idx]), 6),
                            "x": row_x,
                            "best_so_far": round(float(best_trace[idx]), 6) if best_trace[idx] is not None else None,
                        })

        best_x_val = getattr(opt, "best_x", None)
        best_x_list = None
        if best_x_val is not None:
            arr = np.asarray(best_x_val, dtype=float).ravel()
            best_x_list = [round(float(z), 5) for z in arr[:8].tolist()]

        best_params_repr = None
        bp = getattr(opt, "best_params", None)
        if bp is not None:
            try:
                best_params_repr = repr(bp)
                if len(best_params_repr) > 2400:
                    best_params_repr = best_params_repr[:2397] + "..."
            except Exception:
                best_params_repr = None

        summary.update({
            "var": opt_name,
            "dim": int(opt.dim),
            "epoch": int(opt.epoch),
            "num_observations": int(opt.num_observations),
            "num_regions": int(opt.num_regions),
            "tr_length": _json_scalar(opt.tr_length),
            "best_y": _json_scalar(opt.best_y),
            "best_x": best_x_list,
            "best_params": best_params_repr,
            "pending_count": 0 if opt.pending_ids is None else len(opt.pending_ids),
            "y_history": y_list[-128:],
            "best_history": best_trace[-128:],
            "recent": recent_rows,
        })

    if fig_out is None and len(all_opts) >= 2:
        auto_series = []
        for name, item in all_opts[:5]:
            ya = np.asarray(getattr(item, "y_obs", None), dtype=float)
            if ya.ndim == 1:
                ya = ya[:, None]
            if ya.shape[0] <= 0:
                continue
            prim = ya[:, 0] if ya.shape[1] == 1 else np.sum(ya, axis=1)
            running = -math.inf
            bt = []
            for v in prim.tolist():
                fv = float(v)
                if fv > running:
                    running = fv
                if math.isfinite(running):
                    bt.append(round(running, 6))
            if bt:
                auto_series.append({
                    "label": str(name)[:40],
                    "x": list(range(1, len(bt) + 1))[-128:],
                    "y": bt[-128:],
                    "dashed": False,
                })
        if len(auto_series) >= 2:
            summary["fig"] = {
                "kind": "curves",
                "title": "Cumulative Best Objective Comparison",
                "subtitle": "Auto-detected OptimizerState trajectories in workspace",
                "x_label": "Evaluations",
                "y_label": "Best Objective",
                "log_y": False,
                "series": auto_series,
            }

    return summary


def _summarize_scope(ns, limit=28):
    items = []
    OptimizerState = getattr(baux, "OptimizerState", None)
    priority_order = {"state": 0, "array": 1, "coll": 2, "scalar": 3, "fn": 4, "obj": 5}
    for k, v in ns.items():
        if k.startswith("_") or isinstance(v, types.ModuleType):
            continue
        kind = "obj"
        typ = type(v).__name__
        summary = ""
        if OptimizerState is not None and isinstance(v, OptimizerState):
            kind = "state"
            dim = int(getattr(v, "dim", 0))
            obs = int(getattr(v, "num_observations", 0))
            by = _json_scalar(getattr(v, "best_y", None))
            typ = f"OptimizerState({dim}D)"
            summary = f"obs={obs}" + (f" · best={by:.4g}" if by is not None else "")
        elif hasattr(v, "shape") and hasattr(v, "dtype"):
            kind = "array"
            try:
                shp = tuple(int(s) for s in v.shape)
                typ = f"ndarray{shp}"
                if int(getattr(v, "size", 0)) > 0:
                    arr = np.asarray(v, dtype=float)
                    lo = _json_scalar(np.nanmin(arr))
                    hi = _json_scalar(np.nanmax(arr))
                    if lo is not None and hi is not None:
                        summary = f"{v.dtype} [{lo:.3g}, {hi:.3g}]"
                    else:
                        summary = str(v.dtype)
                else:
                    summary = f"{v.dtype} (empty)"
            except Exception:
                summary = str(getattr(v, "dtype", ""))
        elif isinstance(v, bool):
            kind = "scalar"
            typ = "bool"
            summary = repr(v)
        elif isinstance(v, (int, float)):
            kind = "scalar"
            typ = type(v).__name__
            summary = repr(v) if isinstance(v, int) else f"{v:.6g}" if math.isfinite(v) else repr(v)
        elif isinstance(v, str):
            kind = "scalar"
            typ = f"str[{len(v)}]"
            summary = repr(v if len(v) <= 44 else v[:41] + "...")
        elif isinstance(v, (dict, list, tuple, set)):
            kind = "coll"
            typ = f"{type(v).__name__}[{len(v)}]"
            if isinstance(v, dict):
                keys = [repr(str(kk)[:20]) for kk in list(v.keys())[:4]]
                summary = "{" + ", ".join(keys) + (", ..." if len(v) > 4 else "}")
            else:
                summary = f"{len(v)} items"
        elif callable(v):
            mod = getattr(v, "__module__", None)
            if mod not in (None, "__main__", "builtins"):
                continue
            kind = "fn"
            typ = "callable"
            summary = "def " + k + "(...)"
        else:
            try:
                summary = repr(v)[:60]
            except Exception:
                summary = typ
        items.append({
            "name": str(k),
            "kind": kind,
            "type": typ,
            "summary": summary,
        })
    items.sort(key=lambda it: (priority_order.get(it["kind"], 9), it["name"]))
    return items[:limit]


def _baux_exec_script(src, reset_ns=False):
    if reset_ns:
        _reset_namespace()
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = out_buf
    sys.stderr = err_buf
    t0 = time.perf_counter()
    ok = True
    try:
        tree = ast.parse(src, filename="<playground>", mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(body=tree.body.pop().value)
            if tree.body:
                exec(compile(tree, "<playground>", "exec"), _REPL_NS)
            val = eval(compile(last_expr, "<playground>", "eval"), _REPL_NS)
            if val is not None:
                print(repr(val))
        else:
            exec(compile(tree, "<playground>", "exec"), _REPL_NS)
    except BaseException:
        ok = False
        traceback.print_exc(file=err_buf)
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        sys.stdout = old_out
        sys.stderr = old_err

    err_txt = err_buf.getvalue()
    err_line = None
    if not ok and err_txt:
        m = re.findall(r'<playground>", line (\d+)', err_txt)
        if m:
            try:
                err_line = int(m[-1])
            except Exception:
                err_line = None

    summary = _summarize_state(_REPL_NS)
    scope = _summarize_scope(_REPL_NS)
    return json.dumps({
        "ok": ok,
        "stdout": out_buf.getvalue(),
        "stderr": err_txt,
        "error_line": err_line,
        "incomplete": False,
        "elapsed_ms": round(dt_ms, 2),
        "state": summary,
        "scope": scope,
    })


def _baux_repl_push(line):
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = out_buf
    sys.stderr = err_buf
    t0 = time.perf_counter()
    ok = True
    incomplete = False
    try:
        if chr(10) in line:
            for sub in line.split(chr(10)):
                incomplete = bool(_CONSOLE.push(sub))
            if incomplete:
                incomplete = bool(_CONSOLE.push(""))
        else:
            incomplete = bool(_CONSOLE.push(line))
    except BaseException:
        ok = False
        traceback.print_exc(file=err_buf)
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        sys.stdout = old_out
        sys.stderr = old_err

    err_txt = err_buf.getvalue()
    if err_txt and not incomplete:
        ok = False
    err_line = None
    if not ok and err_txt:
        m = re.findall(r'<playground>", line (\d+)', err_txt)
        if m:
            try:
                err_line = int(m[-1])
            except Exception:
                err_line = None
    summary = _summarize_state(_REPL_NS)
    scope = _summarize_scope(_REPL_NS)
    return json.dumps({
        "ok": ok,
        "stdout": out_buf.getvalue(),
        "stderr": err_txt,
        "error_line": err_line,
        "incomplete": incomplete,
        "elapsed_ms": round(dt_ms, 2),
        "state": summary,
        "scope": scope,
    })


def _baux_complete(prefix):
    comp = rlcompleter.Completer(_REPL_NS)
    items = []
    idx = 0
    while len(items) < 24:
        cand = comp.complete(prefix, idx)
        if cand is None:
            break
        items.append(cand)
        idx += 1
    return json.dumps(items)
`;

async function ensureRuntime() {
  if (runtimePromise !== null) {
    return runtimePromise;
  }
  runtimePromise = (async () => {
    self.postMessage({
      type: "status",
      stage: "downloading_manifest",
      step: 1,
      total_steps: 4,
      detail: "Fetching manifest...",
    });
    const manifest = await fetchJsonStrict("./repl_manifest.json", { cache: "no-store" });
    const zipSpec = manifest.baux_py_zip;
    const companion =
      manifest && typeof manifest.companion_files === "object" && manifest.companion_files !== null
        ? manifest.companion_files
        : {};

    self.postMessage({
      type: "status",
      stage: "downloading_wasm",
      step: 2,
      total_steps: 4,
      detail: "Loading Pyodide & WASM...",
    });
    let webGpuPromise = Promise.resolve();
    const bridgePromise = loadVerifiedScript("./webgpu_bridge.js", companion.webgpu_bridge_js || null).then(() => {
      if (typeof BauxWebGpu === "undefined" || !BauxWebGpu.SAB_RPC) {
        throw new Error("webgpu_bridge.js did not expose BauxWebGpu.SAB_RPC");
      }
      webGpuPromise = initWebGpuChannel(companion.webgpu_worker_js || null);
    });
    const envPromise = fetch("./baux_env.json", { credentials: "same-origin", cache: "no-cache" })
      .then((resp) => (resp.ok ? resp.arrayBuffer() : null))
      .catch(() => null);
    const [, , pyodide, zipBuf, envBuf] = await Promise.all([
      bridgePromise,
      loadVerifiedScript("./oxcaml_eval.js", companion.oxcaml_eval_js || null),
      loadPyodide({ indexURL: "./pyodide/" }),
      fetchVerifiedBytes(withCacheVer("./" + zipSpec.path, zipSpec), zipSpec.size, zipSpec.sha256),
      envPromise,
    ]);
    if (!globalThis.BauxOxcaml || typeof globalThis.BauxOxcaml.evalOne !== "function") {
      throw new Error("oxcaml_eval.js did not expose globalThis.BauxOxcaml");
    }
    globalThis._baux_oxcaml = globalThis.BauxOxcaml;

    applyInterruptBuffer(pyodide);
    // Register all bauxgpu_* C symbols on Emscripten wasmImports BEFORE
    // instantiating baux_rust.cpython-312-wasm32-emscripten.so.
    registerWebGpuSymbols(pyodide);

    self.postMessage({
      type: "status",
      stage: "loading_numpy",
      step: 3,
      total_steps: 4,
      detail: "Loading NumPy & workspace...",
    });

    pyodide.FS.mkdirTree("/home/pyodide/baux_pkg");
    pyodide.unpackArchive(zipBuf, "zip", { extractDir: "/home/pyodide/baux_pkg" });

    if (envBuf) {
      pyodide.FS.writeFile("/home/pyodide/baux_env.json", new Uint8Array(envBuf));
    }

    const soPath = "/home/pyodide/baux_pkg/baux/baux_rust.cpython-312-wasm32-emscripten.so";
    if (!pyodide.FS.analyzePath(soPath).exists) {
      throw new Error("Missing compiled baux_rust.cpython-312-wasm32-emscripten.so in baux_py.zip");
    }

    await Promise.all([
      pyodide.loadPackage(["numpy"]),
      mountWorkspaceFs(pyodide, wantOpfsMount),
      webGpuPromise,
    ]);

    self.postMessage({
      type: "status",
      stage: "mounting_baux",
      step: 4,
      total_steps: 4,
      detail: "Mounting baux runtime...",
    });

    if (pyodide._api && typeof pyodide._api.loadDynlib === "function") {
      await pyodide._api.loadDynlib(soPath, false);
    } else if (typeof pyodide.loadDynamicLibrary === "function") {
      await pyodide.loadDynamicLibrary(soPath, {
        loadAsync: true,
        nodelete: true,
        global: false,
        fs: pyodide.FS,
      });
    } else if (pyodide._module && typeof pyodide._module.loadDynamicLibrary === "function") {
      await pyodide._module.loadDynamicLibrary(soPath, {
        loadAsync: true,
        nodelete: true,
        allowUndefined: true,
        global: false,
        fs: pyodide.FS,
      });
    }

    await pyodide.runPythonAsync(BOOTSTRAP_PY);

    self.postMessage({
      type: "status",
      stage: "ready",
      step: 4,
      total_steps: 4,
      detail: "Ready",
      caps: buildCaps(),
      opfs_mounted: opfsMounted,
    });
    return pyodide;
  })().catch((err) => {
    self.postMessage({
      type: "status",
      stage: "error",
      detail: err && err.message ? String(err.message) : String(err),
    });
    runtimePromise = null;
    throw err;
  });
  return runtimePromise;
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  const id = msg.id !== undefined ? msg.id : null;
  try {
    if (msg.type === "interrupt") {
      if (
        msg.interruptBuffer &&
        typeof SharedArrayBuffer === "function" &&
        msg.interruptBuffer instanceof SharedArrayBuffer
      ) {
        pendingInterruptBuffer = msg.interruptBuffer;
        interruptView = new Int32Array(pendingInterruptBuffer);
      }
      if (interruptView && typeof Atomics === "object") {
        Atomics.store(interruptView, 0, 2);
      }
      self.postMessage({
        type: "status",
        stage: "interrupting",
        detail: "Stopping...",
        caps: buildCaps(),
      });
      if (id !== null) {
        self.postMessage({
          type: "result",
          id,
          ok: true,
          stdout: "",
          stderr: "",
          incomplete: false,
          caps: buildCaps(),
          opfs_mounted: opfsMounted,
        });
      }
      return;
    }

    if (msg.type === "init") {
      if (msg.mountOpfs !== undefined) {
        wantOpfsMount = Boolean(msg.mountOpfs);
      }
      if (
        msg.interruptBuffer &&
        typeof SharedArrayBuffer === "function" &&
        msg.interruptBuffer instanceof SharedArrayBuffer
      ) {
        pendingInterruptBuffer = msg.interruptBuffer;
        interruptView = new Int32Array(pendingInterruptBuffer);
      }
      const pyodide = await ensureRuntime();
      applyInterruptBuffer(pyodide);
      self.postMessage({
        type: "result",
        id,
        ok: true,
        stdout: "",
        stderr: "",
        incomplete: false,
        caps: buildCaps(),
        opfs_mounted: opfsMounted,
      });
      return;
    }

    const pyodide = await ensureRuntime();

    if (msg.type === "reset_ns") {
      pyodide.globals.get("_reset_namespace")();
      await flushOpfs();
      self.postMessage({
        type: "result",
        id,
        ok: true,
        stdout: "",
        stderr: "",
        incomplete: false,
        state: null,
        scope: [],
        caps: buildCaps(),
        opfs_mounted: opfsMounted,
      });
      return;
    }

    if (msg.type === "exec") {
      clearStaleInterruptSignal();
      let raw;
      try {
        const fn = pyodide.globals.get("_baux_exec_script");
        raw = fn(String(msg.code || ""), Boolean(msg.resetNamespace));
      } finally {
        clearStaleInterruptSignal();
      }
      await flushOpfs();
      const payload = JSON.parse(raw);
      self.postMessage(
        Object.assign(
          { type: "result", id, caps: buildCaps(), opfs_mounted: opfsMounted },
          payload
        )
      );
      return;
    }

    if (msg.type === "repl") {
      clearStaleInterruptSignal();
      let raw;
      try {
        const fn = pyodide.globals.get("_baux_repl_push");
        raw = fn(String(msg.line || ""));
      } finally {
        clearStaleInterruptSignal();
      }
      await flushOpfs();
      const payload = JSON.parse(raw);
      self.postMessage(
        Object.assign(
          { type: "result", id, caps: buildCaps(), opfs_mounted: opfsMounted },
          payload
        )
      );
      return;
    }

    if (msg.type === "complete") {
      const fn = pyodide.globals.get("_baux_complete");
      const raw = fn(String(msg.prefix || ""));
      self.postMessage({ type: "complete", id, items: JSON.parse(raw) });
      return;
    }

    self.postMessage({
      type: "result",
      id,
      ok: false,
      stdout: "",
      stderr: "Unknown worker command",
      incomplete: false,
      caps: buildCaps(),
      opfs_mounted: opfsMounted,
    });
  } catch (err) {
    self.postMessage({
      type: "result",
      id,
      ok: false,
      stdout: "",
      stderr: err && err.stack ? String(err.stack) : String(err),
      incomplete: false,
      caps: buildCaps(),
      opfs_mounted: opfsMounted,
    });
  }
};
