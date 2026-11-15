/* Dedicated Worker hosting async WebGPU compute + SharedArrayBuffer Atomics RPC */
"use strict";

if (typeof BauxWebGpu === "undefined" && typeof importScripts === "function") {
  importScripts("./webgpu_bridge.js");
}

(function initWebGpuWorker() {
  if (typeof BauxWebGpu === "undefined" || !BauxWebGpu.SAB_RPC) {
    throw new Error("webgpu_worker: BauxWebGpu.SAB_RPC unavailable");
  }

  const rpc = BauxWebGpu.SAB_RPC;
  let bridge = null;
  let webgpuAvailable = false;
  let adapterInfo = null;
  let adapterLabel = "";
  let sab = null;
  let ctrlI32 = null;
  let ctrlU32 = null;
  let ctrlF32 = null;
  let lastCompletedSeq = 0;
  let activeRunPromise = null;

  function adoptSab(nextSab) {
    if (
      typeof SharedArrayBuffer !== "function" ||
      !(nextSab instanceof SharedArrayBuffer) ||
      nextSab.byteLength < rpc.HEADER_BYTES
    ) {
      throw new Error("webgpu_worker: invalid SharedArrayBuffer");
    }
    if (sab === nextSab) {
      return;
    }
    sab = nextSab;
    ctrlI32 = new Int32Array(sab, 0, rpc.HEADER_I32_LEN);
    ctrlU32 = new Uint32Array(sab, 0, rpc.HEADER_I32_LEN);
    ctrlF32 = new Float32Array(sab, 0, rpc.HEADER_I32_LEN);
  }

  function checkSliceBounds(byteOffset, elemCount) {
    const off = byteOffset >>> 0;
    const len = elemCount >>> 0;
    if ((off & 3) !== 0 || off < rpc.HEADER_BYTES) {
      throw new RangeError("webgpu_worker: invalid SAB slice offset " + off);
    }
    const byteLen = len * 4;
    if (!Number.isSafeInteger(byteLen) || byteLen <= 0 || off + byteLen > sab.byteLength) {
      throw new RangeError(
        "webgpu_worker: SAB slice out of bounds (" + off + " + " + byteLen + " > " + sab.byteLength + ")"
      );
    }
    return { off, len };
  }

  function readSabF32(offSlot, lenSlot) {
    const { off, len } = checkSliceBounds(ctrlU32[offSlot], ctrlU32[lenSlot]);
    // .slice() detaches into an unshared ArrayBuffer so GPUQueue.writeBuffer
    // never depends on [AllowShared] TypedArray support across browsers.
    return new Float32Array(sab, off, len).slice();
  }

  function readSabU32(offSlot, lenSlot) {
    const { off, len } = checkSliceBounds(ctrlU32[offSlot], ctrlU32[lenSlot]);
    return new Uint32Array(sab, off, len).slice();
  }

  function writeSabF32(offSlot, lenSlot, data) {
    const { off, len } = checkSliceBounds(ctrlU32[offSlot], ctrlU32[lenSlot]);
    if (data.length !== len) {
      throw new RangeError("webgpu_worker: f32 output length mismatch (" + data.length + " != " + len + ")");
    }
    new Float32Array(sab, off, len).set(data);
  }

  function writeSabU32(offSlot, lenSlot, data) {
    const { off, len } = checkSliceBounds(ctrlU32[offSlot], ctrlU32[lenSlot]);
    if (data.length !== len) {
      throw new RangeError("webgpu_worker: u32 output length mismatch (" + data.length + " != " + len + ")");
    }
    new Uint32Array(sab, off, len).set(data);
  }

  async function executeOp(op) {
    switch (op) {
      case rpc.OP_BUFFER_ALLOC: {
        const byteSize = ctrlU32[rpc.IDX_P0];
        const handle = bridge.allocBuffer(byteSize, "wasm_c_buf") >>> 0;
        ctrlU32[rpc.IDX_RESULT_U32] = handle;
        return handle > 0 ? rpc.ERR_OK : rpc.ERR_GPU_EXEC;
      }

      case rpc.OP_BUFFER_FREE: {
        const handle = ctrlU32[rpc.IDX_P0];
        const freed = bridge.freeBuffer(handle);
        ctrlU32[rpc.IDX_RESULT_U32] = 0;
        return freed ? rpc.ERR_OK : rpc.ERR_INVALID_ARGS;
      }

      case rpc.OP_BUFFER_READ_F32: {
        const handle = ctrlU32[rpc.IDX_P0];
        const len = ctrlU32[rpc.IDX_LEN0];
        checkSliceBounds(ctrlU32[rpc.IDX_OFF0], len);
        const out = await bridge.readBufferF32(handle, len);
        writeSabF32(rpc.IDX_OFF0, rpc.IDX_LEN0, out);
        ctrlU32[rpc.IDX_RESULT_U32] = 0;
        return rpc.ERR_OK;
      }

      case rpc.OP_EXACT_DIST: {
        const q = ctrlU32[rpc.IDX_P0];
        const n = ctrlU32[rpc.IDX_P1];
        const d = ctrlU32[rpc.IDX_P2];
        const hasScale = ctrlU32[rpc.IDX_P3] !== 0;
        checkSliceBounds(ctrlU32[rpc.IDX_OFF3], ctrlU32[rpc.IDX_LEN3]);
        const queries = readSabF32(rpc.IDX_OFF0, rpc.IDX_LEN0);
        const rows = readSabF32(rpc.IDX_OFF1, rpc.IDX_LEN1);
        const invScale2 = hasScale ? readSabF32(rpc.IDX_OFF2, rpc.IDX_LEN2) : null;
        const outDist2 = await bridge.runExactDist(queries, rows, q, n, d, invScale2);
        writeSabF32(rpc.IDX_OFF3, rpc.IDX_LEN3, outDist2);
        return rpc.ERR_OK;
      }

      case rpc.OP_DENSE_DIST2: {
        const q = ctrlU32[rpc.IDX_P0];
        const n = ctrlU32[rpc.IDX_P1];
        const d = ctrlU32[rpc.IDX_P2];
        checkSliceBounds(ctrlU32[rpc.IDX_OFF2], ctrlU32[rpc.IDX_LEN2]);
        const a = readSabF32(rpc.IDX_OFF0, rpc.IDX_LEN0);
        const b = readSabF32(rpc.IDX_OFF1, rpc.IDX_LEN1);
        const outDist2 = await bridge.runDenseDist2(a, b, q, n, d);
        writeSabF32(rpc.IDX_OFF2, rpc.IDX_LEN2, outDist2);
        return rpc.ERR_OK;
      }

      case rpc.OP_KNN_TOPK: {
        const q = ctrlU32[rpc.IDX_P0];
        const n = ctrlU32[rpc.IDX_P1];
        const k = ctrlU32[rpc.IDX_P2];
        const skip = ctrlU32[rpc.IDX_P3];
        checkSliceBounds(ctrlU32[rpc.IDX_OFF1], ctrlU32[rpc.IDX_LEN1]);
        checkSliceBounds(ctrlU32[rpc.IDX_OFF2], ctrlU32[rpc.IDX_LEN2]);
        const dist2 = readSabF32(rpc.IDX_OFF0, rpc.IDX_LEN0);
        const { dist, idx } = await bridge.runKnnTopk(dist2, q, n, k, skip);
        writeSabF32(rpc.IDX_OFF1, rpc.IDX_LEN1, dist);
        writeSabU32(rpc.IDX_OFF2, rpc.IDX_LEN2, idx);
        return rpc.ERR_OK;
      }

      case rpc.OP_DEVICE_POSTERIOR: {
        const queries = ctrlU32[rpc.IDX_P0];
        const n = ctrlU32[rpc.IDX_P1];
        const metrics = ctrlU32[rpc.IDX_P2];
        const inputK = ctrlU32[rpc.IDX_P3];
        const k = ctrlU32[rpc.IDX_P4];
        const skip = ctrlU32[rpc.IDX_P5];
        const observationNoise = ctrlU32[rpc.IDX_P6] !== 0;
        const hasVariances = ctrlU32[rpc.IDX_P7] !== 0;
        const epistemicScale = ctrlF32[rpc.IDX_F0];
        const aleatoricScale = ctrlF32[rpc.IDX_F1];
        const epsilon = ctrlF32[rpc.IDX_F2];

        checkSliceBounds(ctrlU32[rpc.IDX_OFF5], ctrlU32[rpc.IDX_LEN5]);
        checkSliceBounds(ctrlU32[rpc.IDX_OFF6], ctrlU32[rpc.IDX_LEN6]);
        checkSliceBounds(ctrlU32[rpc.IDX_OFF7], ctrlU32[rpc.IDX_LEN7]);

        const distances = readSabF32(rpc.IDX_OFF0, rpc.IDX_LEN0);
        const ids = readSabU32(rpc.IDX_OFF1, rpc.IDX_LEN1);
        const outcomes = readSabF32(rpc.IDX_OFF2, rpc.IDX_LEN2);
        const scales = readSabF32(rpc.IDX_OFF3, rpc.IDX_LEN3);
        const variances = hasVariances ? readSabF32(rpc.IDX_OFF4, rpc.IDX_LEN4) : null;

        const res = await bridge.runDevicePosterior({
          queries,
          n,
          metrics,
          inputK,
          k,
          skip,
          epistemicScale,
          aleatoricScale,
          epsilon,
          observationNoise,
          distances,
          ids,
          outcomes,
          variances,
          scales,
        });
        writeSabF32(rpc.IDX_OFF5, rpc.IDX_LEN5, res.summary);
        writeSabU32(rpc.IDX_OFF6, rpc.IDX_LEN6, res.selected);
        writeSabF32(rpc.IDX_OFF7, rpc.IDX_LEN7, res.weights);
        return rpc.ERR_OK;
      }

      case rpc.OP_DENSE_APPLY: {
        const m = ctrlU32[rpc.IDX_P0];
        const n = ctrlU32[rpc.IDX_P1];
        const k = ctrlU32[rpc.IDX_P2];
        const hasBias = ctrlU32[rpc.IDX_P3] !== 0;
        checkSliceBounds(ctrlU32[rpc.IDX_OFF3], ctrlU32[rpc.IDX_LEN3]);
        const x = readSabF32(rpc.IDX_OFF0, rpc.IDX_LEN0);
        const w = readSabF32(rpc.IDX_OFF1, rpc.IDX_LEN1);
        const bias = hasBias ? readSabF32(rpc.IDX_OFF2, rpc.IDX_LEN2) : null;
        const y = await bridge.runDenseApply(x, w, m, n, k, bias);
        writeSabF32(rpc.IDX_OFF3, rpc.IDX_LEN3, y);
        return rpc.ERR_OK;
      }

      case rpc.OP_WEIGHT_SELECT: {
        const batch = ctrlU32[rpc.IDX_P0];
        const candidates = ctrlU32[rpc.IDX_P1];
        const mode = ctrlU32[rpc.IDX_P2];
        const emitSoftmax = ctrlU32[rpc.IDX_P3] !== 0;
        const hasZ = ctrlU32[rpc.IDX_P4] !== 0;
        const beta = ctrlF32[rpc.IDX_F0];
        const temperature = ctrlF32[rpc.IDX_F1];
        const minProb = ctrlF32[rpc.IDX_F2];

        checkSliceBounds(ctrlU32[rpc.IDX_OFF3], ctrlU32[rpc.IDX_LEN3]);
        checkSliceBounds(ctrlU32[rpc.IDX_OFF4], ctrlU32[rpc.IDX_LEN4]);
        checkSliceBounds(ctrlU32[rpc.IDX_OFF5], ctrlU32[rpc.IDX_LEN5]);
        checkSliceBounds(ctrlU32[rpc.IDX_OFF6], ctrlU32[rpc.IDX_LEN6]);

        const mu = readSabF32(rpc.IDX_OFF0, rpc.IDX_LEN0);
        const se = readSabF32(rpc.IDX_OFF1, rpc.IDX_LEN1);
        const z = hasZ ? readSabF32(rpc.IDX_OFF2, rpc.IDX_LEN2) : null;

        const res = await bridge.runWeightSelect({
          batch,
          candidates,
          mode,
          emitSoftmax,
          beta,
          temperature,
          minProb,
          mu,
          se,
          z,
        });
        writeSabF32(rpc.IDX_OFF3, rpc.IDX_LEN3, res.scores);
        writeSabF32(rpc.IDX_OFF4, rpc.IDX_LEN4, res.probs);
        writeSabU32(rpc.IDX_OFF5, rpc.IDX_LEN5, res.bestIdx);
        writeSabF32(rpc.IDX_OFF6, rpc.IDX_LEN6, res.bestScore);
        return rpc.ERR_OK;
      }

      default:
        return rpc.ERR_INVALID_ARGS;
    }
  }

  async function handleOneRequest(seq) {
    let rc = rpc.ERR_GPU_EXEC;
    let finalState = rpc.STATE_ERROR;
    try {
      if (!webgpuAvailable || !bridge || bridge.lostReason !== null) {
        ctrlU32[rpc.IDX_RESULT_U32] = 0;
        rc = rpc.ERR_UNAVAILABLE;
      } else {
        const op = ctrlI32[rpc.IDX_OP];
        ctrlU32[rpc.IDX_RESULT_U32] = 0;
        rc = await executeOp(op);
        if (rc === rpc.ERR_OK) {
          finalState = rpc.STATE_DONE;
        }
      }
    } catch (err) {
      ctrlU32[rpc.IDX_RESULT_U32] = 0;
      rc = err instanceof RangeError ? rpc.ERR_INVALID_ARGS : rpc.ERR_GPU_EXEC;
    } finally {
      lastCompletedSeq = seq;
      Atomics.store(ctrlI32, rpc.IDX_RESULT_I32, rc | 0);
      Atomics.store(ctrlI32, rpc.IDX_STATE, finalState);
      Atomics.notify(ctrlI32, rpc.IDX_STATE, 1);
    }
  }

  async function drainPending() {
    while (ctrlI32 && Atomics.load(ctrlI32, rpc.IDX_STATE) === rpc.STATE_PENDING) {
      const seq = Atomics.load(ctrlU32, rpc.IDX_SEQ) >>> 0;
      if (seq === lastCompletedSeq) {
        break;
      }
      await handleOneRequest(seq);
    }
  }

  function scheduleProcess() {
    if (activeRunPromise !== null) {
      return;
    }
    activeRunPromise = drainPending().finally(() => {
      activeRunPromise = null;
      if (ctrlI32 && Atomics.load(ctrlI32, rpc.IDX_STATE) === rpc.STATE_PENDING) {
        scheduleProcess();
      }
    });
  }

  self.onmessage = async (event) => {
    const msg = event.data || {};
    if (msg.type === "dispatch") {
      try {
        if (msg.sab) {
          adoptSab(msg.sab);
        }
        scheduleProcess();
      } catch (err) {
        if (ctrlI32) {
          Atomics.store(ctrlI32, rpc.IDX_RESULT_I32, rpc.ERR_INVALID_ARGS);
          Atomics.store(ctrlI32, rpc.IDX_STATE, rpc.STATE_ERROR);
          Atomics.notify(ctrlI32, rpc.IDX_STATE, 1);
        }
      }
      return;
    }

    if (msg.type === "init") {
      try {
        if (msg.sab) {
          adoptSab(msg.sab);
        }
        const res = await BauxWebGpu.init({ eagerCompile: msg.eagerCompile !== false });
        bridge = res && res.bridge ? res.bridge : null;
        adapterInfo = res && res.adapterInfo ? res.adapterInfo : null;
        adapterLabel = BauxWebGpu.formatAdapterLabel(adapterInfo);
        webgpuAvailable = Boolean(res && res.available && bridge);
        if (webgpuAvailable && bridge.device && bridge.device.lost && typeof bridge.device.lost.then === "function") {
          bridge.device.lost.then((info) => {
            webgpuAvailable = false;
            self.postMessage({
              type: "lost",
              reason: String((info && info.message) || (info && info.reason) || "WebGPU device lost"),
            });
          });
        }
        self.postMessage({
          type: "ready",
          available: webgpuAvailable,
          adapterInfo,
          adapterLabel,
          reason: res && res.reason ? res.reason : null,
          limits: res && res.limits ? res.limits : null,
        });
        scheduleProcess();
      } catch (err) {
        webgpuAvailable = false;
        bridge = null;
        self.postMessage({
          type: "ready",
          available: false,
          adapterInfo: null,
          adapterLabel: "",
          reason: err && err.message ? String(err.message) : String(err),
          limits: null,
        });
      }
      return;
    }

    if (msg.type === "shutdown") {
      webgpuAvailable = false;
      if (bridge) {
        try {
          bridge.destroyAll();
        } catch (_) {
          // Ignore teardown errors during worker shutdown.
        }
        bridge = null;
      }
      if (typeof self.close === "function") {
        self.close();
      }
    }
  };
})();
