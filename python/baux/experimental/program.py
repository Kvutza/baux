"""Structured calls to verified StableHLO programs.

References
----------
OpenXLA, StableHLO specification: https://openxla.org/stablehlo/spec
OpTree, tree operations: https://optree.readthedocs.io/en/stable/api.html

Compilation specializes tensor shapes and dtypes. Transfers are explicit through
``put`` and ``get``; calls accept resident buffers and return resident buffers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import optree
import optree.dataclasses

from . import _program


@dataclass(frozen=True)
class TensorSpec:
    """Shape and dtype of one tensor leaf; scalar tensors have shape ``()``."""

    shape: tuple[int, ...]
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if any(
            not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in self.shape
        ):
            raise ValueError("tensor dimensions must be nonnegative integers")
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.dtype == "bfloat16":
            import ml_dtypes

            dtype = np.dtype(ml_dtypes.bfloat16)
        else:
            dtype = np.dtype(self.dtype)
        object.__setattr__(self, "dtype", str(dtype))


def register(cls: type) -> type:
    """Register a dataclass; ``pytree_node=False`` field metadata is static.

    Reconstruction invokes the dataclass constructor. Classes with transforming
    constructors need an explicit OpTree registration in ``baux.program``.
    """
    return optree.dataclasses.register_node(cls, namespace="baux.program")


def _flatten(tree: Any) -> tuple[list[Any], Any]:
    return optree.tree_flatten(tree, namespace="baux.program")


def _spec(value: Any) -> TensorSpec:
    if isinstance(value, TensorSpec):
        return value
    return TensorSpec(tuple(value.shape), str(value.dtype))


def _check(values: list[Any], specs: tuple[TensorSpec, ...]) -> None:
    if len(values) != len(specs):
        raise ValueError(f"expected {len(specs)} tensors, got {len(values)}")
    for index, (value, spec) in enumerate(zip(values, specs, strict=True)):
        if _spec(value) != spec:
            raise ValueError(f"tensor {index}: expected {spec}, got {_spec(value)}")


@dataclass(frozen=True)
class Buffer:
    """A backend-owned tensor retaining its execution context."""

    _owner: Any
    _value: Any

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor dimensions without a host transfer."""
        return tuple(self._value.shape)

    @property
    def dtype(self) -> str:
        """Element dtype without a host transfer."""
        return str(self._value.dtype)

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("resident buffers require executable.get() for host transfer")


class Executable:
    """A compiled structured function; state is passed and returned explicitly."""

    def __init__(self, native: Any, inputs: Any, outputs: Any, engine: Any) -> None:
        self.native = native
        ins, self._intree = _flatten(inputs)
        outs, self._outtree = _flatten(outputs)
        self._inputs = tuple(map(_spec, ins))
        self._outputs = tuple(map(_spec, outs))
        self._engine = engine

    def put(self, tree: Any) -> Any:
        """Upload host tensors, preserving tree structure and dtype."""
        leaves, structure = _flatten(tree)
        if structure != self._intree:
            raise ValueError("input tree does not match the compiled signature")
        _check(leaves, self._inputs)
        if any(not isinstance(value, np.ndarray) for value in leaves):
            raise TypeError(
                "put accepts NumPy tensors; device inputs must already be buffers"
            )
        return structure.unflatten(
            [
                Buffer(
                    self._engine, self._engine.put(np.array(x, copy=True, order="C"))
                )
                for x in leaves
            ]
        )

    def get(self, tree: Any) -> Any:
        """Wait for results and copy resident tensors to host arrays."""
        leaves, structure = _flatten(tree)
        self._owned(leaves)
        return structure.unflatten([self._engine.get(x._value) for x in leaves])

    def _owned(self, leaves: list[Any]) -> None:
        if any(
            not isinstance(x, Buffer) or x._owner is not self._engine for x in leaves
        ):
            raise ValueError("buffers must belong to this executable")

    def __call__(self, tree: Any) -> Any:
        """Execute on resident inputs and reconstruct the structured results."""
        return self._results(self._engine.run(self._arguments(tree)))

    def _arguments(self, tree: Any) -> list[Any]:
        leaves, structure = _flatten(tree)
        if structure != self._intree:
            raise ValueError("input tree does not match the compiled signature")
        self._owned(leaves)
        _check(leaves, self._inputs)
        return [x._value for x in leaves]

    def _results(self, results: list[Any]) -> Any:
        _check(results, self._outputs)
        return self._outtree.unflatten([Buffer(self._engine, x) for x in results])

    def submit(self, tree: Any) -> Invocation:
        """Queue backend work; no host readback or waiting in this call."""
        return Invocation(self, self._engine.submit(self._arguments(tree)))


class Invocation:
    """Single-consumer completion; dropping it does not cancel queued work."""

    def __init__(self, executable: Executable, pending: Any) -> None:
        self._executable = executable
        self._pending = pending

    def ready(self) -> bool:
        """Whether resident results are available, without transferring them."""
        return self._pending.ready()

    def wait(self) -> Any:
        """Consume completion, releasing the GIL while native work runs."""
        return self._executable._results(self._pending.wait())

    def __await__(self) -> Any:
        import asyncio

        return asyncio.to_thread(self.wait).__await__()


@dataclass(frozen=True)
class ModelOutput:
    """Resident per-token losses and per-sequence negative log likelihoods."""

    token_losses: Any
    sequence_nlls: Any


class ModelInvocation:
    """One queued FBT/PISA model forward pass."""

    def __init__(self, model: ResidentModel, pending: Any) -> None:
        self._model = model
        self._pending = pending

    def ready(self) -> bool:
        """Whether native model outputs are ready without reading them."""
        return self._pending.ready()

    def wait(self) -> ModelOutput:
        """Consume the completion and return resident model outputs."""
        return self._model._output(self._pending.wait())

    def __await__(self) -> Any:
        import asyncio

        return asyncio.to_thread(self.wait).__await__()


class ResidentModel:
    """Initialized fixed-shape FBT/PISA model with resident Metal parameters.

    Inputs are copied into owned native allocations at submission. Results stay
    resident until :meth:`get`; dropping an invocation does not cancel work.
    ``backend="tvm"`` invokes the production Metal kernels through native Relax
    with retained device buffers. ``backend="native"`` invokes them directly.
    """

    def __init__(
        self,
        *,
        model: str = "fbt-pisa1-mhc4-v1",
        initialization: str = "fbt-pisa1-mhc4",
        seed: int = 42,
        backend: Literal["native", "tvm"] = "native",
    ) -> None:
        from .. import baux_rust

        self._native = baux_rust.ResidentModel(
            model=model, initialization=initialization, seed=seed, backend=backend
        )

    @property
    def backend(self) -> str:
        """Execution runtime; both options retain the production Metal kernels."""
        return self._native.backend

    @property
    def parameters(self) -> int:
        """Number of resident FP16 model parameters."""
        return self._native.parameters

    @property
    def rows(self) -> int:
        """Required number of token, label, and mask elements per call."""
        return self._native.rows

    def _inputs(
        self, tokens: Any, labels: Any, mask: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = (
            np.asarray(tokens, dtype=np.uint32, order="C"),
            np.asarray(labels, dtype=np.uint32, order="C"),
            np.asarray(mask, dtype=np.uint8, order="C"),
        )
        names = ("tokens", "labels", "mask")
        for name, value in zip(names, values, strict=True):
            if value.ndim != 1 or value.size != self.rows:
                raise ValueError(f"{name} must have shape ({self.rows},)")
        return values

    def _output(self, values: list[Any]) -> ModelOutput:
        if len(values) != 2:
            raise RuntimeError("resident model returned an invalid output count")
        return ModelOutput(
            Buffer(self, values[0]),
            Buffer(self, values[1]),
        )

    def submit(self, tokens: Any, labels: Any, mask: Any) -> ModelInvocation:
        """Queue a forward pass and return without reading its outputs."""
        inputs = self._inputs(tokens, labels, mask)
        return ModelInvocation(self, self._native.submit(*inputs))

    def weights(self) -> np.ndarray:
        """Copy a consistent parameter snapshot from Metal to the host.

        Returns
        -------
        numpy.ndarray
            Independent, writable FP16 array of shape ``(parameters,)`` in
            canonical packed order. Mutating it does not update the model.

        Notes
        -----
        Blocks until readback completes and holds an immutable optimizer lease
        during the read. This explicit export is outside the resident hot path.
        """
        return np.frombuffer(self._native.weights(), dtype=np.float16).copy()

    def __call__(self, tokens: Any, labels: Any, mask: Any) -> ModelOutput:
        """Run one model forward pass and retain its outputs on device."""
        return self.submit(tokens, labels, mask).wait()

    def get(self, output: ModelOutput) -> ModelOutput:
        """Copy a model output to host NumPy arrays."""
        if not isinstance(output, ModelOutput):
            raise TypeError("get expects a ModelOutput")
        for value in (output.token_losses, output.sequence_nlls):
            if value._owner is not self:
                raise ValueError("model outputs must belong to this model")

        def read(value: Buffer) -> np.ndarray:
            return (
                np.frombuffer(value._value.read(), dtype=value.dtype)
                .reshape(value.shape)
                .copy()
            )

        return ModelOutput(read(output.token_losses), read(output.sequence_nlls))


def compile(
    source: str | Callable[[Any], Any],
    inputs: Any,
    outputs: Any = None,
    *,
    backend: Literal["iree", "tvm", "xla"] = "iree",
    device: str = "cpu",
    entry: str = "main",
) -> Executable:
    """Compile StableHLO with a checked structured tensor signature.

    Parameters
    ----------
    source : str or callable
        StableHLO text, or a JAX-traceable function taking one tensor tree and
        returning one tensor tree. Callable output specifications are inferred.
    inputs : tree of TensorSpec
        Static tensor shapes and dtypes in OpTree's deterministic leaf order.
    outputs : tree of TensorSpec, optional
        Required for source text; must be omitted for a callable.
    backend : {"iree", "tvm", "xla"}, optional
        Native compiler/runtime. Defaults to IREE. TVM uses TIR/Relax and LLVM;
        XLA uses the bundled PJRT CPU plugin. No backend fallback is performed.
    device : {"cpu", "metal"}, optional
        Execution device. Defaults to CPU. Metal is supported by IREE; the
        native StableHLO TVM and XLA compilation paths currently target CPU.
    entry : str, optional
        Exported function name. Defaults to ``main``; callables and XLA require
        that name.

    Returns
    -------
    Executable
        Owner of the compiled session and structured tensor signature. Upload
        with ``put``, execute synchronously by calling it or asynchronously with
        ``submit``, and explicitly copy results to the host with ``get``.

    Raises
    ------
    ValueError
        Invalid signature, source, entry, backend or device selection.
    RuntimeError
        Native compilation or runtime initialization fails.

    Notes
    -----
    Compilation is synchronous and does not upload model inputs. Buffers and
    pending calls retain the native session. A submitted call retains inputs
    until completion; dropping its completion object does not cancel work.
    Supported operations and floating-point results depend on the compiler.
    Successful compilation alone does not establish model numerical parity.

    Examples
    --------
    >>> import numpy as np
    >>> spec = TensorSpec((4,), "float32")
    >>> program = compile(lambda x: x + x, spec, backend="tvm")
    >>> values = program.put(np.arange(4, dtype=np.float32))
    >>> program.get(program(values)).tolist()
    [0.0, 2.0, 4.0, 6.0]

    See Also
    --------
    Fbt : Experimental compiler-visible model graph.
    ResidentModel : Native Metal model execution and parameter export.
    """
    from .. import baux_rust

    if callable(source):
        from ._trace import lower

        if outputs is not None or entry != "main":
            raise ValueError("callable programs infer outputs and use entry main")
        source, outputs = lower(source, inputs)
    ins, _ = _flatten(inputs)
    outs, _ = _flatten(outputs)
    in_specs, out_specs = tuple(map(_spec, ins)), tuple(map(_spec, outs))
    native = baux_rust.Program(
        source,
        [(x.shape, x.dtype) for x in in_specs],
        [(x.shape, x.dtype) for x in out_specs],
        entry=entry,
    )
    _program.verify(native)
    engine = _program.create(native, backend, device)
    return Executable(native, inputs, outputs, engine)
