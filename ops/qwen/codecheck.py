"""Fail-closed macOS execution of Qwen MBPP solution/setup/test strings.

The OS sandbox is the host-access boundary. The Python completion protocol is
NOT a security boundary for scores against deliberately adversarial Python
(which can inspect frames or alter the runner's interpreter state). A plain
exit, including os._exit(0), never counts as successful completion.

Requires sandbox-exec and Apple's installed dyld-support.sb. There is no Docker
or unsandboxed fallback. Use from an authorized host process: an outer sandbox
may prohibit installing the profile, in which case this returns unavailable.
Only the current interpreter, its stdlib (not site-packages), and system loader
paths are readable; optional third-party compiled-library imports may fail.
"""

from __future__ import annotations

import json
import os
import secrets
import selectors
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
SANDBOX_LIBRARY = "/usr/lib/libsandbox.1.dylib"
MAX_INPUT_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 16 * 1024
MAX_PROTOCOL_BYTES = 4096
MAX_TESTS = 256

# This trusted bootstrap runs in a separate isolated Python interpreter. Source
# is data in payload.json and is not read until the confinement probes pass.
_RUNNER = r"""
import builtins
import ctypes
import errno
import json
import math
import os
import resource
import socket
import sys

def main():
    fd, nonce, canary, timeout = sys.argv[1:]
    fd = int(fd)
    write, dumps = os.write, json.dumps
    def report(event):
        event["nonce"] = nonce
        write(fd, (dumps(event) + "\n").encode("ascii"))

    try:
        for kind, value in (
            (resource.RLIMIT_CORE, 0),
            (resource.RLIMIT_CPU, max(1, math.ceil(float(timeout)))),
            (resource.RLIMIT_FSIZE, 1024 * 1024),
            (resource.RLIMIT_NOFILE, 64),
            (resource.RLIMIT_NPROC, 0),
        ):
            hard = resource.getrlimit(kind)[1]
            if hard != resource.RLIM_INFINITY:
                value = min(value, hard)
            resource.setrlimit(kind, (value, value))

        # Probe only our own benign canary, never a host secret. O_WRONLY does
        # not truncate or modify it if a broken profile unexpectedly allows it.
        for flags in (os.O_RDONLY, os.O_WRONLY):
            try:
                probe = os.open(canary, flags)
            except OSError as exc:
                if exc.errno not in (errno.EPERM, errno.EACCES):
                    raise RuntimeError("filesystem probe inconclusive")
            else:
                os.close(probe)
                raise RuntimeError("filesystem confinement probe allowed")

        check = ctypes.CDLL("/usr/lib/libsandbox.1.dylib").sandbox_check
        check.restype = ctypes.c_int
        # No-filter checks must deny these operations in their entirety.
        for operation in (b"process-fork", b"network-outbound", b"network-inbound"):
            if check(os.getpid(), operation, 0) != 1:
                raise RuntimeError("sandbox operation probe failed")
        try:
            with socket.socket() as sock:
                sock.settimeout(0.2)
                sock.connect(("127.0.0.1", 9))
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.EACCES):
                raise RuntimeError("network probe inconclusive")
        else:
            raise RuntimeError("network confinement probe allowed")
        with open("work-probe", "w") as handle:
            handle.write("ok")
        os.unlink("work-probe")
    except BaseException as exc:
        report({"event": "unavailable", "detail": type(exc).__name__ + ": " + str(exc)[:512]})
        return

    report({"event": "ready"})
    with open("payload.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    # Keep command-line protocol details out of the ordinary solution namespace.
    sys.argv = ["<solution>"]
    namespace = {"__name__": "__main__", "__builtins__": dict(vars(builtins))}
    compile_source, execute = compile, exec
    syntax_error, assertion_error = SyntaxError, AssertionError
    base_exception, exception_type = BaseException, type
    phase, index, completed = "setup", None, 0
    try:
        execute(compile_source(payload["setup"], "<setup>", "exec"), namespace)
        phase = "solution"
        execute(compile_source(payload["code"], "<solution>", "exec"), namespace)
        phase = "test"
        for index, test in enumerate(payload["tests"]):
            execute(compile_source(test, "<test-%d>" % index, "exec"), namespace)
            completed += 1
    except base_exception as exc:
        kind = "syntax_error" if isinstance(exc, syntax_error) else (
            "assertion_error" if isinstance(exc, assertion_error) else "runtime_error"
        )
        # Do not invoke arbitrary exception __str__ or render source/tracebacks.
        report({"event": "complete", "status": "failed", "failure": kind,
                "phase": phase, "test_index": index, "tests_passed": completed,
                "exception": exception_type(exc).__name__[:128]})
    else:
        report({"event": "complete", "status": "passed", "tests_passed": completed})

main()
"""


def _interpreter() -> Path:
    executable = Path(sys.executable).resolve(strict=True)
    if sysconfig.get_config_var("PYTHONFRAMEWORK"):
        stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
        application = (
            stdlib.parent.parent / "Resources/Python.app/Contents/MacOS/Python"
        )
        if application.is_file():
            # Homebrew's bin/python is a launcher that needs posix_spawn. Start
            # the actual interpreter directly, keeping process-fork denied.
            executable = application
    return executable


def _quoted(path) -> str:
    # JSON string quoting also escapes the SBPL path string; reject non-ASCII
    # paths because JSON's \u escapes are not an SBPL portability guarantee.
    value = str(path)
    if not value.isascii() or any(ord(char) < 32 for char in value):
        raise ValueError("sandbox paths must be printable ASCII")
    return json.dumps(value)


def _profilereads(executable: Path, stdlib: Path):
    library_dir = Path(sysconfig.get_config_var("LIBDIR") or "").resolve()
    reads = {executable, Path(SANDBOX_LIBRARY)}
    read_subpaths = {stdlib, Path("/usr/lib"), Path("/System/Library")}
    if library_dir.is_dir():
        read_subpaths.add(library_dir)
    framework = sysconfig.get_config_var("PYTHONFRAMEWORK")
    if framework:
        reads.add((stdlib.parent.parent / framework).resolve(strict=True))
        return reads, read_subpaths
    library = sysconfig.get_config_var("LDLIBRARY")
    if not library:
        return reads, read_subpaths
    candidate = Path(sysconfig.get_config_var("LIBDIR")) / library
    if candidate.is_file():
        reads.add(candidate.resolve())
    return reads, read_subpaths


def _profile(work: Path) -> str:
    executable = _interpreter()
    stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    reads, read_subpaths = _profilereads(executable, stdlib)
    literals = " ".join(f"(literal {_quoted(path)})" for path in sorted(reads))
    ancestors = {
        parent
        for path in reads | read_subpaths | {stdlib, work}
        for parent in path.parents
    }
    metadata = " ".join(f"(literal {_quoted(path)})" for path in sorted(ancestors))
    allowed_subpaths = " ".join(
        f"(subpath {_quoted(path)})" for path in sorted(read_subpaths)
    )
    return f"""(version 1)
(deny default)
(import "dyld-support.sb")
(allow sysctl-read)
(allow process-exec (literal {_quoted(executable)}))
(allow file-read-metadata {metadata})
(allow file-read* file-map-executable
    {literals}
    {allowed_subpaths})
(deny file-read* file-map-executable (subpath {_quoted(stdlib / "site-packages")}))
(allow file-read* file-write* (subpath {_quoted(work)}))
(allow file-read* (literal "/dev/urandom") (literal "/dev/random"))
"""


def _drainstep(selector, output, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return "timeout", True
    for key, _ in selector.select(min(remaining, 0.05)):
        chunk = os.read(key.fd, 4096)
        if not chunk:
            selector.unregister(key.fileobj)
            continue
        buffer = key.data
        limit = MAX_OUTPUT_BYTES if buffer is output else MAX_PROTOCOL_BYTES
        room = limit - len(buffer)
        buffer.extend(chunk[:room])
        if len(chunk) > room:
            reason = "output_limit" if buffer is output else "protocol_error"
            return reason, True
    return None, False


def _drainloop(process, read_fd, output, protocol, deadline):
    assert process.stdout is not None and process.stderr is not None
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ, output)
        selector.register(process.stderr, selectors.EVENT_READ, output)
        selector.register(read_fd, selectors.EVENT_READ, protocol)
        reason = None
        while selector.get_map():
            reason, stop = _drainstep(selector, output, deadline)
            if stop:
                break
    if reason is not None:
        return reason
    try:
        process.wait(timeout=max(0.001, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        return "timeout"
    return None


def _terminate(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _collect(command, work, read_fd, writer, timeout):
    output, protocol = bytearray(), bytearray()
    deadline = time.monotonic() + timeout
    # No ambient credentials, PYTHONPATH, loader overrides, user site, shell,
    # inherited host descriptors, or host stdin. The only writable path is work.
    env = {"HOME": str(work), "TMPDIR": str(work), "LANG": "C", "LC_ALL": "C"}
    with subprocess.Popen(
        command,
        cwd=work,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(writer.fileno(),),
        start_new_session=True,
    ) as process:
        writer.close()
        try:
            reason = _drainloop(process, read_fd, output, protocol, deadline)
        finally:
            _terminate(process)
        return (
            process.returncode,
            reason,
            output.decode("utf-8", "replace"),
            bytes(protocol),
        )


def _validinput(code, setup, tests, timeout) -> bool:
    return (
        type(code) is str
        and type(setup) is str
        and type(tests) is list
        and 1 <= len(tests) <= MAX_TESTS
        and not any(type(test) is not str or not test.strip() for test in tests)
        and bool(code.strip())
        and type(timeout) in (int, float)
        and 0.1 <= timeout <= 30
    )


def _preflight(code, setup, tests, timeout) -> tuple[str | None, str]:
    if not _validinput(code, setup, tests, timeout):
        return (
            None,
            "Expected source strings, 1..256 tests, and timeout in [0.1, 30].",
        )
    payload = json.dumps({"code": code, "setup": setup, "tests": tests})
    if len(payload.encode("utf-8")) > MAX_INPUT_BYTES:
        return None, "Payload exceeds input limit."
    return payload, ""


def _execpipe(work: Path, profile: str, canary: Path, nonce: str, timeout: float):
    read_fd, write_fd = os.pipe()
    command = [
        SANDBOX_EXEC,
        "-p",
        profile,
        str(_interpreter()),
        "-I",
        "-S",
        "-B",
        "-c",
        _RUNNER,
        str(write_fd),
        nonce,
        str(canary),
        str(timeout),
    ]
    try:
        with os.fdopen(write_fd, "wb") as writer:
            return _collect(command, work, read_fd, writer, timeout)
    finally:
        os.close(read_fd)


def _runsandbox(payload: str, nonce: str, timeout: float):
    with tempfile.TemporaryDirectory(prefix="baux-codecheck-") as directory:
        root = Path(directory).resolve()
        work = root / "work"
        work.mkdir(mode=0o700)
        canary = root / "outside-canary"
        canary.write_text("benign isolation probe", encoding="ascii")
        profile = _profile(work)
        (work / "payload.json").write_text(payload, encoding="utf-8")
        return _execpipe(work, profile, canary, nonce, timeout)


def _parseevents(protocol: bytes, nonce: str):
    events = [json.loads(line) for line in protocol.splitlines()]
    if any(type(event) is not dict or event.get("nonce") != nonce for event in events):
        raise ValueError("invalid completion envelope")
    return events


def _validfailed(final: dict) -> bool:
    return (
        final.get("status") == "failed"
        and final.get("failure") in ("syntax_error", "assertion_error", "runtime_error")
        and final.get("phase") in ("setup", "solution", "test")
        and type(final.get("exception")) is str
        and "test_index" in final
        and (final["test_index"] is None or type(final["test_index"]) is int)
    )


def _finaloutcome(final: dict, total: int, extra: dict) -> tuple[str, str, dict]:
    count = final.get("tests_passed")
    if type(count) is not int or not 0 <= count <= total:
        return "protocol_error", "Invalid completed test count.", extra
    if final.get("status") == "passed" and count == total:
        return "passed", "", {**extra, "tests_passed": count}
    if _validfailed(final):
        fields = {
            key: final[key]
            for key in ("failure", "phase", "test_index", "tests_passed", "exception")
        }
        return "failed", "", {**extra, **fields}
    return "protocol_error", "Invalid completion result.", extra


def _outcome(events, returncode, reason, total: int) -> tuple[str, str, dict]:
    if not events or events[0].get("event") != "ready":
        detail = events[0].get("detail") if events else None
        if type(detail) is not str:
            detail = "Confinement bootstrap failed."
        return "sandbox_unavailable", detail, {}
    extra: dict = {"isolation_verified": True}
    if reason:
        return reason, "Runner exceeded its limit or violated its protocol.", extra
    if returncode in (-signal.SIGXCPU, -signal.SIGKILL, -signal.SIGXFSZ):
        return "resource_limit", "Runner terminated by a resource-limit signal.", extra
    if returncode != 0 or len(events) != 2 or events[1].get("event") != "complete":
        return "protocol_error", "Runner exited without valid completion.", extra
    return _finaloutcome(events[1], total, extra)


def check_solution(code: str, setup: str, tests: list[str], *, timeout=5.0) -> dict:
    """Execute setup, solution, then tests in one fresh macOS sandbox.

    Returns JSON data with status, tests_passed, tests_total, isolation_verified,
    limitations and bounded diagnostics. ``failed`` includes
    failure/phase/test_index/exception.
    Other statuses: invalid_input, sandbox_unavailable, timeout, output_limit,
    protocol_error, resource_limit. Timeout covers bootstrap and execution.

    Hard CPU, fd, process and per-file limits are installed before reading
    source. This is not a total RSS or aggregate disk quota: macOS rejects or
    ignores the corresponding setrlimit memory limits. Re-exec of this same
    interpreter retains confinement; creating child processes is denied by the OS.
    No claim is made of tamper-proof scoring against intentionally hostile code.
    """
    base = {
        "tests_passed": 0,
        "tests_total": len(tests) if type(tests) is list else 0,
        "diagnostics": "",
        "isolation_verified": False,
        "limitations": [
            "no_hard_memory_limit",
            "no_aggregate_disk_quota",
            "scores_not_tamper_proof",
        ],
    }

    def result(status, detail="", **fields):
        return {**base, "status": status, "detail": detail[:512], **fields}

    payload, error_detail = _preflight(code, setup, tests, timeout)
    if payload is None:
        return result("invalid_input", error_detail)
    if sys.platform != "darwin" or not os.access(SANDBOX_EXEC, os.X_OK):
        return result("sandbox_unavailable", "macOS sandbox-exec is required.")
    nonce = secrets.token_hex(32)
    try:
        returncode, reason, diagnostics, protocol = _runsandbox(payload, nonce, timeout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return result("sandbox_unavailable", f"{type(exc).__name__}: {exc}")

    base["diagnostics"] = diagnostics
    base["returncode"] = returncode
    try:
        events = _parseevents(protocol, nonce)
    except (ValueError, UnicodeError, RecursionError):
        return result("protocol_error", "Invalid runner protocol.")
    status, detail, fields = _outcome(events, returncode, reason, len(tests))
    return result(status, detail, **fields)
