PYTHON_VERSION = read_config("baux", "python_version", "3.14")

_PYTHON_CONFIGS = {
    "3.14": struct(
        abi = "cp314",
        environment = "baux-py314",
        extension_infix = "314",
    ),
    "3.15": struct(
        abi = "cp315",
        environment = "baux-py315",
        extension_infix = "315",
    ),
}
if PYTHON_VERSION not in _PYTHON_CONFIGS:
    fail("baux.python_version must be 3.14 through 3.15, got {}".format(PYTHON_VERSION))

_PYTHON = _PYTHON_CONFIGS[PYTHON_VERSION]

PYTHON_ABI = _PYTHON.abi
PYTHON_ENVIRONMENT = _PYTHON.environment
PYTHON_REQUIRES = ">=3.14,<3.16"

def python_extension_suffix(platform):
    if platform == "linux-aarch64":
        return ".cpython-{}-aarch64-linux-gnu.so".format(_PYTHON.extension_infix)
    if platform == "linux-x86_64":
        return ".cpython-{}-x86_64-linux-gnu.so".format(_PYTHON.extension_infix)
    if platform == "macos-arm64":
        return ".cpython-{}-darwin.so".format(_PYTHON.extension_infix)
    fail("unsupported Python extension platform: {}".format(platform))
