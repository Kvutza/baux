def _patch_mujoco_macos_impl(ctx):
    archive = ctx.attrs.archive[DefaultInfo].default_outputs[0]
    src = archive.project("mujoco/libmujoco.3.17.0.dylib")
    out = ctx.actions.declare_output("libmujoco.3.17.0.dylib")
    script = ctx.actions.write(
        "patch-mujoco.sh",
        "#!/bin/sh\nset -eu\ncp \"$1\" \"$2\"\nchmod u+w \"$2\"\n/usr/bin/install_name_tool -id \"@rpath/libmujoco.3.17.0.dylib\" \"$2\"\n/usr/bin/codesign --force --sign - \"$2\"\n",
        is_executable = True,
    )
    ctx.actions.run([script, src, out.as_output()], category = "mujoco_patch")
    return [DefaultInfo(default_output = out)]

patch_mujoco_macos = rule(
    impl = _patch_mujoco_macos_impl,
    attrs = {
        "archive": attrs.dep(),
    },
)

def _gymnasium_assets_header_impl(ctx):
    archive = ctx.attrs.archive[DefaultInfo].default_outputs[0]
    out = ctx.actions.declare_output("gymnasium_assets.h")
    py_code = "\n".join([
        "import pathlib, sys",
        "root = pathlib.Path(sys.argv[1])",
        "out = pathlib.Path(sys.argv[2])",
        "cheetah = (root / 'gymnasium/envs/mujoco/assets/half_cheetah.xml').read_bytes()",
        "hopper = (root / 'gymnasium/envs/mujoco/assets/hopper.xml').read_bytes()",
        "def c_arr(name, data):",
        "    vals = ', '.join(str(b) for b in data) + ', 0'",
        "    return f'static const char {name}[] = {{ {vals} }};\\n'",
        "out.write_text('#pragma once\\n' + c_arr('GYMNASIUM_HALF_CHEETAH_XML', cheetah) + c_arr('GYMNASIUM_HOPPER_XML', hopper))",
    ])
    ctx.actions.run(["python3", "-c", py_code, archive, out.as_output()], category = "gymnasium_assets")
    return [DefaultInfo(default_output = out)]

gymnasium_assets_header = rule(
    impl = _gymnasium_assets_header_impl,
    attrs = {
        "archive": attrs.dep(),
    },
)
