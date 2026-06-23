import ctypes
import os
from collections import OrderedDict
import re
import subprocess
import threading
import sys
from pathlib import Path


def _bridge_filenames():
    if os.name == "nt":
        return ("microtexbridge.dll", "libmicrotexbridge.dll")
    if sys.platform == "darwin":
        return ("libmicrotexbridge.dylib", "microtexbridge.dylib")
    return ("libmicrotexbridge.so", "microtexbridge.so")


def _microtex_library_filenames():
    if os.name == "nt":
        return ("LaTeX.lib", "libLaTeX.a")
    if sys.platform == "darwin":
        return ("libLaTeX.a", "LaTeX.lib")
    return ("libLaTeX.a", "LaTeX.lib")


def _find_first_existing(root, filenames):
    if not root.exists():
        return None
    search_roots = [root]
    if os.name == "nt":
        search_roots = [
            root / "Release",
            root / "RelWithDebInfo",
            root,
            root / "Debug",
        ]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for filename in filenames:
            for candidate in sorted(search_root.rglob(filename)):
                if candidate.is_file():
                    return candidate
    return None


def _cmake_build_cmd(build_dir):
    cmd = ["cmake", "--build", str(build_dir), "-j2"]
    if os.name == "nt":
        cmd += ["--config", "Release"]
    return cmd


def _packaged_bridge_candidates():
    packaged_root = _package_root() / "microtex"
    for filename in _bridge_filenames():
        yield packaged_root / filename


def _build_bridge_candidate():
    return _find_first_existing(_bridge_build_dir(), _bridge_filenames())


def _package_root():
    return Path(__file__).resolve().parents[1]


def _veusz_root():
    package_root = _package_root()
    if (package_root / "VERSION").exists():
        return package_root
    return Path(__file__).resolve().parents[2]


def _build_root():
    env = os.environ.get("VEUSZ_BUILD_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _veusz_root()


def _bridge_build_dir():
    return _build_root() / "build-microtexbridge"


def _microtex_src_root():
    return _veusz_root() / "third_party" / "MicroTeX"


def _microtex_build_root():
    return _build_root() / "build-microtex"


def _read_cache_value(cache_path, name):
    if not cache_path.exists():
        return None
    pattern = re.compile(rf"^{re.escape(name)}(?::[^=]+)?=(.*)$")
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1).strip()
    return None


def _guess_qt6_dir():
    env = os.environ.get("Qt6_DIR")
    if env:
        return env

    for cache_path in (
        _microtex_build_root() / "CMakeCache.txt",
        _bridge_build_dir() / "CMakeCache.txt",
    ):
        qt6_dir = _read_cache_value(cache_path, "Qt6_DIR")
        if qt6_dir:
            return qt6_dir

    try:
        import PyQt6
    except ImportError:
        return None

    candidate = Path(PyQt6.__file__).resolve().parent / "Qt6" / "lib" / "cmake" / "Qt6"
    if candidate.exists():
        return str(candidate)
    return None


def _build_microtex():
    build_dir = _microtex_build_root()
    src_dir = _microtex_src_root()

    if not src_dir.exists():
        raise RuntimeError(f"MicroTeX source not found: {src_dir}")

    cmake_configure = [
        "cmake",
        "-S", str(src_dir),
        "-B", str(build_dir),
        "-DQT=ON",
        "-DBUILD_EXAMPLE=OFF",
    ]
    qt6_dir = _guess_qt6_dir()
    if qt6_dir:
        cmake_configure.append(f"-DQt6_DIR={qt6_dir}")

    subprocess.run(cmake_configure, check=True, cwd=_veusz_root())
    subprocess.run(
        _cmake_build_cmd(build_dir),
        check=True,
        cwd=_veusz_root(),
    )


def _build_bridge():
    build_dir = _bridge_build_dir()
    src_dir = _veusz_root() / "src" / "microtexbridge"
    microtex_src = _microtex_src_root()
    microtex_build = _microtex_build_root()
    microtex_lib = _find_first_existing(microtex_build, _microtex_library_filenames())
    cache = microtex_build / "CMakeCache.txt"

    if not microtex_src.exists():
        raise RuntimeError(f"MicroTeX source not found: {microtex_src}")
    if microtex_lib is None:
        _build_microtex()
        microtex_lib = _find_first_existing(microtex_build, _microtex_library_filenames())
    if not microtex_lib:
        raise RuntimeError(f"MicroTeX static library not found in {microtex_build}")

    cmake_configure = [
        "cmake",
        "-S", str(src_dir),
        "-B", str(build_dir),
        f"-DMICROTEX_SRC={microtex_src}",
        f"-DMICROTEX_LIB={microtex_lib}",
    ]
    qt6_dir = _read_cache_value(cache, "Qt6_DIR") or _guess_qt6_dir()
    if qt6_dir:
        cmake_configure.append(f"-DQt6_DIR={qt6_dir}")

    subprocess.run(cmake_configure, check=True, cwd=_veusz_root())
    subprocess.run(
        _cmake_build_cmd(build_dir),
        check=True,
        cwd=_veusz_root(),
    )


_LIB = None
_LOCK = threading.RLock()
_DLL_DIRS = []
_SVG_CACHE = OrderedDict()
_SVG_CACHE_LIMIT = 128


def _cache_get(key):
    data = _SVG_CACHE.get(key)
    if data is None:
        return None
    _SVG_CACHE.move_to_end(key)
    return data


def _cache_put(key, data):
    _SVG_CACHE[key] = data
    _SVG_CACHE.move_to_end(key)
    while len(_SVG_CACHE) > _SVG_CACHE_LIMIT:
        _SVG_CACHE.popitem(last=False)


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB

    env = os.environ.get("VEUSZ_MICROTEX_BRIDGE")
    if env:
        env_path = Path(env)
        if env_path.exists():
            return _load_library(env_path)

    for path in _packaged_bridge_candidates():
        if path.exists():
            return _load_library(path)

    path = _build_bridge_candidate()
    if path is None:
        _build_bridge()
        path = _build_bridge_candidate()
    if path and path.exists():
        return _load_library(path)
    return None


def _load_library(path):
    global _LIB
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        try:
            _DLL_DIRS.append(os.add_dll_directory(str(path.parent)))
        except OSError:
            pass
    lib = ctypes.CDLL(str(path))
    lib.microtex_render_svg.argtypes = [
        ctypes.c_char_p,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.microtex_render_svg.restype = ctypes.c_int
    lib.microtex_free.argtypes = [ctypes.c_void_p]
    lib.microtex_free.restype = None
    _LIB = lib
    return lib


def _resource_root():
    env = os.environ.get("VEUSZ_MICROTEX_RES")
    if env:
        return env
    packaged_root = _package_root() / "microtex" / "res"
    if packaged_root.exists():
        os.environ["VEUSZ_MICROTEX_RES"] = str(packaged_root)
        return str(packaged_root)
    root = _microtex_src_root() / "res"
    if root.exists():
        os.environ["VEUSZ_MICROTEX_RES"] = str(root)
        return str(root)
    return None


def render_svg(tex, text_size=20.0, width=720, foreground="#000000", background="transparent"):
    with _LOCK:
        lib = _load()
        if lib is None:
            raise RuntimeError("MicroTeX bridge library not found")
        resource_root = _resource_root()
        if resource_root is None:
            raise RuntimeError("MicroTeX resource root not found")
        cache_key = (
            tex,
            float(text_size),
            int(width),
            foreground or "",
            background or "",
            resource_root,
        )
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        out_svg = ctypes.c_void_p()
        out_len = ctypes.c_size_t()
        out_error = ctypes.c_void_p()
        rc = lib.microtex_render_svg(
            tex.encode("utf-8"),
            float(text_size),
            int(width),
            foreground.encode("utf-8") if foreground else None,
            background.encode("utf-8") if background else None,
            ctypes.byref(out_svg),
            ctypes.byref(out_len),
            ctypes.byref(out_error),
        )
        if rc != 0:
            msg = "MicroTeX bridge failed"
            if out_error.value:
                msg = ctypes.cast(out_error, ctypes.c_char_p).value.decode("utf-8", "replace")
                lib.microtex_free(out_error)
            raise RuntimeError(msg)
        try:
            data = ctypes.string_at(out_svg.value, out_len.value)
            _cache_put(cache_key, data)
            return data
        finally:
            if out_svg.value:
                lib.microtex_free(out_svg)
