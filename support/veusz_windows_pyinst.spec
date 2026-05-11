# -*- mode: python -*-

# windows pyinstaller file

import glob
import os.path
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
icon = str(ROOT / 'icons' / 'veusz.ico')
BUILD_ROOT = Path(os.environ.get('VEUSZ_BUILD_ROOT', ROOT)).resolve()


def _bridge_candidates():
    if os.name == 'nt':
        return ('microtexbridge.dll', 'libmicrotexbridge.dll')
    return ('libmicrotexbridge.so', 'microtexbridge.so')


def _tinyxml2_runtime_candidate():
    if os.name != 'nt':
        return None

    cache_path = BUILD_ROOT / 'build-microtexbridge' / 'CMakeCache.txt'
    if not cache_path.exists():
        return None

    lib_path = None
    for line in cache_path.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('TINYXML2_LIB:') or line.startswith('TINYXML2_LIB='):
            lib_path = line.split('=', 1)[1].strip()
            break
    if not lib_path:
        return None

    lib = Path(lib_path)
    if lib.suffix.lower() == '.dll':
        return lib if lib.exists() else None

    stems = [lib.stem]
    if lib.stem.endswith('d'):
        stems.append(lib.stem[:-1])
    else:
        stems.append(lib.stem + 'd')

    search_dirs = [
        lib.parent,
        lib.parent.parent / 'bin',
        lib.parent.parent / 'debug' / 'bin',
        lib.parent.parent.parent / 'bin',
        lib.parent.parent.parent / 'debug' / 'bin',
    ]
    for directory in search_dirs:
        for stem in stems:
            candidate = directory / f'{stem}.dll'
            if candidate.exists():
                return candidate
    return None


def _find_first_existing(root, names):
    if not root.exists():
        return None
    search_roots = [root]
    if os.name == 'nt':
        search_roots = [
            root / 'Release',
            root / 'RelWithDebInfo',
            root,
            root / 'Debug',
        ]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for name in names:
            for candidate in sorted(search_root.rglob(name)):
                if candidate.is_file():
                    return candidate
    return None


def _add_data_tree(srcroot, destroot, datas, kind):
    if not srcroot.is_dir():
        return
    for dirpath, _, filenames in os.walk(srcroot):
        relpath = Path(dirpath).relative_to(srcroot)
        for name in sorted(filenames):
            src = Path(dirpath) / name
            if relpath == Path('.'):
                dest = Path(destroot) / name
            else:
                dest = Path(destroot) / relpath / name
            datas.append((str(dest), str(src), kind))

analysis = Analysis(
    [str(ROOT / 'veusz' / 'veusz_main.py')],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[])

# TODO set version

pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    exclude_binaries=True,
    name='veusz.exe',
    debug=False,
    strip=None,
    upx=False,
    console=False,
    contents_directory='.', # do not use _internal
    icon=icon)

# add necessary documentation, licence
data_glob = [
    ROOT / 'VERSION',
    ROOT / 'ChangeLog',
    ROOT / 'AUTHORS',
    ROOT / 'README.md',
    ROOT / 'INSTALL.md',
    ROOT / 'COPYING',
    ROOT / 'icons' / '*.png',
    ROOT / 'icons' / '*.ico',
    ROOT / 'icons' / '*.svg',
    ROOT / 'examples' / '*.vsz',
    ROOT / 'examples' / '*.dat',
    ROOT / 'examples' / '*.csv',
    ROOT / 'examples' / '*.py',
    ROOT / 'ui' / '*.ui',
]

datas = analysis.datas
for pattern in data_glob:
    for fn in glob.glob(str(pattern)):
        datas.append((os.path.relpath(fn, ROOT), fn, 'DATA'))

_add_data_tree(
    ROOT / 'third_party' / 'MicroTeX' / 'res',
    Path('veusz') / 'microtex' / 'res',
    datas,
    'DATA',
)

microtex_license = ROOT / 'third_party' / 'MicroTeX' / 'LICENSE'
if microtex_license.exists():
    datas.append((
        os.path.join('veusz', 'microtex', 'LICENSE'),
        str(microtex_license),
        'DATA',
    ))

microtex_bridge = _find_first_existing(BUILD_ROOT / 'build-microtexbridge', _bridge_candidates())
if microtex_bridge is not None:
    analysis.binaries.append((
        os.path.join('veusz', 'microtex', microtex_bridge.name),
        str(microtex_bridge),
        'BINARY',
    ))

tinyxml2_runtime = _tinyxml2_runtime_candidate()
if tinyxml2_runtime is not None:
    analysis.binaries.append((
        os.path.join('veusz', 'microtex', tinyxml2_runtime.name),
        str(tinyxml2_runtime),
        'BINARY',
    ))

# add API files
datas += [
    ('veusz/embed.py', str(ROOT / 'veusz' / 'embed.py'), 'DATA'),
    ('veusz/__init__.py', str(ROOT / 'veusz' / '__init__.py'), 'DATA'),
]

# exclude files listed (currently unused)
excludes = set([
])
analysis.binaries[:] = [b for b in analysis.binaries if b[0] not in excludes]

# binaries += [
#     ('msvcp140.dll', r'c:\windows\system32\msvcp140.dll', 'BINARY'),
#     ('msvcrt.dll', r'c:\windows\system32\msvcrt.dll', 'BINARY'),
#     ('dcomp.dll', r'c:\windows\system32\dcomp.dll', 'BINARY'),
#     ]

coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.zipfiles,
    datas,
    strip=False,
    upx=False,
    name='veusz_main'
)
