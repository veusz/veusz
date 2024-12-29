# -*- mode: python -*-

# linux pyinstaller file

import glob

# get version
with open('VERSION') as fin:
    version = fin.read().strip()

analysis = Analysis(
    ['../veusz/veusz_main.py'],
    pathex=[],
    hiddenimports=[
        'h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy',
        'iminuit', 'iminuit.latex', 'iminuit.util'
        ],
    hookspath=None,
    runtime_hooks=None)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    exclude_binaries=True,
    name='veusz',
    debug=False,
    strip=True,
    upx=False,
    console=False,
)

# add necessary documentation, licence
data_glob = [
    'VERSION',
    'ChangeLog',
    'AUTHORS',
    'README',
    'INSTALL',
    'COPYING',
    'icons/*.png',
    'icons/*.ico',
    'icons/*.svg',
    'examples/*.vsz',
    'examples/*.dat',
    'examples/*.csv',
    'examples/*.py',
    'ui/*.ui',
]

datas = analysis.datas
for pattern in data_glob:
    for fn in glob.glob(pattern):
        datas.append((fn, fn, 'DATA'))

# add API files
datas += [
    ('embed.py', 'veusz/embed.py', 'DATA'),
    ('__init__.py', 'veusz/__init__.py', 'DATA'),
]

# exclude files listed (currently unused)
excludes = set([
])
analysis.binaries[:] = [b for b in analysis.binaries if b[0] not in excludes]

# collect together results
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.zipfiles,
    datas,
    strip=True,
    upx=False,
    name=f'veusz-{version}',
)
