# -*- mode: python -*-

# linux pyinstaller file

import glob

a = Analysis(
    ['../veusz/veusz_main.py'],
    pathex=[],
    hiddenimports=[
        'h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy',
        'iminuit', 'iminuit.latex', 'iminuit.util'
        ],
    hookspath=None,
    runtime_hooks=None)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='veusz.exe',
    debug=False,
    strip=True,
    upx=False,
    console=True,
    )

# add necessary documentation, licence
binaries = a.binaries
for bin in ('VERSION', 'ChangeLog', 'AUTHORS', 'README', 'INSTALL', 'COPYING'):
    binaries += [ (bin, bin, 'DATA') ]

binaries += [
    ('embed.py', 'veusz/embed.py', 'DATA'),
    ('__init__.py', 'veusz/__init__.py', 'DATA'),
    ]

# add various required files to distribution
data_glob = [
    'icons/*.png',
    'icons/*.ico',
    'icons/*.svg',
    'examples/*.vsz',
    'examples/*.dat',
    'examples/*.csv',
    'examples/*.py',
    'ui/*.ui',
]

for pattern in data_glob:
    for fn in glob.glob(pattern):
        binaries.append((fn, fn, 'DATA'))

excludes = set([
])
# remove libraries in the set above
binaries[:] = [b for b in binaries if b[0] not in excludes]

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=False,
    name='veusz')
