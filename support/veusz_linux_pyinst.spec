# -*- mode: python -*-

# linux pyinstaller file

import glob
import os

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
    strip=True, # note this breaks std numpy wheel
    upx=False,
    console=True,
)

# add necessary documentation, licence
data_glob = [
    'VERSION',
    'ChangeLog',
    'AUTHORS',
    'README.md',
    'INSTALL.md',
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
outdir = f'veusz-{version}'
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.zipfiles,
    datas,
    strip=True,
    upx=False,
    name=outdir,
)

# make symlinks to make it easier to find files
print('Making symlinks')
symlink = [
    'ChangeLog',
    'AUTHORS',
    'README.md',
    'INSTALL.md',
    'embed.py',
    'COPYING',
    'examples',
]
for fn in symlink:
    os.symlink(f'_internal/{fn}', f'dist/{outdir}/{fn}')

# make a highly compressed tar file
print(f'Creating tar file')
tardir = 'tar'
tarfn = f'{tardir}/veusz-{version}-linux-x86_64.tar.xz'
os.makedirs(tardir, exist_ok=True)
cmd = f'tar -C dist/ --owner=0 --group=0 -cf - veusz-{version} | xz -9 -c - > {tarfn}'
print(cmd)
retn = os.system(cmd)
if retn != 0:
    raise RuntimeError('tar failed')
