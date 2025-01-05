# -*- mode: python -*-
import glob
import os.path

icon = os.path.abspath('icons\\veusz.ico')

analysis = Analysis(
    ['..\\veusz\\veusz_main.py'],
    hiddenimports=[
    ],
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
    icon=icon)

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
    strip=True,
    upx=False,
    name='veusz_main'
)
