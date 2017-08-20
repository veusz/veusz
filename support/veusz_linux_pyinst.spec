# -*- mode: python -*-
import glob
import os.path
# linux pyinstaller file

#thisdir=os.path.dirname(os.path.abspath(__file__))

a = Analysis(
    ['veusz/veusz_main.py'],
    pathex=[],
    hiddenimports=['h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy',
                   'iminuit', 'iminuit.latex', 'iminuit._plotting', 'iminuit.frontends'],
    hookspath=None,
    runtime_hooks=None)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='veusz',
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
for f in ( glob.glob('icons/*.png')  + glob.glob('icons/*.ico') +
	   glob.glob('icons/*.svg') +
           glob.glob('examples/*.vsz') +
           glob.glob('examples/*.dat') + glob.glob('examples/*.csv') +
           glob.glob('examples/*.py') +
           glob.glob('ui/*.ui') ):
    binaries.append( (f, f, 'DATA') )

excludes = set([
        'libXi.so.6', 'libX11-xcb.so.1', 'libX11.so.6',
        'libXext.so.6', 'libXau.so.6', 'libICE.so.6',
        'libreadline.so.6', 'readline.so',
        '_curses.so', 'libncursesw.so.5',
        'termios.so', 'libtinfo.so.5',
        'libz.so.1',
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
