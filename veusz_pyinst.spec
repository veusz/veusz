# -*- mode: python -*-

# linux pyinstaller file

a = Analysis(['veusz/veusz_main.py'],
             pathex=['/home/jss/veusz'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='veusz',
          debug=False,
          strip=None,
          upx=False,
          console=True )

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
    'libfontconfig.so.1', 'libfreetype.so.6', 'libICE.so.6',
    'libSM.so.6', 'libX11.so.6', 'libXau.so.6', 'libXdmcp.so.6',
    'libXext.so.6', 'libXrender.so.1', 'libz.so.1', 'libutil.so.1',
    'libQtNetwork.so.4', 'libreadline.so.5'
    ])
# remove libraries in the set above
# works a lot better if we do this...
binaries[:] = [b for b in binaries if b[0] not in excludes]

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=False,
               name='veusz')
