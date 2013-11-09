# -*- mode: python -*-
a = Analysis(['veusz\\veusz_main.py'],
             pathex=['C:\\src\\veusz-msvc\\veusz'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)

# get rid of debugging binaries
#a.binaries = [b for b in a.binaries if b[0][-6:] != 'd4.dll']
# this doesn't work - have to go and delete the debugging libraries from Qt
# otherwise, we get sxs errors

# don't want kernel32, etc
a.binaries = [b for b in a.binaries if not (os.path.basename(b[0]) in
              ('kernel32.dll', 'Qt3Support4.dll',
               'QtNetwork4.dll', 'QtOpenGL4.dll', 'QtSql4.dll'))]

# remove unnedded plugins
for pdir in ('accessible', 'codecs', 'graphicssystems'):
    a.binaries = [b for b in a.binaries if b[1].find(os.path.join('plugins', pdir)) == -1]

pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='veusz.exe',
          debug=False,
          strip=None,
          upx=False,
          console=False,
          icon='icons\\veusz.ico')

# add necessary documentation, licence
binaries = a.binaries
for bin in ('VERSION', 'ChangeLog', 'AUTHORS', 'README', 'INSTALL', 'COPYING'):
    binaries += [ (bin, bin, 'DATA') ]

# add various required files to distribution
for f in ( glob.glob('icons/*.png') + glob.glob('icons/*.ico') +
           glob.glob('icons/*.svg') +
           glob.glob('examples/*.vsz') +
           glob.glob('examples/*.dat') + glob.glob('examples/*.csv') +
           glob.glob('examples/*.py') +
           glob.glob('ui/*.ui') ):
    binaries.append( (f, f, 'DATA') )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=False,
               name='veusz_main')
