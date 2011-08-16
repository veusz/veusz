# -*- mode: python -*-

from glob import glob
import os.path

a = Analysis([os.path.join(HOMEPATH,'support\\_mountzlib.py'), os.path.join(CONFIGDIR,'support\\useUnicode.py'), 'veusz_main.py'],
             pathex=['C:\\src\\veusz-msvc'],
             excludes=['Tkinter'])

# get rid of debugging binaries
a.binaries = [b for b in a.binaries if b[0][-6:] != 'd4.dll']

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
          exclude_binaries=1,
          name=os.path.join('buildveusz_pyinst', 'veusz.exe'),
          debug=False,
          strip=None,
          upx=False,
          console=False,
          icon='windows\\icons\\veusz.ico')

# add necessary documentation, licence
binaries = a.binaries
for bin in ('VERSION', 'ChangeLog', 'AUTHORS', 'README', 'INSTALL', 'COPYING'):
    binaries += [ (bin, bin, 'DATA') ]

# add various required files to distribution
for f in ( glob('windows/icons/*.png') + glob('windows/icons/*.ico') +
           glob('windows/icons/*.svg') +
           glob('examples/*.vsz') +
           glob('examples/*.dat') + glob('examples/*.csv') +
           glob('examples/*.py') +
           glob('dialogs/*.ui') + glob('widgets/data/*.dat')):
    binaries.append( (f, f, 'DATA') )

coll = COLLECT( exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=False,
               name='distveusz_main')
