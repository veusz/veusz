# This is a pyinstaller script file

# to make the distribution you need to create a directory, e.g. temp
# add a symlink called veusz inside temp, pointing to the veusz directory

# you may need to edit the paths below

from glob import glob
import os.path

thisdir = '/home/jss/veusz.qt4'
print thisdir
a = Analysis([os.path.join(HOMEPATH,'support/_mountzlib.py'), os.path.join(HOMEPATH,'support/useUnicode.py'), 'veusz_main.py'],
             pathex=[thisdir, os.path.join(thisdir, 'temp')])
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name='buildveusz_main/veusz_main',
          debug=False,
          strip=True,
          upx=True,
          console=1 )

# add necessary documentation, licence
binaries = a.binaries
for bin in ('VERSION', 'ChangeLog', 'AUTHORS', 'README', 'INSTALL', 'COPYING'):
    binaries += [ (bin, bin, 'DATA') ]

# add various required files to distribution
for f in ( glob('windows/icons/*.png') + glob('examples/*.vsz') +
           glob('examples/*.dat') + glob('examples/*.csv') +
           glob('dialogs/*.ui') + glob('widgets/data/*.dat')):
    binaries.append( (f, f, 'DATA') )

coll = COLLECT( exe,
               a.binaries,
               strip=False,
               upx=False,
               name='distveusz_main')
