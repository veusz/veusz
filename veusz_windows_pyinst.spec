# This is a pyinstaller script file

# to make the distribution you need to create a directory, e.g. temp
# add a symlink called veusz inside temp, pointing to the veusz directory

# you will need to edit the paths below to get the correct input directory

# $Id: veusz_pyinst.spec 607 2007-05-20 10:23:15Z jeremysanders $

from glob import glob
import os.path
import sys

# platform specific stuff
if sys.platform == 'win32':
    # windows
    name = r'buildveusz_pyinst\veusz.exe'
    thisdir = r'c:\source\veusz'
    console = 0
    aargs = {'icon': os.path.join(thisdir,'windows/icons/veusz.ico')}
    upx = True
    strip = False
else:
    # unix
    name = 'buildveusz_pyinst/veusz'
    thisdir = '/home/jss/veusz.qt4'
    console = 1
    aargs = {}
    upx = False

print name

a = Analysis([os.path.join(HOMEPATH,'support/_mountzlib.py'),
              os.path.join(HOMEPATH,'support/useUnicode.py'),
              'veusz_main.py'],
             pathex=[thisdir, os.path.join(thisdir, 'temp')],
             excludes=['Tkinter'])
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name=name,
          debug=False,
          strip=strip,
          upx=upx,
          console=console, **aargs)

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


# fix this to remove debugging dlls (and maybe mng)
ti = Tree(os.path.join(sys.prefix, 'Lib', 'site-packages', 'PyQt4', 'plugins',
                       'imageformats'), 'imageformats', excludes=['*.a'])
tic = Tree(os.path.join(sys.prefix, 'Lib', 'site-packages', 'PyQt4', 'plugins',
                       'iconengines'), 'iconengines', excludes=['*.a'])

coll = COLLECT( exe,
                a.binaries, ti, tic,
                strip=strip,
                upx=upx,
                name='distveusz_main' )
