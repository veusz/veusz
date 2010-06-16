# This is a pyinstaller script file

# to make the distribution you need to create a directory, e.g. temp
# add a symlink called veusz inside temp, pointing to the veusz directory

# you will need to edit the paths below to get the correct input directory

# $Id$

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
else:
    # unix
    name = 'buildveusz_pyinst/veusz'
    thisdir = '/home/jss/veusz.qt4'
    console = 1
    aargs = {}
    upx = False

a = Analysis([os.path.join(HOMEPATH,'support/_mountzlib.py'),
              os.path.join(HOMEPATH,'support/useUnicode.py'), 'veusz_main.py'],
             pathex=[thisdir, os.path.join(thisdir, 'temp')],
             excludes=['Tkinter', 'readline', 'termios'])
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name=name,
          debug=False,
          strip=True,
          upx=upx,
          console=console,
          **aargs)

# add necessary documentation, licence
binaries = a.binaries
for bin in ('VERSION', 'ChangeLog', 'AUTHORS', 'README', 'INSTALL', 'COPYING',
	'embed.py', '__init__.py'):
    binaries += [ (bin, bin, 'DATA') ]

# add various required files to distribution
for f in ( glob('windows/icons/*.png')  + glob('windows/icons/*.ico') +
	   glob('windows/icons/*.svg') +
           glob('examples/*.vsz') +
           glob('examples/*.dat') + glob('examples/*.csv') +
           glob('examples/*.py') +
           glob('dialogs/*.ui') + glob('widgets/data/*.dat')):
    binaries.append( (f, f, 'DATA') )

excludes = set(['ld-linux.so.2', 'libcom_err.so.3', 'libcrypto.so.4',
                'libdl.so.2', 'libfontconfig.so.1', 'libfreetype.so.6',
                'libgpm.so.1', 'libgssapi_krb5.so.2', 'libICE.so.6',
                'libk5crypto.so.3', 'libkrb5.so.3', 'libncurses.so.5',
                'libncursesw.so.5', 'libreadline.so.4', 'libresolv.so.2',
                'libSM.so.6', 'libssl.so.4', 'libutil.so.1', 'libX11.so.6',
                'libXext.so.6', 'libXrender.so.1', 'libz.so.1', 'readline.so',
                'termios.so'])

# remove libraries in the set above
# works a lot better if we do this...
binaries[:] = [b for b in binaries if b[0] not in excludes]

coll = COLLECT( exe,
                a.binaries, a.zipfiles, a.datas,
                strip=False,
                upx=upx,
                name='distveusz_main' )
