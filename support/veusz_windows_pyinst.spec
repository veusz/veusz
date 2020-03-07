# -*- mode: python -*-
import glob
import os.path

icon = os.path.abspath('icons\\veusz.ico')

a = Analysis(['..\\veusz\\veusz_main.py'],
             hiddenimports=['iminuit.iminuit_warnings','iminuit.latex','iminiuit._libiminuit','iminuit._minuit_methods'],
             hookspath=[],
             runtime_hooks=[])

# remove unnedded plugins
# for pdir in ('accessible', 'codecs', 'graphicssystems'):
#     a.binaries = [b for b in a.binaries if b[1].find(os.path.join('plugins', pdir)) == -1]

pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='veusz.exe',
          debug=False,
          strip=None,
          upx=False,
          console=False,
          icon=icon)

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

binaries += [
    ('msvcp140.dll', r'c:\windows\system32\msvcp140.dll', 'BINARY'),
    ('msvcrt.dll', r'c:\windows\system32\msvcrt.dll', 'BINARY'),
    ('dcomp.dll', r'c:\windows\system32\dcomp.dll', 'BINARY'),
    ]

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=False,
               name='veusz_main')
