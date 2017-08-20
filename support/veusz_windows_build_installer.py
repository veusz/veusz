# automate building the binary installer on windows

import sys
import shutil
import os.path
import os
import subprocess

root = r'c:\build\veusz'
nsisexe = r'c:\Program Files (x86)\NSIS\makensis.exe'

def call(args):
    print(' '.join(args))
    subprocess.call(args)

os.chdir(root)
for d in ('build', 'dist'):
    try:
        shutil.rmtree(d)
    except EnvironmentError:
        pass

call(['pyinstaller', r'support\veusz_windows_pyinst.spec'])

version=open('VERSION').read().strip()

call([nsisexe, '/DPRODUCT_VERSION='+version, r'support\veusz_windows_setup.nsi'])
