# automate building the binary installer on windows

import sys
import shutil
import os.path
import os
import subprocess
import glob

root = r'c:\build\veusz'
nsisexe = r'c:\Program Files (x86)\NSIS\makensis.exe'

def call(args):
    print(' '.join(args))
    subprocess.check_call(args)

os.chdir(root)
for d in ('build', 'dist'):
    try:
        shutil.rmtree(d)
    except EnvironmentError:
        pass

call(['python', 'setup.py', 'build'])

for ext in glob.glob('build/lib.win32-*/veusz/helpers/*.pyd'):
    print('copying', ext)
    shutil.copy2(ext, 'veusz/helpers/')

os.environ['PYTHONPATH'] = root
os.environ['VEUSZ_RESOURCE_DIR'] = root
call(['python', 'tests/runselftest.py'])

del os.environ['PYTHONPATH']
del os.environ['VEUSZ_RESOURCE_DIR']

call(['pyinstaller', r'support\veusz_windows_pyinst.spec'])

version=open('VERSION').read().strip()

call([nsisexe, '/DPRODUCT_VERSION='+version, r'support\veusz_windows_setup.nsi'])
