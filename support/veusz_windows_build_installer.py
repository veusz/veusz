# automate building the binary installer on windows

import shutil
import os.path
import os
import subprocess
import glob

root = r'c:\build\veusz'
nsisexe = r'c:\Program Files (x86)\NSIS\makensis.exe'
verpatch = r'c:\build\verpatch\verpatch.exe'

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

for ext in glob.glob('build/lib.win-amd64-*/veusz/helpers/*.pyd'):
    print('copying', ext)
    shutil.copy2(ext, 'veusz/helpers/')

os.environ['PYTHONPATH'] = root
os.environ['VEUSZ_RESOURCE_DIR'] = root
call(['python', 'tests/runselftest.py'])

del os.environ['PYTHONPATH']
del os.environ['VEUSZ_RESOURCE_DIR']

call(['pyinstaller', r'support\veusz_windows_pyinst.spec'])

version=open('VERSION').read().strip()
ver4 = version
while len(ver4.split('.'))<4:
    ver4 += '.0'

call([verpatch, os.path.join(root, 'dist', 'veusz_main', 'veusz.exe'),
      ver4, '/va', '/pv', ver4, '/s', 'description', 'Veusz scientific plotting',
      '/s', 'product', 'Veusz', '/s', 'copyright', 'Jeremy Sanders and contributers'])

call([nsisexe, '/DPRODUCT_VERSION='+version, r'support\veusz_windows_setup.nsi'])
