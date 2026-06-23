# automate building the binary installer on windows

import os.path
import os
import subprocess
import glob
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NSISEXE = os.environ.get('NSIS_EXE', r'c:\Program Files (x86)\NSIS\makensis.exe')
VERPATCH = os.environ.get('VERPATCH_EXE', r'c:\build\verpatch\verpatch.exe')

def call(args):
    args = [str(arg) for arg in args]
    print(' '.join(args))
    subprocess.check_call(args, cwd=str(ROOT))

for d in ('build', 'dist'):
    try:
        shutil.rmtree(ROOT / d)
    except EnvironmentError:
        pass

call([sys.executable, 'setup.py', 'build'])

for ext in glob.glob(str(ROOT / 'build' / 'lib.win-amd64-*' / 'veusz' / 'helpers' / '*.pyd')):
    print('copying', ext)
    shutil.copy2(ext, ROOT / 'veusz' / 'helpers')

os.environ['PYTHONPATH'] = str(ROOT)
os.environ['VEUSZ_RESOURCE_DIR'] = str(ROOT)
call([sys.executable, 'tests/runselftest.py'])

del os.environ['PYTHONPATH']
del os.environ['VEUSZ_RESOURCE_DIR']

call([sys.executable, '-m', 'PyInstaller', ROOT / 'support' / 'veusz_windows_pyinst.spec'])

generated_nsi = ROOT / 'build' / 'veusz_windows_setup.nsi'
call([sys.executable, ROOT / 'support' / 'veusz_windows_make_nsi.py', generated_nsi])

version = (ROOT / 'VERSION').read_text(encoding='utf-8').strip()
ver4 = version
while len(ver4.split('.'))<4:
    ver4 += '.0'

call([VERPATCH, ROOT / 'dist' / 'veusz_main' / 'veusz.exe',
      ver4, '/va', '/pv', ver4, '/s', 'description', 'Veusz scientific plotting',
      '/s', 'product', 'Veusz', '/s', 'copyright', 'Jeremy Sanders and contributers'])

call([NSISEXE, '/DPRODUCT_VERSION='+version, generated_nsi])
