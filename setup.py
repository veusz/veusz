#!/usr/bin/env python

import glob
import distutils.sysconfig
from distutils.core import setup

version = open('VERSION').read().strip()

maindir = '%s/veusz/' % distutils.sysconfig.get_python_lib()
datafiles = [ (maindir + 'windows/icons', glob.glob('windows/icons/*.png')),
              (maindir + 'images', ['images/logo.png'] ),
              (maindir, ['VERSION']) ]

setup(name = 'veusz',
      version = version,
      description = 'A scientific plotting package',
      author = 'Jeremy Sanders',
      author_email = 'jeremy@jeremysanders.net',
      url = 'http://home.gna.org/veusz/',
      classifiers = ['Programming Language :: Python'],
      package_dir = { 'veusz': '',
                      'veusz.dialogs': 'dialogs',
                      'veusz.document': 'document',
                      'veusz.setting': 'setting',
                      'veusz.utils': 'utils',
                      'veusz.widgets': 'widgets',
                      'veusz.windows': 'windows' },
      data_files = datafiles,
      packages = ['veusz',
                  'veusz.dialogs',
                  'veusz.document',
                  'veusz.setting',
                  'veusz.utils',
                  'veusz.widgets',
                  'veusz.windows']
      )
