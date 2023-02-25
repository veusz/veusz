"""

Code taken from
 distutils.command.install_data
and
 distutils.util

This was removed in setuptools, so we keep a copy here.

Original licence:

-----
1. This LICENSE AGREEMENT is between the Python Software Foundation
   ("PSF"), and the Individual or Organization ("Licensee") accessing
   and otherwise using Python 3.11.2 software in source or binary form
   and its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF
   hereby grants Licensee a nonexclusive, royalty-free, world-wide
   license to reproduce, analyze, test, perform and/or display
   publicly, prepare derivative works, distribute, and otherwise use
   Python 3.11.2 alone or in any derivative version, provided,
   however, that PSF's License Agreement and PSF's notice of
   copyright, i.e., "Copyright © 2001-2023 Python Software Foundation;
   All Rights Reserved" are retained in Python 3.11.2 alone or in any
   derivative version prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on
   or incorporates Python 3.11.2 or any part thereof, and wants to
   make the derivative work available to others as provided herein,
   then Licensee hereby agrees to include in any such work a brief
   summary of the changes made to Python 3.11.2.

4. PSF is making Python 3.11.2 available to Licensee on an "AS IS"
   basis.  PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
   IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND
   DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR
   FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON 3.11.2
   WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON
   3.11.2 FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
   LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
   PYTHON 3.11.2, OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE
   POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
   breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
   relationship of agency, partnership, or joint venture between PSF
   and Licensee.  This License Agreement does not grant permission to
   use PSF trademarks or trade name in a trademark sense to endorse or
   promote products or services of Licensee, or any third party.

8. By copying, installing or otherwise using Python 3.11.2, Licensee
   agrees to be bound by the terms and conditions of this License
   Agreement.
----

"""

import os
from setuptools import Command

def change_root (new_root, pathname):
    """Return 'pathname' with 'new_root' prepended.  If 'pathname' is
    relative, this is equivalent to "os.path.join(new_root,pathname)".
    Otherwise, it requires making 'pathname' relative and then joining the
    two, which is tricky on DOS/Windows and Mac OS.
    """
    if os.name == 'posix':
        if not os.path.isabs(pathname):
            return os.path.join(new_root, pathname)
        else:
            return os.path.join(new_root, pathname[1:])

    elif os.name == 'nt':
        (drive, path) = os.path.splitdrive(pathname)
        if path[0] == '\\':
            path = path[1:]
        return os.path.join(new_root, path)

    else:
        raise RuntimeError("nothing known about platform '%s'" % os.name)

def convert_path (pathname):
    """Return 'pathname' as a name that will work on the native filesystem,
    i.e. split it on '/' and put it back together again using the current
    directory separator.  Needed because filenames in the setup script are
    always supplied in Unix style, and have to be converted to the local
    convention before we can actually use them in the filesystem.  Raises
    ValueError on non-Unix-ish systems if 'pathname' either starts or
    ends with a slash.
    """
    if os.sep == '/':
        return pathname
    if not pathname:
        return pathname
    if pathname[0] == '/':
        raise ValueError("path '%s' cannot be absolute" % pathname)
    if pathname[-1] == '/':
        raise ValueError("path '%s' cannot end with '/'" % pathname)

    paths = pathname.split('/')
    while '.' in paths:
        paths.remove('.')
    if not paths:
        return os.curdir
    return os.path.join(*paths)


# contributed by Bastian Kleineidam
class install_data(Command):

    description = "install data files"

    user_options = [
        ('install-dir=', 'd',
         "base directory for installing data files "
         "(default: installation base dir)"),
        ('root=', None,
         "install everything relative to this alternate root directory"),
        ('force', 'f', "force installation (overwrite existing files)"),
        ]

    boolean_options = ['force']

    def initialize_options(self):
        self.install_dir = None
        self.outfiles = []
        self.root = None
        self.force = 0
        self.data_files = self.distribution.data_files
        self.warn_dir = 1

    def finalize_options(self):
        self.set_undefined_options('install',
                                   ('install_data', 'install_dir'),
                                   ('root', 'root'),
                                   ('force', 'force'),
                                  )

    def run(self):
        self.mkpath(self.install_dir)
        for f in self.data_files:
            if isinstance(f, str):
                # it's a simple file, so copy it
                f = convert_path(f)
                if self.warn_dir:
                    self.warn("setup script did not provide a directory for "
                              "'%s' -- installing right in '%s'" %
                              (f, self.install_dir))
                (out, _) = self.copy_file(f, self.install_dir)
                self.outfiles.append(out)
            else:
                # it's a tuple with path to install to and a list of files
                dir = convert_path(f[0])
                if not os.path.isabs(dir):
                    dir = os.path.join(self.install_dir, dir)
                elif self.root:
                    dir = change_root(self.root, dir)
                self.mkpath(dir)

                if f[1] == []:
                    # If there are no files listed, the user must be
                    # trying to create an empty directory, so add the
                    # directory to the list of output files.
                    self.outfiles.append(dir)
                else:
                    # Copy files, adding them to the list of output files.
                    for data in f[1]:
                        data = convert_path(data)
                        (out, _) = self.copy_file(data, dir)
                        self.outfiles.append(out)

    def get_inputs(self):
        return self.data_files or []

    def get_outputs(self):
        return self.outfiles
