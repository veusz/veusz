# Subclasses setuptools.command.build_ext,
# replacing it with a SIP version that compiles .sip -> .cpp
# before calling the original build_ext command.

# Originally written by Giovanni Bajo <rasky at develer dot com>
# Based on Pyrex.Distutils, written by Graham Fawcett and Darrel Gallion.

import os
import shutil
import subprocess
import tomli

from sysconfig import get_path
from setuptools.command.build_ext import build_ext

##################################################################

def find_on_path(names, mainname):
    """From a list of names of executables, find the 1st one on a path.

    mainname is the generic name to report
    """
    path = os.getenv('PATH', os.path.defpath)
    pathparts = path.split(os.path.pathsep)
    for cmd in names:
        resolved = shutil.which(cmd)
        if resolved:
            return resolved
    raise RuntimeError('Could not find %s executable' % mainname)

def read_command_output(cmd):
    """Get text from a run command."""
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        universal_newlines=True)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        raise RuntimeError('Command %s returned error' % str(cmd))
    return stdout.strip()

class sip_build_ext(build_ext):

    description = ('Compile SIP descriptions, then build C/C++ extensions '
                   '(compile/link to build directory)')

    user_options = build_ext.user_options + [
        ('qmake-exe=', None,
         'override qmake executable'),
        ('qt-include-dir=', None,
         'override Qt include directory'),
        ('qt-library-dir=', None,
         'override Qt library directory'),
        ('qt-libinfix=', None,
         'override Qt infix setting'),
        ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.qmake_exe = None
        self.qt_include_dir = None
        self.qt_library_dir = None
        self.qt_libinfix = None

    def _get_qmake(self, build_cmd):
        """Get qmake executable."""
        return (
            build_cmd.qmake_exe or
            os.environ.get('QMAKE_EXE') or
            find_on_path(
                ('qmake-qt6', 'qmake6', 'qmake', 'qmake6.exe', 'qmake.exe'),
                'qmake')
        )

    def _get_qt_inc_dir(self, build_cmd):
        """Get Qt include directory."""
        return (
            build_cmd.qt_include_dir or
            os.environ.get('QT_INCLUDE_DIR') or
            read_command_output(
                [self._get_qmake(build_cmd), '-query', 'QT_INSTALL_HEADERS'])
            )

    def _get_qt_library_dir(self, build_cmd):
        """Get Qt library directory."""
        return (
            build_cmd.qt_library_dir or
            os.environ.get('QT_LIBRARY_DIR') or
            read_command_output(
                [self._get_qmake(build_cmd), '-query', 'QT_INSTALL_LIBS'])
            )

    def _get_qt_libinfix(self, build_cmd):
        """Get QT_LIBINFIX setting.

        This is not much fun, as we have to try to find qconfig.pri,
        and parse it.
        """

        infix = build_cmd.qt_libinfix
        if infix is not None:
            return infix
        if 'QT_LIBINFIX' in os.environ:
            return os.environ['QT_LIBINFIX']

        # use this to find location of qconfig file
        archdir = read_command_output(
            [self._get_qmake(build_cmd), '-query', 'QT_INSTALL_ARCHDATA'])
        qconfig = os.path.join(archdir, 'mkspecs', 'qconfig.pri')

        libinfix = ''
        for line in open(qconfig):
            p = [x.strip() for x in line.split('=')]
            if p[0] == 'QT_LIBINFIX':
                libinfix = p[1]

        return libinfix

    def _is_qt_framework(self, build_cmd):
        """Is the Qt a framework?"""
        return os.path.exists(
            os.path.join(
                self._get_qt_library_dir(build_cmd), 'QtCore.framework'))

    def _get_cpp_includes(self, build_cmd):
        """Get list of include directories to add."""
        inc_dir = self._get_qt_inc_dir(build_cmd)
        incdirs = [inc_dir]
        for mod in ('QtCore', 'QtGui', 'QtWidgets', 'QtXml'):
            if self._is_qt_framework(build_cmd):
                incdirs.append(
                    os.path.join(
                        self._get_qt_library_dir(build_cmd),
                        mod+'.framework', 'Headers') )
            else:
                incdirs.append(os.path.join(inc_dir, mod))
        return incdirs

    def swig_sources(self, sources, extension=None):
        """Compile SIP files and setup Qt compile options."""

        if not self.extensions:
            return

        build_cmd = self.get_finalized_command('build_ext')

        # add directory of input files as include path
        indirs = list(set([os.path.dirname(x) for x in sources]))

        # Add the SIP and Qt include directories to the include path
        extension.include_dirs += indirs

        libinfix = self._get_qt_libinfix(build_cmd)

        # link against libraries
        if extension.language == 'c++':
            extension.include_dirs += self._get_cpp_includes(build_cmd)
            lib_dir = self._get_qt_library_dir(build_cmd)
            if self._is_qt_framework(build_cmd):
                # Mac OS framework
                extension.extra_link_args = [
                    '-F', os.path.join(lib_dir),
                    '-framework', 'QtGui'+libinfix,
                    '-framework', 'QtCore'+libinfix,
                    '-framework', 'QtXml'+libinfix,
                    '-framework', 'QtWidgets'+libinfix,
                    '-Wl,-rpath,@executable_path/Frameworks',
                    '-Wl,-rpath,' + lib_dir,
                ]
                extension.extra_compile_args = [
                    '-F', lib_dir,
                ]
            else:
                extension.libraries = [
                    'Qt6Gui'+libinfix,
                    'Qt6Core'+libinfix,
                    'Qt6Xml'+libinfix,
                    'Qt6Widgets'+libinfix,
                ]
            extension.library_dirs = [lib_dir]

            # may cause problems with compilers which don't allow this
            if self.compiler.compiler_type == 'unix':
                extension.extra_compile_args.append('-std=c++17')

        depends = extension.depends

        # Filter dependencies list: we are interested only in .sip files,
        # since the main .sip files can only depend on additional .sip
        # files. For instance, if a .h changes, there is no need to
        # run sip again.
        depends = [f for f in depends if os.path.splitext(f)[1] == '.sip']

        # Create the temporary directory if it does not exist already
        if not os.path.isdir(self.build_temp):
            os.makedirs(self.build_temp)

        # Collect the names of the source (.sip) files
        sip_sources = []
        sip_sources = [source for source in sources if source.endswith('.sip')]
        other_sources = [
            source for source in sources if not source.endswith('.sip')]
        generated_sources = []

        for sip in sip_sources:
            sip_basename = os.path.basename(sip)[:-4]
            sip_builddir = os.path.join(self.build_temp, 'sip-' + sip_basename)
            if not os.path.exists(sip_builddir) or self.force:
                os.makedirs(sip_builddir, exist_ok=True)
                self._sip_compile(sip, sip_builddir)

            # files get put in sip_builddir + modulename
            modulename = os.path.splitext(os.path.basename(sip))[0]
            dirname = os.path.join(sip_builddir, 'output', modulename)

            source_files = [
                os.path.join(dirname, fn)
                for fn in sorted(os.listdir(dirname))
                if fn.endswith(".cpp") or fn.endswith(".c")
            ]

            generated_sources.extend(source_files)

        return generated_sources + other_sources

    def _sip_compile(self, source, sip_builddir):
        """Compile sip file to sources."""

        pyqt6_include_dir = os.path.join(
            get_path('platlib'), 'PyQt6', 'bindings')
        pyqt6_toml = os.path.join(pyqt6_include_dir, 'QtCore', 'QtCore.toml')
        with open(pyqt6_toml, 'rb') as fin:
            pyqt6_cfg = tomli.load(fin)
        abi_version = pyqt6_cfg.get('sip-abi-version')

        modulename = os.path.splitext(os.path.basename(source))[0]
        srcdir = os.path.abspath(os.path.dirname(source))

        # location of sip output files
        output_dir = os.path.abspath(os.path.join(sip_builddir, 'output'))
        os.makedirs(output_dir)

        def toml_esc(s):
            s = s.replace("\\", "\\\\").replace('"', r'\"')
            return '"'+s+'"'

        toml_text=f'''
[build-system]
requires=["sip >= 5.5.0, <7"]
build-backend="sipbuild.api"

[tool.sip.metadata]
name="{modulename}"

[tool.sip.project]
sip-include-dirs=[{toml_esc(pyqt6_include_dir)}]
abi-version="{abi_version}"
build-dir={toml_esc(output_dir)}
sip-module="PyQt6.sip"
sip-files-dir={toml_esc(srcdir)}

[tool.sip.bindings.{modulename}]
pep484-pyi=false
protected-is-public=false
'''

        pyproject_fname = os.path.join(sip_builddir, 'pyproject.toml')
        with open(pyproject_fname, 'w') as fout:
            fout.write(toml_text)

        # generate the source files for the bindings
        build_cmd = shutil.which('sip-build')
        if not build_cmd:
            raise RuntimeError('Could not find sip-build command on PATH')
        subprocess.check_call([build_cmd, '--no-compile'], cwd=sip_builddir)

        # put sip header in correct location
        shutil.copyfile(
            os.path.join(output_dir, 'sip.h'),
            os.path.join(output_dir, modulename, 'sip.h')
        )
