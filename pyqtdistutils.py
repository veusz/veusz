# Subclasses disutils.command.build_ext,
# replacing it with a SIP version that compiles .sip -> .cpp
# before calling the original build_ext command.
# Written by Giovanni Bajo <rasky at develer dot com>
# Based on Pyrex.Distutils, written by Graham Fawcett and Darrel Gallion.

from __future__ import division, print_function, absolute_import
import os
import sys
import sysconfig
import subprocess

from distutils.sysconfig import customize_compiler
import distutils.command.build_ext
from distutils.dep_util import newer, newer_group

import PyQt5.QtCore

##################################################################
# try to get various useful things we need in order to build

SIP_FLAGS = PyQt5.QtCore.PYQT_CONFIGURATION['sip_flags']

try:
    # sipconfig is deprecated but necessary to find sip reliably
    import sipconfig
except ImportError:
    # try to guess locations
    DEF_SIP_DIR = None
    DEF_SIP_BIN = None
    DEF_SIP_INC_DIR = None
else:
    # use sipconfig if found
    DEF_SIP_DIR = sipconfig.Configuration().default_sip_dir
    DEF_SIP_BIN = sipconfig.Configuration().sip_bin
    DEF_SIP_INC_DIR = sipconfig.Configuration().sip_inc_dir

##################################################################

def replace_suffix(path, new_suffix):
    return os.path.splitext(path)[0] + new_suffix

def find_on_path(names, mainname):
    """From a list of names of executables, find the 1st one on a path.

    mainname is the generic name to report
    """
    path = os.getenv('PATH', os.path.defpath)
    pathparts = path.split(os.path.pathsep)
    for cmd in names:
        for dirname in pathparts:
            cmdtry = os.path.join(dirname.strip('"'), cmd)
            if os.path.isfile(cmdtry) and os.access(cmdtry, os.X_OK):
                return cmdtry
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

class build_ext(distutils.command.build_ext.build_ext):

    description = ('Compile SIP descriptions, then build C/C++ extensions '
                   '(compile/link to build directory)')

    user_options = distutils.command.build_ext.build_ext.user_options + [
        ('sip-exe=', None,
         'override sip executable'),
        ('sip-dir=', None,
         'override sip file directory'),
        ('sip-include-dir=', None,
         'override sip include directory'),
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
        distutils.command.build_ext.build_ext.initialize_options(self)
        self.sip_exe = None
        self.sip_dir = None
        self.sip_include_dir = None
        self.qmake_exe = None
        self.qt_include_dir = None
        self.qt_library_dir = None
        self.qt_libinfix = None

    def _get_sip_output_list(self, sbf):
        '''
        Parse the sbf file specified to extract the name of the generated source
        files. Make them absolute assuming they reside in the temp directory.
        '''
        for line in open(sbf):
            key, value = line.split('=', 1)
            if key.strip() == 'sources':
                out = []
                for o in value.split():
                    out.append(os.path.join(self.build_temp, o))
                return out
        raise RuntimeError('cannot parse SIP-generated "%s"' % sbf)

    def _get_sip_exe(self, build_cmd):
        """Get exe for sip. Sources are:
        --sip-exe option, environment, DEF_SIP_BIN, search on path."""
        return (
            build_cmd.sip_exe or
            os.environ.get('SIP_EXE') or
            DEF_SIP_BIN or
            find_on_path(
                ('sip5', 'sip-qt5', 'sip', 'sip5.exe', 'sip.exe'), 'sip')
        )

    def _get_sip_inc_dir(self, build_cmd):
        """Get include directory for sip."""
        return (
            build_cmd.sip_include_dir or
            os.environ.get('SIP_INCLUDE_DIR') or
            DEF_SIP_INC_DIR or
            sysconfig.get_path('include')
        )

    def _get_sip_dir(self, build_cmd):
        """Get sip directory."""
        data_dir = sys.prefix if sys.platform=='win32' else sys.prefix+'/share'
        return (
            build_cmd.sip_dir or
            os.environ.get('SIP_DIR') or
            DEF_SIP_DIR or
            os.path.join(data_dir, 'sip')
        )

    def _get_qmake(self, build_cmd):
        """Get qmake executable."""
        return (
            build_cmd.qmake_exe or
            os.environ.get('QMAKE_EXE') or
            find_on_path(
                ('qmake-qt5', 'qmake5', 'qmake', 'qmake5.exe', 'qmake.exe'),
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

        # executable in order of priority using or
        sip_exe = self._get_sip_exe(build_cmd)
        sip_inc_dir = self._get_sip_inc_dir(build_cmd)

        # python data directory
        sip_dir = self._get_sip_dir(build_cmd)

        # add directory of input files as include path
        indirs = list(set([os.path.dirname(x) for x in sources]))

        # Add the SIP and Qt include directories to the include path
        extension.include_dirs += [sip_inc_dir] + indirs

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
                    '-Wl,-rpath,' + lib_dir
                    ]
                extension.extra_compile_args = [
                    '-F', lib_dir,
                    ]
            else:
                extension.libraries = [
                    'Qt5Gui'+libinfix,
                    'Qt5Core'+libinfix,
                    'Qt5Xml'+libinfix,
                    'Qt5Widgets'+libinfix,
                ]
            extension.library_dirs = [lib_dir]

            # may cause problems with compilers which don't allow this
            if self.compiler.compiler_type == 'unix':
                extension.extra_compile_args.append('-std=c++11')

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
        other_sources = [source for source in sources
                         if not source.endswith('.sip')]
        generated_sources = []

        for sip in sip_sources:
            # Use the sbf file as dependency check
            sipbasename = os.path.basename(sip)
            sbf = os.path.join(self.build_temp,
                               replace_suffix(sipbasename, '.sbf'))
            if newer_group([sip]+depends, sbf) or self.force:
                self._sip_compile(sip_exe, sip_dir, sip, sbf)
            out = self._get_sip_output_list(sbf)
            generated_sources.extend(out)

        return generated_sources + other_sources

    def _sip_compile(self, sip_exe, sip_dir, source, sbf):
        """Compile sip file to sources."""
        self.spawn(
            [
                sip_exe,
                '-c', self.build_temp
            ] + SIP_FLAGS.split() + [
                '-I', os.path.join(sip_dir, 'PyQt5'),
                '-b', sbf,
                source
            ]
        )

    def build_extensions(self):
        # remove annoying flag which causes warning for c++ sources
        # https://stackoverflow.com/a/36293331/351771
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        distutils.command.build_ext.build_ext.build_extensions(self)
