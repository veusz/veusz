# Subclasses disutils.command.build_ext,
# replacing it with a SIP version that compiles .sip -> .cpp
# before calling the original build_ext command.
# Written by Giovanni Bajo <rasky at develer dot com>
# Based on Pyrex.Distutils, written by Graham Fawcett and Darrel Gallion.

from __future__ import division, print_function, absolute_import
import os
import sys
import sysconfig

from distutils.sysconfig import customize_compiler
import distutils.command.build_ext
from distutils.dep_util import newer, newer_group

import PyQt5.QtCore

##################################################################
# try to get various useful things we need in order to build

QT_LIB_DIR = PyQt5.QtCore.QLibraryInfo.location(
    PyQt5.QtCore.QLibraryInfo.LibrariesPath)
QT_INC_DIR = PyQt5.QtCore.QLibraryInfo.location(
    PyQt5.QtCore.QLibraryInfo.HeadersPath)
QT_IS_FRAMEWORK = os.path.exists(
    os.path.join(QT_LIB_DIR, 'QtCore.framework') )

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

def findSipOnPath():
    '''Get SIP executable from PATH.'''
    path = os.getenv('PATH', os.path.defpath)
    pathparts = path.split(os.path.pathsep)
    for cmd in 'sip', 'sip5', 'sip.exe', 'sip5.exe':
        for dirname in pathparts:
            cmdtry = os.path.join(dirname.strip('"'), cmd)
            if os.path.isfile(cmdtry) and os.access(cmdtry, os.X_OK):
                return cmdtry
    raise RuntimeError('Could not find SIP executable')

def replace_suffix(path, new_suffix):
    return os.path.splitext(path)[0] + new_suffix

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
        ]

    def initialize_options(self):
        distutils.command.build_ext.build_ext.initialize_options(self)
        self.sip_exe = None
        self.sip_dir = None
        self.sip_include_dir = None

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

    def get_cpp_includes(self):
        incdirs = [QT_INC_DIR]
        for mod in ('QtCore', 'QtGui', 'QtWidgets', 'QtXml'):
            if QT_IS_FRAMEWORK:
                incdirs.append(
                    os.path.join(QT_LIB_DIR, mod + '.framework', 'Headers') )
            else:
                incdirs.append( os.path.join(QT_INC_DIR, mod) )
        return incdirs

    def swig_sources(self, sources, extension=None):
        """Compile SIP files and setup Qt compile options."""

        if not self.extensions:
            return

        build_cmd = self.get_finalized_command('build_ext')

        # executable in order of priority using or
        sip_exe = build_cmd.sip_exe or DEF_SIP_BIN or findSipOnPath()
        sip_inc_dir = (
            build_cmd.sip_include_dir or DEF_SIP_INC_DIR or
            sysconfig.get_path('include'))
        # python data directory
        data_dir = sys.prefix if sys.platform=='win32' else sys.prefix+'/share'
        sip_dir = (
            build_cmd.sip_dir or DEF_SIP_DIR or
            os.path.join(data_dir, 'sip'))

        # add directory of input files as include path
        indirs = list(set([os.path.dirname(x) for x in sources]))

        # Add the SIP and Qt include directories to the include path
        extension.include_dirs += [sip_inc_dir] + indirs

        # link against libraries
        if extension.language == 'c++':
            extension.include_dirs += self.get_cpp_includes()

            if QT_IS_FRAMEWORK:
                # Mac OS framework
                extension.extra_link_args = [
                    '-F', os.path.join(QT_LIB_DIR),
                    '-framework', 'QtGui',
                    '-framework', 'QtCore',
                    '-framework', 'QtXml',
                    '-framework', 'QtWidgets',
                    '-Wl,-rpath,@executable_path/Frameworks',
                    '-Wl,-rpath,' + QT_LIB_DIR
                    ]
                extension.extra_compile_args = [
                    '-F', QT_LIB_DIR,
                    ]
            else:
                extension.libraries = [
                    'Qt5Gui', 'Qt5Core', 'Qt5Xml', 'Qt5Widgets']
            extension.library_dirs = [QT_LIB_DIR]

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
        self.spawn([sip_exe,
                    '-c', self.build_temp,
                    ] + SIP_FLAGS.split() + [
                    '-I', os.path.join(sip_dir, 'PyQt5'),
                    '-b', sbf,
                    source])

    def build_extensions(self):
        # remove annoying flag which causes warning for c++ sources
        # https://stackoverflow.com/a/36293331/351771
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        distutils.command.build_ext.build_ext.build_extensions(self)
