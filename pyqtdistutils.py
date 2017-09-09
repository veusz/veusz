# Subclasses disutils.command.build_ext,
# replacing it with a SIP version that compiles .sip -> .cpp
# before calling the original build_ext command.
# Written by Giovanni Bajo <rasky at develer dot com>
# Based on Pyrex.Distutils, written by Graham Fawcett and Darrel Gallion.

from __future__ import division, print_function, absolute_import
import os

from distutils.sysconfig import customize_compiler
import distutils.command.build_ext
from distutils.dep_util import newer, newer_group

import sipconfig
import PyQt5.QtCore

##################################################################
# try to get various useful things we need in order to build
# this is likely to break, I'm sure

QT_LIB_DIR = PyQt5.QtCore.QLibraryInfo.location(
    PyQt5.QtCore.QLibraryInfo.LibrariesPath)
QT_INC_DIR = PyQt5.QtCore.QLibraryInfo.location(
    PyQt5.QtCore.QLibraryInfo.HeadersPath)
QT_IS_FRAMEWORK = os.path.exists(
    os.path.join(QT_LIB_DIR, 'QtCore.framework') )

SIP_FLAGS = PyQt5.QtCore.PYQT_CONFIGURATION['sip_flags']

PYQT_SIP_DIR = os.path.join(
    sipconfig.Configuration().default_sip_dir, 'PyQt5')

SIP_BIN = sipconfig.Configuration().sip_bin
SIP_INC_DIR = sipconfig.Configuration().sip_inc_dir

##################################################################

def replace_suffix(path, new_suffix):
    return os.path.splitext(path)[0] + new_suffix

class build_ext (distutils.command.build_ext.build_ext):

    description = ('Compile SIP descriptions, then build C/C++ extensions '
                   '(compile/link to build directory)')

    def _get_sip_output_list(self, sbf):
        '''
        Parse the sbf file specified to extract the name of the generated source
        files. Make them absolute assuming they reside in the temp directory.
        '''
        for L in open(sbf):
            key, value = L.split('=', 1)
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

        # add directory of input files as include path
        indirs = list(set([os.path.dirname(x) for x in sources]))

        # Add the SIP and Qt include directories to the include path
        extension.include_dirs += [SIP_INC_DIR] + indirs

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
                    # not sure how to detect below, so hard coded
                    '-std=gnu++11',
                    ]
            else:
                extension.libraries = [
                    'Qt5Gui', 'Qt5Core', 'Qt5Xml', 'Qt5Widgets']
            extension.library_dirs = [QT_LIB_DIR]

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
                self._sip_compile(sip, sbf)
            out = self._get_sip_output_list(sbf)
            generated_sources.extend(out)

        return generated_sources + other_sources

    def _sip_compile(self, source, sbf):
        self.spawn([SIP_BIN,
                    '-c', self.build_temp,
                    ] + SIP_FLAGS.split() + [
                    '-I', PYQT_SIP_DIR,
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
