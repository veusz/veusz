# Subclasses disutils.command.build_ext,
# replacing it with a SIP version that compiles .sip -> .cpp
# before calling the original build_ext command.
# Written by Giovanni Bajo <rasky at develer dot com>
# Based on Pyrex.Distutils, written by Graham Fawcett and Darrel Gallion.

from __future__ import division
import distutils.command.build_ext
from distutils.dep_util import newer, newer_group
import os
import sys

import sip
sip.setapi('QString', 2)

import sipconfig
import PyQt4.QtCore

##################################################################
# try to get various useful things we need in order to build
# this is likely to break, I'm sure

QT_LIB_DIR = PyQt4.QtCore.QLibraryInfo.location(
    PyQt4.QtCore.QLibraryInfo.LibrariesPath)
QT_INC_DIR = PyQt4.QtCore.QLibraryInfo.location(
    PyQt4.QtCore.QLibraryInfo.HeadersPath)
QT_IS_FRAMEWORK = os.path.exists(
    os.path.join(QT_LIB_DIR, 'QtCore.framework') )

try:
    # >= 4.10
    SIP_FLAGS = PyQt4.QtCore.PYQT_CONFIGURATION['sip_flags']
except:
    import PyQt4.pyqtconfig
    SIP_FLAGS = PyQt4.pyqtconfig.Configuration().pyqt_sip_flags

PYQT_SIP_DIR = os.path.join(
    sipconfig.Configuration().default_sip_dir, 'PyQt4')

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

    def get_includes(self):

        incdirs = []
        for mod in ('QtCore', 'QtGui', 'QtXml'):
            if QT_IS_FRAMEWORK:
                incdirs.append(
                    os.path.join(QT_LIB_DIR, mod + '.framework', 'Headers') )
            else:
                incdirs.append( os.path.join(QT_INC_DIR, mod) )
        return incdirs

    def swig_sources (self, sources, extension=None):
        if not self.extensions:
            return

        # add directory of input files as include path
        indirs = list(set([os.path.dirname(x) for x in sources]))

        # Add the SIP and Qt include directories to the include path
        extension.include_dirs += [
            SIP_INC_DIR,
            QT_INC_DIR,
            ] + self.get_includes() + indirs

        # link against libraries
        if QT_IS_FRAMEWORK:
            extension.extra_link_args = [
                '-F', os.path.join(QT_LIB_DIR),
                '-framework', 'QtGui',
                '-framework', 'QtCore',
                '-framework', 'QtXml'
                ]
        elif sys.platform == 'win32':
            extension.libraries = ['QtGui4', 'QtCore4', 'QtXml4']
        else:
            extension.libraries = ['QtGui', 'QtCore', 'QtXml']
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
