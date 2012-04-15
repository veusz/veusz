# Subclasses disutils.command.build_ext,
# replacing it with a SIP version that compiles .sip -> .cpp
# before calling the original build_ext command.
# Written by Giovanni Bajo <rasky at develer dot com>
# Based on Pyrex.Distutils, written by Graham Fawcett and Darrel Gallion.

import distutils.command.build_ext
from distutils.dep_util import newer, newer_group
import os
import sys

import PyQt4.pyqtconfig

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
        for L in file(sbf):
            key, value = L.split('=', 1)
            if key.strip() == 'sources':
                out = []
                for o in value.split():
                    out.append(os.path.join(self.build_temp, o))
                return out

        raise RuntimeError, 'cannot parse SIP-generated "%s"' % sbf

    def _find_sip(self):
        cfg = PyQt4.pyqtconfig.Configuration()
        return cfg.sip_bin

    def _sip_inc_dir(self):
        cfg = PyQt4.pyqtconfig.Configuration()
        return cfg.sip_inc_dir

    def get_includes(self, cfg):
        incdirs = []
        for mod in ('QtCore', 'QtGui', 'QtXml'):
            if cfg.qt_framework:
                incdirs.append( os.path.join(cfg.qt_lib_dir,
                                             mod + '.framework', 'Headers') )
            else:
                incdirs.append( os.path.join(cfg.qt_inc_dir, mod) )
        return incdirs

    def swig_sources (self, sources, extension=None):
        if not self.extensions:
            return

        cfg = PyQt4.pyqtconfig.Configuration()

        # add directory of input files as include path
        indirs = list(set([os.path.dirname(x) for x in sources]))

        # Add the SIP and Qt include directories to the include path
        extension.include_dirs += [
            cfg.sip_inc_dir,
            cfg.qt_inc_dir,
            ] + self.get_includes(cfg) + indirs

        # link against libraries
        if cfg.qt_framework:
            extension.extra_link_args = ['-framework', 'QtGui',
                                         '-framework', 'QtCore',
                                         '-framework', 'QtXml']
        elif sys.platform == 'win32':
            extension.libraries = ['QtGui4', 'QtCore4', 'QtXml4']
        else:
            extension.libraries = ['QtGui', 'QtCore', 'QtXml']
        extension.library_dirs = [cfg.qt_lib_dir]

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

        sip_bin = self._find_sip()

        for sip in sip_sources:
            # Use the sbf file as dependency check
            sipbasename = os.path.basename(sip)
            sbf = os.path.join(self.build_temp,
                               replace_suffix(sipbasename, '.sbf'))
            if newer_group([sip]+depends, sbf) or self.force:
                self._sip_compile(sip_bin, sip, sbf)
            out = self._get_sip_output_list(sbf)
            generated_sources.extend(out)

        return generated_sources + other_sources

    def _sip_compile(self, sip_bin, source, sbf):
        cfg = PyQt4.pyqtconfig.Configuration()
        self.spawn([sip_bin,
                    '-c', self.build_temp,
                    ] + cfg.pyqt_sip_flags.split() + [
                    '-I', cfg.pyqt_sip_dir,
                    '-b', sbf,
                    source])
