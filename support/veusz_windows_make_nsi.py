#!/usr/bin/env python3

# Take template NSI file and fill out sections
# Provide filename for output

import os
import sys

def getFileList(rootdir):
    """Get a list of directories and files within the root given."""
    outfiles = {}

    oldcwd = os.getcwd()
    os.chdir(rootdir)

    for dirname, childirs, childfiles in os.walk('.'):
        dirname = os.path.relpath(dirname) # strip ./
        files = []
        for child in childfiles:
            files.append(os.path.relpath(os.path.join(dirname, child)))
        outfiles[dirname] = sorted(files)

    os.chdir(oldcwd)

    return outfiles

def main():
    outfname = sys.argv[1]

    thisdir = os.path.dirname(os.path.abspath(__file__))
    srcdir = os.path.abspath(os.path.join(thisdir, '..'))
    distdir = os.path.join(srcdir, 'dist', 'veusz_main')
    intempl = os.path.join(thisdir, 'veusz_windows_setup_templ.nsi')

    # read input template
    print(f'Reading {intempl}')
    with open(intempl) as fin:
        templ = fin.read()

    # get version
    with open(os.path.join(srcdir, 'VERSION')) as fin:
        version = fin.readline().strip()

    # get files
    files = getFileList(distdir)
    copy_files = []
    delete_files = []
    for dirname in sorted(files):
        copy_files.append(f'  SetOutPath "$INSTDIR\\{dirname}"')
        delete_files.append(f'  RMDir "$INSTDIR\\{dirname}"')
        for fname in files[dirname]:
            copy_files.append(f'  File "${{PYINST_DIR}}\\{fname}"')
            delete_files.append(f'  Delete "$INSTDIR\\{fname}"')
        copy_files.append('')
        delete_files.append('')

    copy_files = '\n'.join(copy_files)
    delete_files = '\n'.join(delete_files[::-1])

    for search, replace in (
            ('VEUSZ_SRC_DIR', srcdir),
            ('VEUSZ_DIST_DIR', distdir),
            ('PRODUCT_VERSION', version),
            ('COPY_FILES', copy_files),
            ('DELETE_FILES', delete_files),
            ):
        templ = templ.replace(f'@@{search}@@', replace)

    # debug
    for line in templ.split('\n'):
        print(line)

    print(f'Writing {outfname}')
    with open(outfname, 'w') as fout:
        fout.write(templ)

if __name__ == '__main__':
    main()
