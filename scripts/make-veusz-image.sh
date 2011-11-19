#!/bin/bash

set -x


vzdir=/Users/apple/veusz-git/veusz
distdir=${vzdir}/dist
vzversion=`cat ${vzdir}/VERSION`

rm -rf ${distdir} ${vzdir}/build
export PYTHONPATH=/Users/veusz-git

cd ${vzdir}
python setup.py py2app

/Users/apple/fixup_veusz_dist.sh

cp -f ${vzdir}/README ${distdir}/README.txt
cp -f ${vzdir}/ChangeLog ${distdir}/ChangeLog.txt
cp -f ${vzdir}/Mac\ OS\ X\ README.txt ${distdir}/

mkdir -p ${distdir}/examples
rsync -ra ${vzdir}/examples/ ${distdir}/examples/ --delete \
    --exclude="*.pdf" --exclude="*.eps" --exclude="*.svg" --exclude="*.png"

mkdir -p ${distdir}/embedding
cp -f ${vzdir}/embed.py ${distdir}/embedding
cp -f ${vzdir}/__init__.py ${distdir}/embedding

outimage=veusz-${vzversion}-AppleOSX.dmg
rm -f ${outimage}
hdiutil create -srcfolder ${distdir} -volname "Veusz ${vzversion}" \
    -uid 99 -gid 99 \
    -format UDZO -imagekey zlib-level=9 ${outimage}
