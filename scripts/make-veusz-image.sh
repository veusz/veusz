#!/bin/bash

set -x


vzdir=/Users/jss/veusz
distdir=${vzdir}/dist
vzversion=`cat ${vzdir}/VERSION`

rm -rf ${distdir} ${vzdir}/build
export PYTHONPATH=/Users/jss

cd ${vzdir}
python setup.py py2app

/Users/jss/fixup_veusz_dist.sh

cp -f ${vzdir}/README ${distdir}/README.txt
cp -f ${vzdir}/ChangeLog ${distdir}/ChangeLog.txt
cp -f ${vzdir}/../Mac\ OS\ X\ README.txt ${distdir}/

mkdir -p ${distdir}/examples
rsync -ra ${vzdir}/examples/ ${distdir}/examples/ --delete \
    --exclude="*.pdf" --exclude="*.eps" --exclude="*.svg" --exclude="*.png"

mkdir -p ${distdir}/embedding/veusz
cp -f ${vzdir}/veusz/embed.py ${distdir}/embedding/veusz
cp -f ${vzdir}/veusz/__init__.py ${distdir}/embedding/veusz

outimage=veusz-${vzversion}-AppleOSX.dmg
rm -f ${outimage}
hdiutil create -srcfolder ${distdir} -volname "Veusz ${vzversion}" \
    -uid 99 -gid 99 \
    -format UDZO -imagekey zlib-level=9 ${outimage}
