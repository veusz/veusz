#!/bin/sh

# helper script to make a Veusz installer image for MacOS

set -xe

outdir=dist
vzver=$(cat VERSION)

# make pyinstaller image
pyinstaller support/veusz_mac_pyinst.spec

# delete unneeded veusz directory
rm -rf $outdir/veusz

# copy various files
cp README.md ChangeLog support/MacOS_ReadMe.txt $outdir

# copy examples
mkdir $outdir/examples
rsync -ra examples/ $outdir/examples/ \
      --exclude='*.pdf' --exclude='*.eps' --exclude='*.svg' --exclude='*.png'

# copy embedding
mkdir -p $outdir/embedding/veusz
cp veusz/embed.py veusz/__init__.py $outdir/embedding/veusz

# make installer image
mkdir installer
hdiutil create -srcfolder "$outdir" -volname "Veusz ${vzver}" \
    -format UDZO -imagekey zlib-level=9 installer/veusz-AppleOSX.dmg
