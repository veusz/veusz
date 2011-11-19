#!/bin/bash

distdir=/Users/apple/veusz-git/veusz/dist/Veusz.app
qtdir=/Developer/Applications/Qt

cat > ${distdir}/Contents/Resources/qt.conf <<EOF
[paths]
Plugins=plugins
EOF

mkdir -p ${distdir}/Contents/plugins
cd ${distdir}/Contents/plugins
for plugin in iconengines imageformats; do
    cp -R ${qtdir}/plugins/${plugin} .

    pushd $plugin
    for x in *.dylib; do
	frames=`otool -L $x | grep Qt | awk '{print $1}'`
	for f in $frames; do
	    install_name_tool -change $f @executable_path/../Frameworks/$f $x
	done
    done
    popd
done

