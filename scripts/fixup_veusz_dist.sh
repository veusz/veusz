#!/bin/bash

distdir=/Users/jss/veusz/dist/Veusz.app
qtdir=/usr/local/Trolltech/Qt-4.8.4

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
	    # get part of path after frameworks
	    relpath=$(echo $f | rev | cut -d/ -f1-4 | rev)
	    install_name_tool -change $f @executable_path/../Frameworks/${relpath} $x
	done
    done
    popd
done

find ${distdir} -name "*_debug*" -print0 | xargs -0 rm

# delete unused frameworks
for x in QtHelp.framework QtNetwork.framework QtOpenGL.framework QtSql.framework QtTest.framework QtXmlPatterns.framework QtDesigner.framework; do
    rm -rf ${distdir}/Contents/Frameworks/${x}
done

rm ${distir}/Contents/Frameworks/libQtCLucene*
