#!/bin/bash

# run pychecker on all script files

for f in `find . -name "*.py"`; do
    pychecker $f >> pychecker-out.txt
done
