#!/usr/bin/env bash

# generate the output manual files from the input docbook file
# DO NOT EDIT THE OUTPUT FILES!

infile=manual.xml

docbook2pdf $infile
docbook2html -u $infile
docbook2txt $infile

###################################################################
# $Id$
