#    Copyright (C) 2010 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

"""
Document veusz widget types and settings
Creates an xml file designed to be processed into a web page using xsl
"""

from __future__ import division, print_function
import re

import veusz.widgets as widgets
import veusz.document as document
import veusz.setting as setting

#import cElementTree as ET
#import elementtree.ElementTree as ET

import xml.etree.ElementTree as ET

def processSetting(parent, setn):
    """Convert setting to xml element."""
    setnxml = ET.SubElement(parent, "setting")

    ET.SubElement(setnxml, "apiname").text = setn.name
    ET.SubElement(setnxml, "displayname").text = setn.usertext
    ET.SubElement(setnxml, "description").text = setn.descr
    ET.SubElement(setnxml, "formatting").text = str(setn.formatting)

    typename = str(type(setn))
    typename = re.match(r"^<class '(.*)'>$",
                        typename).group(1)
    typename = typename.split('.')[-1]
    ET.SubElement(setnxml, "type").text = typename

    # show list of possible choices if there is one
    if isinstance(setn, setting.Choice):
        for choice in setn.vallist:
            ET.SubElement(setnxml, "choice").text = choice

    if not isinstance(setn.default, setting.ReferenceBase):
        ET.SubElement(setnxml, "default").text = setn.toText()
    else:
        ET.SubElement(setnxml, "default").text = "to reference"

def processSettings(parent, setns):
    """Convert setting to xml element."""
    setnsxml = ET.SubElement(parent, "settings")

    ET.SubElement(setnsxml, "apiname").text = setns.name
    ET.SubElement(setnsxml, "displayname").text = setns.usertext
    ET.SubElement(setnsxml, "description").text = setns.descr
    for s in setns.getSettingList():
        processSetting(setnsxml, s)

def processWidgetType(root, name):
    """Produce documentation for a widget type."""
    widgetxml = ET.SubElement(root, "widget")

    klass = document.thefactory.getWidgetClass(name)
    print(klass)

    ET.SubElement(widgetxml, "apiname").text = name

    try:
        ET.SubElement(widgetxml, "description").text = klass.description
    except AttributeError:
        pass

    for parent in [k for k in klass.allowedParentTypes() if k is not None]:
        ET.SubElement(widgetxml, "allowedparent").text = parent.typename

    ET.SubElement(widgetxml, "usercreation").text = str(klass.allowusercreation)

    thesettings = setting.Settings('')
    klass.addSettings(thesettings)

    #for s in thesettings.getSettingList():
    processSettings(widgetxml, thesettings)
    for s in thesettings.getSettingsList():
        processSettings(widgetxml, s)

def indent(elem, level=0):
    """Indent output, from elementtree manual."""
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def addXSL(filename):
    f = open(filename)
    l = f.readlines()
    f.close()
    l.insert(1, '<?xml-stylesheet type="text/xsl" href="widget_doc.xsl"?>\n')
    f = open(filename, 'w')
    f.writelines(l)
    f.close()

def main():
    widgettypes = document.thefactory.listWidgets()

    root = ET.Element("widgets")
    for wt in widgettypes:
        processWidgetType(root, wt)

    tree = ET.ElementTree(root)
    indent(root)
    tree.write('widget_doc.xml', encoding="utf8")
    addXSL('widget_doc.xml')

if __name__ == '__main__':
    main()
