"""
Copyright (c) 2009 Nokia Corporation and/or its subsidiary(-ies).
All rights reserved.
Contact: Nokia Corporation (qt-info@nokia.com)

This file is part of a Qt Solutions component.

Commercial Usage
Licensees holding valid Qt Commercial licenses may use this file in
accordance with the Qt Solutions Commercial License Agreement provided
with the Software or, alternatively, in accordance with the terms
contained in a written agreement between you and Nokia.

GNU Lesser General Public License Usage
Alternatively, this file may be used under the terms of the GNU Lesser
General Public License version 2.1 as published by the Free Software
Foundation and appearing in the file LICENSE.LGPL included in the
packaging of this file.  Please review the following information to
ensure the GNU Lesser General Public License version 2.1 requirements
will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.

In addition, as a special exception, Nokia gives you certain
additional rights. These rights are described in the Nokia Qt LGPL
Exception version 1.1, included in the file LGPL_EXCEPTION.txt in this
package.

GNU General Public License Usage
Alternatively, this file may be used under the terms of the GNU
General Public License version 3.0 as published by the Free Software
Foundation and appearing in the file LICENSE.GPL included in the
packaging of this file.  Please review the following information to
ensure the GNU General Public License version 3.0 requirements will be
met: http://www.gnu.org/copyleft/gpl.html.

Please note Third Party Software included with Qt Solutions may impose
additional restrictions and it is the user's responsibility to ensure
that they have met the licensing requirements of the GPL, LGPL, or Qt
Solutions Commercial license and the relevant license of the Third
Party Software they are using.

If you are unsure which license is appropriate for your use, please
contact Nokia at qt-info@nokia.com.
"""

from typing import TYPE_CHECKING

from .MmlTokenNode import MmlTokenNode
from ..Mml import Mml
from .._data import g_min_font_point_size
from .._functions import interpretColAlign, interpretRowAlign
from .... import qtall as qt

if TYPE_CHECKING:
    from ..MmlDocument import MmlDocument
    from .MmlNode import MmlNode

MmlAttributeMap = dict[str, str]


# noinspection PyPep8Naming
class MmlMtdNode(MmlTokenNode):
    def __init__(
        self,
        document: MmlDocument | None,
        attribute_map: MmlAttributeMap,
    ) -> None:
        super().__init__(Mml.NodeType.MtdNode, document, attribute_map)

        # added or subtracted to scriptlevel to make contents fit the cell
        self.m_scriptlevel_adjust: int = 0

    def scriptlevel(self, child: MmlNode | None = None) -> int:
        sl: int = super().scriptlevel()
        if child is not None and child is self.firstChild():
            return sl + self.m_scriptlevel_adjust
        else:
            return sl

    def setMyRect(self, rect: qt.QRect) -> None:
        super().setMyRect(rect)
        child: MmlNode | None = self.firstChild()
        if child is None:
            return

        if rect.width() < child.myRect().width():
            while (
                rect.width() < child.myRect().width()
                and child.font().pointSize() > g_min_font_point_size
            ):

                # qt.qWarning(
                #     f"MmlMtdNode::setMyRect(): "
                #     f"rect.width()={rect.width():d}, "
                #     f"child()->myRect().width={child.myRect().width():d} "
                #     f"sl={self.m_scriptlevel_adjust:d}"
                # )

                self.m_scriptlevel_adjust += 1
                child.layout()

            # qt.qWarning(
            #     f"MmlMtdNode::setMyRect(): "
            #     f"rect.width()={rect.width():d}, "
            #     f"child()->myRect().width={child.myRect().width():d} "
            #     f"sl={self.m_scriptlevel_adjust:d}"
            # )

        mr: qt.QRect = self.myRect()
        cmr: qt.QRect = child.myRect()

        child_rel_origin: qt.QPoint = qt.QPoint()

        match self.columnalign():
            case Mml.ColAlign.ColAlignLeft:
                child_rel_origin.setX(0)
            case Mml.ColAlign.ColAlignCenter:
                child_rel_origin.setX(mr.left() + (mr.width() - cmr.width()) // 2)
            case Mml.ColAlign.ColAlignRight:
                child_rel_origin.setX(mr.right() - cmr.width())

        match self.rowalign():
            case Mml.RowAlign.RowAlignTop:
                child_rel_origin.setY(mr.top() - cmr.top())
            case Mml.RowAlign.RowAlignCenter | Mml.RowAlign.RowAlignBaseline:
                child_rel_origin.setY(
                    mr.top() - cmr.top() + (mr.height() - cmr.height()) // 2
                )
            case Mml.RowAlign.RowAlignBottom:
                child_rel_origin.setY(mr.bottom() - cmr.bottom())
            case Mml.RowAlign.RowAlignAxis:
                child_rel_origin.setY(0)

        child.setRelOrigin(child_rel_origin)

    def colNum(self) -> int:
        syb: MmlNode | None = self.previousSibling()

        i: int = 0
        while syb is not None:
            i += 1
            syb = syb.previousSibling()

        return i

    def rowNum(self) -> int:
        row: MmlNode | None = self.parent().previousSibling()

        i: int = 0
        while row is not None:
            i += 1
            row = row.previousSibling()

        return i

    def columnalign(self) -> Mml.ColAlign:
        val: str = self.explicitAttribute("columnalign")
        if val:
            return interpretColAlign(val, 0)[0]

        node: MmlNode | None = self.parent()  # <mtr>
        if node is None:
            return Mml.ColAlign.ColAlignCenter

        colnum: int = self.colNum()
        val = node.explicitAttribute("columnalign")
        if val:
            return interpretColAlign(val, colnum)[0]

        node = node.parent()  # <mtable>
        if node is None:
            return Mml.ColAlign.ColAlignCenter

        val = node.explicitAttribute("columnalign")
        if val:
            return interpretColAlign(val, colnum)[0]

        return Mml.ColAlign.ColAlignCenter

    def rowalign(self) -> Mml.RowAlign:
        val: str = self.explicitAttribute("rowalign")
        if val:
            return interpretRowAlign(val, 0)[0]

        node: MmlNode | None = self.parent()  # <mtr>
        if node is None:
            return Mml.RowAlign.RowAlignAxis

        rownum: int = self.rowNum()
        val = node.explicitAttribute("rowalign")
        if val:
            return interpretRowAlign(val, rownum)[0]

        node = node.parent()  # <mtable>
        if node is None:
            return Mml.RowAlign.RowAlignAxis

        val = node.explicitAttribute("rowalign")
        if val:
            return interpretRowAlign(val, rownum)[0]

        return Mml.RowAlign.RowAlignAxis
