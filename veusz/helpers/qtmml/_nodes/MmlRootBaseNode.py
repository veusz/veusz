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

from .._data import g_mroot_base_margin, g_radical_char
from ..Mml import Mml

if TYPE_CHECKING:
    from ..MmlDocument import MmlDocument
from .MmlNode import MmlNode
from .... import qtall as qt

MmlAttributeMap = dict[str, str]


class MmlRootBaseNode(MmlNode):
    def __init__(
        self,
        type_: Mml.NodeType,
        document: MmlDocument | None,
        attribute_map: MmlAttributeMap,
    ) -> None:
        super().__init__(type_, document, attribute_map)

    def base(self) -> MmlNode | None:
        node: MmlNode | None = self.firstChild()
        # assert node is not None
        return node

    def index(self) -> MmlNode | None:
        b: MmlNode | None = self.base()
        if b is None:
            return None
        return b.nextSibling()

    def scriptlevel(self, child: MmlNode | None = None) -> int:
        sl: int = super().scriptlevel()
        i: MmlNode | None = self.index()
        if child is not None and child is i:
            return sl + 1
        else:
            return sl

    def layoutSymbol(self) -> None:
        # BUG: unused code
        # b: MmlNode | None = self.base()
        # base_size: qt.QSize
        # if b is not None:
        #     b.setRelOrigin(qt.QPoint(0, 0))
        #     base_size = self.base().myRect().size()
        # else:
        #     base_size = qt.QSize(1, 1)

        i: MmlNode | None = self.index()
        if i is not None:
            tw: int = self.tailWidth()
            i_rect: qt.QRect = i.myRect()
            i.setRelOrigin(qt.QPoint(-tw // 2 - i_rect.width(), -i_rect.bottom() - 4))

    def paintSymbol(self, p: qt.QPainter) -> None:
        fn: qt.QFont = self.font()

        p.save()

        sr: qt.QRect = self.symbolRect()

        r: qt.QRect = qt.QRect(sr)
        r.moveTopLeft(self.devicePoint(sr.topLeft()))
        p.setViewport(r)
        p.setWindow(qt.QFontMetrics(fn).boundingRect(g_radical_char))
        p.setFont(self.font())
        p.drawText(0, 0, g_radical_char)

        p.restore()

        p.drawLine(sr.right(), sr.top(), self.myRect().right(), sr.top())

    def symbolRect(self) -> qt.QRect:
        b: MmlNode | None = self.base()
        base_rect: qt.QRect
        if b is None:
            base_rect = qt.QRect(0, 0, 1, 1)
        else:
            base_rect = self.base().myRect()

        margin: int = round(g_mroot_base_margin * base_rect.height())
        tw: int = self.tailWidth()

        return qt.QRect(
            -tw,
            base_rect.top() - margin,
            tw,
            base_rect.height() + 2 * margin,
        )

    def tailWidth(self) -> int:
        fm: qt.QFontMetrics = qt.QFontMetrics(self.font())
        return fm.boundingRect(g_radical_char).width()
