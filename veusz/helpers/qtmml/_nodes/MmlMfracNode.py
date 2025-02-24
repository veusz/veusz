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

from .._data import g_mfrac_spacing
from .._functions import zeroLineThickness
from ..Mml import Mml

if TYPE_CHECKING:
    from ..MmlDocument import MmlDocument
from .MmlNode import MmlNode
from .... import qtall as qt

MmlAttributeMap = dict[str, str]


class MmlMfracNode(MmlNode):
    def __init__(
        self,
        document: MmlDocument | None,
        attribute_map: MmlAttributeMap,
    ) -> None:
        super().__init__(Mml.NodeType.MfracNode, document, attribute_map)

    def numerator(self) -> MmlNode:
        node: MmlNode | None = self.firstChild()
        assert node is not None
        return node

    def denominator(self) -> MmlNode:
        node: MmlNode | None = self.numerator().nextSibling()
        assert node is not None
        return node

    def layoutSymbol(self) -> None:
        num: MmlNode | None = self.numerator()
        denom: MmlNode | None = self.denominator()

        num_rect: qt.QRect = num.myRect()
        denom_rect: qt.QRect = denom.myRect()

        spacing: int = int(g_mfrac_spacing * (num_rect.height() + denom_rect.height()))

        num.setRelOrigin(
            qt.QPoint(
                -num_rect.width() // 2,
                -spacing - num_rect.bottom(),
            )
        )
        denom.setRelOrigin(
            qt.QPoint(
                -denom_rect.width() // 2,
                spacing - denom_rect.top(),
            )
        )

        # TODO: check the `num` and `denom` are changed in the caller function

    def paintSymbol(self, p: qt.QPainter) -> None:
        linethickness_str: str = self.inheritAttributeFromMrow("linethickness", "1")

        # InterpretSpacing returns an int, which might be 0 even if the thickness
        # is > 0, though very, very small. That's ok, because the painter then paints
        # a line of thickness 1. However, we have to run this check if the line
        # thickness really is zero
        if not zeroLineThickness(linethickness_str):
            ok, linethickness = self.interpretSpacing(linethickness_str)
            if not ok:
                linethickness = 1
            p.save()
            pen: qt.QPen = p.pen()
            pen.setWidth(linethickness)
            p.setPen(pen)
            s: qt.QSize = self.myRect().size()
            p.drawLine(-s.width() // 2, 0, s.width() // 2, 0)
            p.restore()

    def symbolRect(self) -> qt.QRect:
        num_width: int = self.numerator().myRect().width()
        denom_width: int = self.denominator().myRect().width()
        my_width: int = max(num_width, denom_width) + 4

        return qt.QRect(-my_width // 2, 0, my_width, 1)
