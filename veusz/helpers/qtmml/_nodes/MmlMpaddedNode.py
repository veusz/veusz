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

from .MmlNode import MmlNode
from ..Mml import Mml
from .... import qtall as qt

if TYPE_CHECKING:
    from ..MmlDocument import MmlDocument

MmlAttributeMap = dict[str, str]


class MmlMpaddedNode(MmlNode):
    def __init__(
        self,
        document: MmlDocument | None,
        attribute_map: MmlAttributeMap,
    ) -> None:
        super().__init__(Mml.NodeType.MpaddedNode, document, attribute_map)

    def interpretSpacing(self, value: str, base_value: int) -> tuple[bool, int]:
        value = value.replace(" ", "")

        percent: bool = False

        # extract the sign
        idx = 0
        sign: str = ""
        if idx < len(value) and value[idx] in ("+", "-"):
            sign = value[idx]
            idx += 1

        # extract the factor
        factor_str: str = ""
        while idx < len(value) and (value[idx].isdigit() or value[idx] == "."):
            factor_str += value[idx]
            idx += 1

        # extract the % sign
        if idx < len(value) and value[idx] == "%":
            percent = True
            idx += 1

        # extract the pseudo-unit
        pseudo_unit: str = value[idx:]

        try:
            factor: float = float(factor_str)
        except ValueError:
            qt.qWarning(
                f'MmlMpaddedNode::interpretSpacing(): could not parse "{value}"'
            )
            return False, 0
        else:
            if factor < 0:
                qt.qWarning(
                    f'MmlMpaddedNode::interpretSpacing(): could not parse "{value}"'
                )
                return False, 0

            if percent:
                factor /= 100.0

            cr: qt.QRect
            if self.firstChild() is None:
                cr = qt.QRect()
            else:
                cr = self.firstChild().myRect()

            unit_size: int

            if not pseudo_unit:
                unit_size = base_value
            elif pseudo_unit == "width":
                unit_size = cr.width()
            elif pseudo_unit == "height":
                unit_size = -cr.top()
            elif pseudo_unit == "depth":
                unit_size = cr.bottom()
            else:
                unit_ok: bool
                unit_ok, unit_size = super().interpretSpacing("1" + pseudo_unit)
                if not unit_ok:
                    qt.qWarning(
                        f'MmlMpaddedNode::interpretSpacing(): could not parse "{value}"'
                    )
                    return False, 0

            if not sign:
                return True, int(factor * unit_size)
            elif sign == "+":
                return True, base_value + int(factor * unit_size)
            else:  # sign == "-"
                return True, base_value - int(factor * unit_size)

    def lspace(self) -> int:
        value: str = self.explicitAttribute("lspace")

        if not value:
            return 0

        ok: bool
        lspace: int
        ok, lspace = self.interpretSpacing(value, 0)

        if ok:
            return lspace

        return 0

    def width(self) -> int:
        child_width: int = 0
        if self.firstChild() is not None:
            child_width = self.firstChild().myRect().width()

        value: str = self.explicitAttribute("width")

        if not value:
            return child_width

        ok: bool
        w: int
        ok, w = self.interpretSpacing(value, child_width)

        if ok:
            return w

        return child_width

    def height(self) -> int:
        cr: qt.QRect
        if self.firstChild() is None:
            cr = qt.QRect()
        else:
            cr = self.firstChild().myRect()

        value: str = self.explicitAttribute("height")
        if not value:
            return -cr.top()

        ok: bool
        h: int
        ok, h = self.interpretSpacing(value, -cr.top())
        if ok:
            return h

        return -cr.top()

    def depth(self) -> int:
        cr: qt.QRect
        if self.firstChild() is None:
            cr = qt.QRect()
        else:
            cr = self.firstChild().myRect()

        value: str = self.explicitAttribute("depth")
        if not value:
            return cr.bottom()

        ok: bool
        h: int
        ok, h = self.interpretSpacing(value, cr.bottom())
        if ok:
            return h

        return cr.bottom()

    def layoutSymbol(self) -> None:
        child: MmlNode | None = self.firstChild()
        if child is None:
            return

        child.setRelOrigin(qt.QPoint(0, 0))

    def symbolRect(self) -> qt.QRect:
        return qt.QRect(
            -self.lspace(),
            -self.height(),
            self.lspace() + self.width(),
            self.height() + self.depth(),
        )
