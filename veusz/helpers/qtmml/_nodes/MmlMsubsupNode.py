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

from ..Mml import Mml

if TYPE_CHECKING:
    from ..MmlDocument import MmlDocument
from .MmlNode import MmlNode
from .... import qtall as qt

MmlAttributeMap = dict[str, str]


class MmlMsubsupNode(MmlNode):
    def __init__(
        self,
        document: MmlDocument | None,
        attribute_map: MmlAttributeMap,
    ) -> None:
        super().__init__(Mml.NodeType.MsubsupNode, document, attribute_map)

    def base(self) -> MmlNode:
        b: MmlNode | None = self.firstChild()
        assert b is not None
        return b

    def superscript(self) -> MmlNode:
        sup: MmlNode | None = self.subscript().nextSibling()
        assert sup is not None
        return sup

    def subscript(self) -> MmlNode:
        sub: MmlNode | None = self.base().nextSibling()
        assert sub is not None
        return sub

    def scriptlevel(self, child: MmlNode | None = None) -> int:
        sl: int = super().scriptlevel()
        sub: MmlNode = self.subscript()
        sup: MmlNode = self.superscript()

        if child in (sup, sub):
            return sl + 1
        else:
            return sl

    def layoutSymbol(self) -> None:
        b: MmlNode = self.base()
        sub: MmlNode = self.subscript()
        sup: MmlNode = self.superscript()

        b.setRelOrigin(qt.QPoint(-b.myRect().width(), 0))
        sub.setRelOrigin(qt.QPoint(0, b.myRect().bottom()))
        sup.setRelOrigin(qt.QPoint(0, b.myRect().top()))


# noinspection PyPep8Naming
