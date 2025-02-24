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

MmlAttributeMap = dict[str, str]


class MmlSubsupBaseNode(MmlNode):
    def __init__(
        self,
        type_: Mml.NodeType,
        document: MmlDocument | None,
        attribute_map: MmlAttributeMap,
    ) -> None:
        super().__init__(type_, document, attribute_map)

    def base(self) -> MmlNode:
        b: MmlNode | None = self.firstChild()
        assert b is not None
        return b

    def sscript(self) -> MmlNode:
        s: MmlNode | None = self.base().nextSibling()
        assert s is not None
        return s

    def scriptlevel(self, child: MmlNode | None = None) -> int:
        sl: int = super().scriptlevel()

        s: MmlNode = self.sscript()
        if child is not None and child is s:
            return sl + 1
        else:
            return sl
