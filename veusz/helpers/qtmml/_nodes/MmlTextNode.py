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

import re
from typing import TYPE_CHECKING

from ..Mml import Mml

if TYPE_CHECKING:
    from ..MmlDocument import MmlDocument
from .MmlNode import MmlNode
from .... import qtall as qt

MmlAttributeMap = dict[str, str]


class MmlTextNode(MmlNode):
    def __init__(self, text: str, document: MmlDocument | None) -> None:
        super().__init__(Mml.NodeType.TextNode, document, MmlAttributeMap())
        # Trim whitespace from ends, but keep nbsp and thinsp
        text = re.sub(r"^[^\S\u00a0\u2009]+", "", text)
        text = re.sub(r"[^\S\u00a0\u2009]+$", "", text)

        if text in (
            "\u2062"  # &InvisibleTimes;
            "\u2063"  # &InvisibleComma;
            "\u2061"  # &ApplyFunction;
        ):
            text = ""

        self.m_text: str = text

    def toStr(self) -> str:
        return super().toStr() + ', text="' + self.m_text + '"'

    def text(self) -> str:
        return self.m_text

    # TextNodes are not xml elements, so they can't have attributes of
    # their own. Everything is taken from the parent.
    def font(self) -> qt.QFont:
        return self.parent().font()

    def scriptlevel(self, child: MmlNode | None = None) -> int:
        return self.parent().scriptlevel(child)

    def color(self) -> qt.QColor:
        return self.parent().color()

    def background(self) -> qt.QColor:
        return self.parent().background()

    def paintSymbol(self, p: qt.QPainter) -> None:
        super().paintSymbol(p)

        fn: qt.QFont = self.font()

        # fi: qt.QFontInfo = qt.QFontInfo(fn)
        # qt.qWarning(
        #     f'MmlTextNode::paintSymbol(): requested: {fn.family()}, used: {fi.family()}, '
        #     f'size={fi.pointSize():d}, italic={int(fi.italic()):d}, bold={int(fi.bold()):d}, '
        #     f'text="{self.m_text}" sl={self.scriptlevel():d}'
        # )

        fm: qt.QFontMetrics = qt.QFontMetrics(fn)

        p.save()
        p.setFont(fn)

        p.drawText(0, fm.strikeOutPos(), self.m_text)
        p.restore()

    def symbolRect(self) -> qt.QRect:
        fm: qt.QFontMetrics = qt.QFontMetrics(self.font())

        br: qt.QRect = fm.tightBoundingRect(self.m_text)
        br.translate(0, fm.strikeOutPos())

        return br
