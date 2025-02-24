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

from typing import TYPE_CHECKING, cast

from veusz import qtall as qt
from ..Mml import Mml
from ..NodeSpec import NodeSpec
from .._data import (
    g_draw_frames,
    g_min_font_point_size,
    g_script_size_multiplier,
)
from .._functions import (
    collectFontAttributes,
    interpretDepreciatedFontAttr,
    interpretMathSize,
    interpretMathVariant,
    interpretSpacing,
    mmlFindNodeSpec,
    rectToStr,
)

if TYPE_CHECKING:
    from veusz.helpers.qtmml.MmlDocument import MmlDocument

MmlAttributeMap = dict[str, str]


# noinspection PyPep8Naming
class MmlNode(Mml):
    def __init__(
        self,
        type_: Mml.NodeType,
        document: "MmlDocument",
        attribute_map: MmlAttributeMap,
    ):
        self.m_parent: MmlNode | None = None
        self.m_first_child: MmlNode | None = None
        self.m_next_sibling: MmlNode | None = None
        self.m_previous_sibling: MmlNode | None = None

        self.m_node_type: Mml.NodeType = type_
        self.m_document: MmlDocument = document
        self.m_attribute_map: MmlAttributeMap = attribute_map

        self.m_my_rect: qt.QRect = qt.QRect()
        self.m_parent_rect: qt.QRect = qt.QRect()
        self.m_rel_origin: qt.QPoint = qt.QPoint()
        self.m_stretched: bool = False

    def __del__(self) -> None:
        n: MmlNode | None = self.firstChild()
        while n is not None:
            tmp: MmlNode | None = n.nextSibling()
            del n
            n = tmp

    # Mml stuff
    def nodeType(self) -> Mml.NodeType:
        return self.m_node_type

    def toStr(self) -> str:
        spec: NodeSpec | None = mmlFindNodeSpec(self.nodeType())
        assert spec is not None

        return "{} {} mr={} pr={} dr={} ro=({}, {}) str={}".format(
            spec.type_str,
            hex(id(self)),
            rectToStr(self.m_my_rect),
            rectToStr(self.parentRect()),
            rectToStr(self.deviceRect()),
            self.m_rel_origin.x(),
            self.m_rel_origin.y(),
            self.m_stretched,
        )

    def setRelOrigin(self, rel_origin: qt.QPoint) -> None:
        self.m_rel_origin = rel_origin + qt.QPoint(-self.myRect().left(), 0)
        self.m_stretched = False

    def relOrigin(self) -> qt.QPoint:
        return self.m_rel_origin

    def stretchTo(self, rect: qt.QRect) -> None:
        self.m_parent_rect = rect
        self.m_stretched = True

    def isStretched(self) -> bool:
        return self.m_stretched

    def devicePoint(self, p: qt.QPoint) -> qt.QPoint:
        mr: qt.QRect = self.myRect()
        dr: qt.QRect = self.deviceRect()

        if self.m_stretched:
            return dr.topLeft() + qt.QPoint(
                (p.x() - mr.left()) * dr.width() / mr.width(),
                (p.y() - mr.top()) * dr.height() / mr.height(),
            )
        else:
            return dr.topLeft() + p - mr.topLeft()

    def myRect(self) -> qt.QRect:
        return self.m_my_rect

    def parentRect(self) -> qt.QRect:
        if self.m_stretched:
            return self.m_parent_rect

        mr: qt.QRect = self.myRect()
        ro: qt.QPoint = self.relOrigin()

        return qt.QRect(ro + mr.topLeft(), mr.size())

    def deviceRect(self) -> qt.QRect:
        if self.parent() is None:
            return qt.QRect(
                self.relOrigin() + self.myRect().topLeft(),
                self.myRect().size(),
            )

        # if not self.m_stretched:
        #     pdr: qt.QRect = self.parent().deviceRect()
        #     pmr: qt.QRect = self.parent().myRect()
        #     pr: qt.QRect = self.parentRect()
        #     mr: qt.QRect = self.myRect()
        #     return qt.QRect(
        #         pdr.left() + pr.left() - pmr.left(),
        #         pdr.top() + pr.top() - pmr.top(),
        #         mr.width(),
        #         mr.height(),
        #     )

        pdr: qt.QRect = self.parent().deviceRect()
        pr: qt.QRect = self.parentRect()
        pmr: qt.QRect = self.parent().myRect()

        scale_w: float = 0.0
        if pmr.width() != 0:
            scale_w = pdr.width() / pmr.width()
        scale_h: float = 0.0
        if pmr.height() != 0:
            scale_h = pdr.height() / pmr.height()

        return qt.QRect(
            pdr.left() + round((pr.left() - pmr.left()) * scale_w),
            pdr.top() + round((pr.top() - pmr.top()) * scale_h),
            round((pr.width() * scale_w)),
            round((pr.height() * scale_h)),
        )

    def updateMyRect(self) -> None:
        self.m_my_rect = self.symbolRect()
        child: MmlNode | None = self.firstChild()
        while child is not None:
            self.m_my_rect |= child.parentRect()
            child = child.nextSibling()

    def setMyRect(self, rect: qt.QRect) -> None:
        self.m_my_rect = rect

    def stretch(self) -> None:
        child: MmlNode | None = self.firstChild()
        while child is not None:
            child.stretch()
            child = child.nextSibling()

    def layout(self) -> None:
        self.m_parent_rect = qt.QRect()
        self.m_stretched = False
        self.m_rel_origin = qt.QPoint()

        child: MmlNode | None = self.firstChild()
        while child is not None:
            child.layout()
            child = child.nextSibling()

        self.layoutSymbol()

        self.updateMyRect()

        if self.parent() is None:
            self.m_rel_origin = qt.QPoint()

    def paint(self, p: qt.QPainter) -> None:
        if not self.myRect().isValid():
            return
        p.save()
        p.setViewport(self.deviceRect())
        p.setWindow(self.myRect())

        fg: qt.QColor = self.color()
        bg: qt.QColor = self.background()

        if bg.isValid():
            p.fillRect(self.myRect(), bg)
        if fg.isValid():
            p.setPen(self.color())

        child: MmlNode | None = self.firstChild()
        while child is not None:
            child.paint(p)
            child = child.nextSibling()

        self.paintSymbol(p)

        p.restore()

    def basePos(self) -> int:
        fm: qt.QFontMetrics = qt.QFontMetrics(self.font())
        return fm.strikeOutPos()

    def overlinePos(self) -> int:
        fm: qt.QFontMetrics = qt.QFontMetrics(self.font())
        return self.basePos() - fm.overlinePos()

    def underlinePos(self) -> int:
        fm: qt.QFontMetrics = qt.QFontMetrics(self.font())
        return self.basePos() + fm.underlinePos()

    def em(self) -> int:
        return qt.QFontMetrics(self.font()).boundingRect("m").width()

    def ex(self) -> int:
        return qt.QFontMetrics(self.font()).boundingRect("x").height()

    def explicitAttribute(self, name: str, def_: str = "") -> str:
        return self.m_attribute_map.get(name, def_)

    def inheritAttributeFromMrow(self, name: str, def_: str = "") -> str:
        p: MmlNode | None = self
        while p is not None:
            if p == self or p.nodeType() == Mml.NodeType.MstyleNode:
                value: str = p.explicitAttribute(name)
                if value:
                    return value
            p = p.parent()

        return def_

    def font(self) -> qt.QFont:
        from ..QtMmlWidget import QtMmlWidget

        fn: qt.QFont = qt.QFont(
            self.m_document.fontName(QtMmlWidget.MmlFont.NormalFont),
            self.m_document.baseFontPointSize(),
        )

        ps: int = fn.pointSize()
        sl: int = self.scriptlevel()
        ps *= g_script_size_multiplier**sl
        if ps < g_min_font_point_size:
            ps = g_min_font_point_size
        fn.setPointSize(ps)

        em: int = qt.QFontMetrics(fn).boundingRect("m").width()
        ex: int = qt.QFontMetrics(fn).boundingRect("x").height()

        font_attr: MmlAttributeMap = collectFontAttributes(self)

        value: str
        if "mathvariant" in font_attr:
            value = font_attr["mathvariant"]

            ok: bool
            mv: int
            mv, ok = interpretMathVariant(value)

            if ok:
                if mv & MmlNode.MathVariant.ScriptMV:
                    fn.setFamily(
                        self.m_document.fontName(QtMmlWidget.MmlFont.ScriptFont)
                    )

                if mv & MmlNode.MathVariant.FrakturMV:
                    fn.setFamily(
                        self.m_document.fontName(QtMmlWidget.MmlFont.FrakturFont)
                    )

                if mv & MmlNode.MathVariant.SansSerifMV:
                    fn.setFamily(
                        self.m_document.fontName(QtMmlWidget.MmlFont.SansSerifFont)
                    )

                if mv & MmlNode.MathVariant.MonospaceMV:
                    fn.setFamily(
                        self.m_document.fontName(QtMmlWidget.MmlFont.MonospaceFont)
                    )

                if mv & MmlNode.MathVariant.DoubleStruckMV:
                    fn.setFamily(
                        self.m_document.fontName(QtMmlWidget.MmlFont.DoublestruckFont)
                    )

                if mv & MmlNode.MathVariant.BoldMV:
                    fn.setBold(True)

                if mv & MmlNode.MathVariant.ItalicMV:
                    fn.setItalic(True)

        if "mathsize" in font_attr:
            value = font_attr["mathsize"]
            fn, _ = interpretMathSize(value, fn, em, ex)

        fn = interpretDepreciatedFontAttr(font_attr, fn, em, ex)

        if (
            self.nodeType() == Mml.NodeType.MiNode
            and "mathvariant" not in font_attr
            and "fontstyle" not in font_attr
        ):
            from .MmlMiNode import MmlMiNode

            if len(cast(MmlMiNode, self).text()) == 1:
                fn.setItalic(True)

        if self.nodeType() == Mml.NodeType.NoNode:
            fn.setItalic(False)
            fn.setBold(False)

        return fn

    def color(self) -> qt.QColor:
        # If we are child of <merror> return red
        p: MmlNode | None = self
        while p is not None:
            if p.nodeType() == Mml.NodeType.MerrorNode:
                return qt.QColor("red")
            p = p.parent()

        value_str: str = self.inheritAttributeFromMrow("mathcolor")
        if not value_str:
            value_str = self.inheritAttributeFromMrow("color")
        if not value_str:
            return qt.QColor()

        return qt.QColor(value_str)

    def background(self) -> qt.QColor:
        value_str: str = self.inheritAttributeFromMrow("mathbackground")
        if not value_str:
            value_str = self.inheritAttributeFromMrow("background")
        if not value_str:
            return qt.QColor()

        return qt.QColor(value_str)

    def scriptlevel(self, child: "MmlNode | None" = None) -> int:
        parent_sl: int
        p: MmlNode | None = self.parent()
        if p is None:
            parent_sl = 0
        else:
            parent_sl = p.scriptlevel(self)

        expl_sl_str: str = self.explicitAttribute("scriptlevel")
        if not expl_sl_str:
            return parent_sl

        expl_sl: int

        if expl_sl_str.startswith(("+", "-")):
            try:
                expl_sl = int(expl_sl_str)
            except ValueError:
                qt.qWarning(f"MmlNode::scriptlevel(): bad value {expl_sl_str}")
                return parent_sl
            else:
                return parent_sl + expl_sl

        try:
            expl_sl = int(expl_sl_str)
        except ValueError:
            pass
        else:
            return expl_sl

        if expl_sl_str == "+":
            return parent_sl + 1
        elif expl_sl_str == "-":
            return parent_sl - 1
        else:
            qt.qWarning(
                f'MmlNode::scriptlevel(): could not parse value: "{expl_sl_str}"'
            )
            return parent_sl

    # Node stuff

    def document(self) -> "MmlDocument":
        return self.m_document

    def parent(self) -> "MmlNode | None":
        return self.m_parent

    def firstChild(self) -> "MmlNode | None":
        return self.m_first_child

    def nextSibling(self) -> "MmlNode | None":
        return self.m_next_sibling

    def previousSibling(self) -> "MmlNode | None":
        return self.m_previous_sibling

    def lastSibling(self) -> "MmlNode | None":
        n: MmlNode | None = self
        while not n.isLastSibling():
            n = n.nextSibling()
        return n

    def firstSibling(self) -> "MmlNode | None":
        n: MmlNode | None = self
        while not n.isFirstSibling():
            n = n.previousSibling()
        return n

    def isLastSibling(self) -> bool:
        return self.m_next_sibling is None

    def isFirstSibling(self) -> bool:
        return self.m_previous_sibling is None

    def hasChildNodes(self) -> bool:
        return self.m_first_child is not None

    def layoutSymbol(self) -> None:
        # default behaves like an mrow

        # now lay them out in a neat row, aligning their origins to my origin
        w: int = 0
        child: MmlNode | None = self.firstChild()
        while child is not None:
            child.setRelOrigin(qt.QPoint(w, 0))
            w += child.parentRect().width() + 1
            child = child.nextSibling()

    def paintSymbol(self, p: qt.QPainter) -> None:
        if g_draw_frames and self.myRect().isValid():
            p.save()
            p.setPen(qt.Qt.GlobalColor.red)
            p.drawRect(self.m_my_rect)
            pen: qt.QPen = p.pen()
            pen.setStyle(qt.Qt.PenStyle.DotLine)
            p.setPen(pen)
            p.drawLine(self.myRect().left(), 0, self.myRect().right(), 0)
            p.restore()

    # noinspection PyMethodMayBeStatic
    def symbolRect(self) -> qt.QRect:
        return qt.QRect()

    # def parentWithExplicitAttribute(
    #     self,
    #     name: str,
    #     type_: Mml.NodeType = Mml.NodeType.NoNode,
    # ) -> "MmlNode | None":
    #     raise NotImplementedError

    def interpretSpacing(self, value: str) -> tuple[bool, int]:
        return interpretSpacing(value, self.em(), self.ex())
