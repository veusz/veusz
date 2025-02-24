# ***************************************************************************
#
# Copyright (c) 2009 Nokia Corporation and/or its subsidiary(-ies).
# All rights reserved.
# Contact: Nokia Corporation (qt-info@nokia.com)
#
# This file is part of a Qt Solutions component.
#
# Commercial Usage
# Licensees holding valid Qt Commercial licenses may use this file in
# accordance with the Qt Solutions Commercial License Agreement provided
# with the Software or, alternatively, in accordance with the terms
# contained in a written agreement between you and Nokia.
#
# GNU Lesser General Public License Usage
# Alternatively, this file may be used under the terms of the GNU Lesser
# General Public License version 2.1 as published by the Free Software
# Foundation and appearing in the file LICENSE.LGPL included in the
# packaging of this file.  Please review the following information to
# ensure the GNU Lesser General Public License version 2.1 requirements
# will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
#
# In addition, as a special exception, Nokia gives you certain
# additional rights. These rights are described in the Nokia Qt LGPL
# Exception version 1.1, included in the file LGPL_EXCEPTION.txt in this
# package.
#
# GNU General Public License Usage
# Alternatively, this file may be used under the terms of the GNU
# General Public License version 3.0 as published by the Free Software
# Foundation and appearing in the file LICENSE.GPL included in the
# packaging of this file.  Please review the following information to
# ensure the GNU General Public License version 3.0 requirements will be
# met: http://www.gnu.org/copyleft/gpl.html.
#
# Please note Third Party Software included with Qt Solutions may impose
# additional restrictions and it is the user's responsibility to ensure
# that they have met the licensing requirements of the GPL, LGPL, or Qt
# Solutions Commercial license and the relevant license of the Third
# Party Software they are using.
#
# If you are unsure which license is appropriate for your use, please
# contact Nokia at qt-info@nokia.com.
#
# ***************************************************************************

# ruff: noqa: N802

import enum
from typing import TYPE_CHECKING

from ._data import g_min_font_point_size

if TYPE_CHECKING:
    from .MmlDocument import MmlDocument
from ... import qtall as qt


# noinspection PyPep8Naming
class QtMmlWidget(qt.QFrame):
    class MmlFont(enum.IntEnum):
        NormalFont = enum.auto()
        FrakturFont = enum.auto()
        SansSerifFont = enum.auto()
        ScriptFont = enum.auto()
        MonospaceFont = enum.auto()
        DoublestruckFont = enum.auto()

    def __init__(self, parent: qt.QWidget | None = None) -> None:
        super().__init__(parent)
        self.m_doc: MmlDocument | None = None
        self.g_draw_frames: bool = False

    def __del__(self) -> None:
        """Destructs a QtMmlWidget object."""
        del self.m_doc

    def fontName(self, type_: int | MmlFont) -> str:
        """
        Returns the name of the font used to render the font :param type_.

        See also: setFontName()  setBaseFontPointSize() baseFontPointSize() QtMmlWidget::MmlFont
        """
        return self.m_doc.fontName(type_)

    def setFontName(self, type_: int | MmlFont, name: str) -> None:
        """
        Sets the name of the font used to render the font :param type_ to :param name.

        See also: fontName() setBaseFontPointSize() baseFontPointSize() QtMmlWidget::MmlFont
        """
        self.m_doc.setFontName(type_, name)
        self.m_doc.layout()
        self.update()

    def baseFontPointSize(self) -> int:
        """
        Returns the point size of the font used to render expressions
        whose scriptlevel is 0.

        See also: setBaseFontPointSize() fontName() setFontName()
        """
        return self.m_doc.baseFontPointSize()

    def setBaseFontPointSize(self, size: int) -> None:
        """
        Sets the point :param size of the font used to render expressions
        whose scriptlevel is 0.

        See also: baseFontPointSize() fontName() setFontName()
        """
        if size < g_min_font_point_size:
            self.m_doc.setBaseFontPointSize(size)
            self.m_doc.layout()
            self.update()

    def setContent(self, text: str) -> tuple[bool, str, int, int]:
        """
        Sets the MathML expression to be rendered. The expression is given
        in the string :param text. If the expression is successfully parsed,
        this method returns true; otherwise it returns false. If an error
        occured :param errorMsg is set to a diagnostic message, while :param
        errorLine and :param errorColumn contain the location of the error.
        Any of :param errorMsg, :param errorLine and :param errorColumn may be 0,
        in which case they are not set.

        :param text should contain MathML 2.0 presentation markup elements enclosed
        in a <math> element.
        """
        result, errorMsg, errorLine, errorColumn = self.m_doc.setContent(text)
        if result:
            self.update()
        return result, errorMsg, errorLine, errorColumn

    def dump(self) -> None:
        """internal"""
        self.m_doc.dump()

    def sizeHint(self) -> qt.QSize:
        """Returns the size of the formula in pixels."""
        size: qt.QSize = self.m_doc.size()
        if size.isNull():
            return qt.QSize(100, 50)
        return size

    def setDrawFrames(self, b: bool) -> None:
        """
        If :param b is true, draws a red bounding rectangle around each
        expression; if :param b is false, no such rectangle is drawn.
        This is mostly useful for debugging MathML expressions.

        See also: drawFrames()
        """
        self.g_draw_frames = b
        self.update()

    def drawFrames(self) -> bool:
        """
        Returns true if each expression should be drawn with a red
        bounding rectangle; otherwise returns false.
        This is mostly useful for debugging MathML expressions.

        See also: setDrawFrames()
        """
        return self.g_draw_frames

    def clear(self) -> None:
        """Clears the contents of this widget."""
        self.m_doc.clear()

    def paintEvent(self, e: qt.QPaintEvent) -> None:
        """internal"""
        super().paintEvent(e)
        p: qt.QPainter = qt.QPainter(self)
        if e.rect().intersects(self.contentsRect()):
            p.setClipRegion(e.region().intersected(self.contentsRect()))

        s: qt.QSize = self.m_doc.size()
        x: int = (self.width() - s.width()) // 2
        y: int = (self.height() - s.height()) // 2
        self.m_doc.paint(p, qt.QPoint(x, y))
