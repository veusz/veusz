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

from typing import TYPE_CHECKING

from .MmlDocument import MmlDocument
from ... import qtall as qt

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from .QtMmlWidget import QtMmlWidget


# noinspection PyPep8Naming
class QtMmlDocument:
    """
    :class QtMmlDocument

    :brief The QtMmlDocument class renders mathematical formulas written in MathML 2.0.

    This class provides a direct API to the rendering engine used by
    QtMmlWidget. It can be used to paint MathML inside other widgets.

    All methods work the same as the corresponding methods in QtMmlWidget.
    """

    def __init__(self) -> None:
        """Constructs an empty MML document."""
        self.m_doc: MmlDocument = MmlDocument()

    def __del__(self) -> None:
        """Destroys the MML document."""
        del self.m_doc

    def clear(self) -> None:
        """Clears the contents of this MML document."""
        self.m_doc.clear()

    def setContent(self, text: str) -> tuple[bool, str, int, int]:
        """
        Sets the MathML expression to be rendered. The expression is given
        in the string :param text. If the expression is successfully parsed,
        this method returns true; otherwise it returns false. If an error
        occurred :param errorMsg is set to a diagnostic message, while :param
        errorLine and :param errorColumn contain the location of the error.
        Any of :param errorMsg, :param errorLine and :param errorColumn may be 0,
        in which case they are not set.

        :param text should contain MathML 2.0 presentation markup elements enclosed
        in a <math> element.
        """
        return self.m_doc.setContent(text)

    def paint(self, p: qt.QPainter, pos: qt.QPoint) -> None:
        """Renders this MML document with the painter :param p at position :param pos."""
        self.m_doc.paint(p, pos)

    def size(self) -> qt.QSize:
        """Returns the size of this MML document, as rendered, in pixels."""
        return self.m_doc.size()

    def fontName(self, _type: "int | QtMmlWidget.MmlFont") -> str:
        """
        Returns the name of the font used to render the font :param type.

        See also: setFontName()  setBaseFontPointSize() baseFontPointSize() QtMmlWidget::MmlFont
        """
        return self.m_doc.fontName(_type)

    def setFontName(self, _type: "int | QtMmlWidget.MmlFont", name: str) -> None:
        """
        Sets the name of the font used to render the font :param type_ to :param name.

        See also: fontName() setBaseFontPointSize() baseFontPointSize() QtMmlWidget::MmlFont
        """
        self.m_doc.setFontName(_type, name)
        self.m_doc.layout()

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
        self.m_doc.setBaseFontPointSize(size)
        self.m_doc.layout()
