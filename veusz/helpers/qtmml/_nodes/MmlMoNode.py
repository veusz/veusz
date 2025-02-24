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
from .._functions import interpretForm, mmlDictAttribute, mmlFindOperSpec

if TYPE_CHECKING:
    from ..MmlDocument import MmlDocument
from .MmlNode import MmlNode
from ..OperSpec import OperSpec
from .... import qtall as qt

MmlAttributeMap = dict[str, str]


class MmlMoNode(MmlTokenNode):
    def __init__(
        self,
        document: MmlDocument | None,
        attribute_map: MmlAttributeMap,
    ) -> None:
        super().__init__(Mml.NodeType.MoNode, document, attribute_map)

        self.m_oper_spec: OperSpec | None = None

    def dictionaryAttribute(self, name: str) -> str:
        p: MmlNode | None = self
        while p is not None:
            if p is self or p.nodeType() == Mml.NodeType.MstyleNode:
                expl_attr: str = p.explicitAttribute(name)
                if expl_attr:
                    return expl_attr
            p = p.parent()

        return mmlDictAttribute(name, self.m_oper_spec)

    def stretch(self) -> None:
        if self.parent() is None:
            return

        if self.m_oper_spec is None:
            return

        if (
            self.m_oper_spec.stretch_dir == OperSpec.StretchDir.HStretch
            and self.parent().nodeType() == Mml.NodeType.MrowNode
            and (self.nextSibling() is not None or self.previousSibling() is not None)
        ):
            return

        pmr: qt.QRect = self.parent().myRect()
        pr: qt.QRect = self.parentRect()

        match self.m_oper_spec.stretch_dir:
            case OperSpec.StretchDir.VStretch:
                self.stretchTo(qt.QRect(pr.left(), pmr.top(), pr.width(), pmr.height()))
            case OperSpec.StretchDir.HStretch:
                self.stretchTo(qt.QRect(pmr.left(), pr.top(), pmr.width(), pr.height()))
            case OperSpec.StretchDir.HVStretch:
                self.stretchTo(pmr)
            case OperSpec.StretchDir.NoStretch:
                pass
            case _ as stretch_dir:
                raise ValueError("Invalid stretch direction:", stretch_dir)

    def lspace(self) -> int:
        assert self.m_oper_spec is not None
        if (
            self.parent() is None
            or self.parent().nodeType()
            not in (
                Mml.NodeType.MrowNode,
                Mml.NodeType.MfencedNode,
                Mml.NodeType.UnknownNode,
            )
            or (self.previousSibling() is None and self.nextSibling() is None)
        ):
            return 0
        else:
            return self.interpretSpacing(self.dictionaryAttribute("lspace"))[1]

    def rspace(self) -> int:
        assert self.m_oper_spec is not None
        if (
            self.parent() is None
            or self.parent().nodeType()
            not in (
                Mml.NodeType.MrowNode,
                Mml.NodeType.MfencedNode,
                Mml.NodeType.UnknownNode,
            )
            or (self.previousSibling() is None and self.nextSibling() is None)
        ):
            return 0
        else:
            return self.interpretSpacing(self.dictionaryAttribute("rspace"))[1]

    def toStr(self) -> str:
        return super().toStr() + f" form={self.form()}"

    def layoutSymbol(self) -> None:
        child: MmlNode | None = self.firstChild()

        if child is None:
            return

        child.setRelOrigin(qt.QPoint(0, 0))

        if self.m_oper_spec is None:
            self.m_oper_spec = mmlFindOperSpec(self.text(), self.form())

    def symbolRect(self) -> qt.QRect:
        child: MmlNode | None = self.firstChild()

        if child is None:
            return qt.QRect()

        cmr: qt.QRect = child.m_my_rect

        return qt.QRect(
            -self.lspace(),
            cmr.top(),
            cmr.width() + self.lspace() + self.rspace(),
            cmr.height(),
        )

    def form(self) -> Mml.FormType:
        value_str: str = self.inheritAttributeFromMrow("form")
        if value_str:
            ok: bool
            value: Mml.FormType
            value, ok = interpretForm(value_str)
            if ok:
                return value
            else:
                qt.qWarning(f"Could not convert {value_str} to form")

        # Default heuristic.
        if self.firstSibling() is self and self.lastSibling() is not self:
            return Mml.FormType.PrefixForm
        elif self.lastSibling() is self and self.firstSibling() is not self:
            return Mml.FormType.PostfixForm
        else:
            return Mml.FormType.InfixForm
