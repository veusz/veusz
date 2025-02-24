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

from .Mml import Mml
from .NodeSpec import NodeSpec
from .QtMmlWidget import QtMmlWidget
from ._data import ChildSpec
from ._functions import (
    domToMmlNodeType,
    entityDeclarations,
    mmlCheckAttributes,
    mmlCheckChildType,
    mmlFindNodeSpec,
)
from ._nodes.MmlNode import MmlNode
from ... import qtall as qt

MmlAttributeMap = dict[str, str]


# noinspection PyPep8Naming
class MmlDocument(Mml):
    def __init__(self) -> None:
        self.m_root_node: MmlNode | None = None

        self.m_normal_font_name: str = ""
        self.m_fraktur_font_name: str = ""
        self.m_sans_serif_font_name: str = ""
        self.m_script_font_name: str = ""
        self.m_monospace_font_name: str = ""
        self.m_doublestruck_font_name: str = ""
        self.m_base_font_point_size: int = 0
        self.m_foreground_color: qt.QColor = qt.QColor()
        self.m_background_color: qt.QColor = qt.QColor()

    def __del__(self) -> None:
        self.clear()

    def clear(self) -> None:
        del self.m_root_node
        self.m_root_node = None

    def setContent(self, text: str) -> tuple[bool, str, int, int]:
        self.clear()

        prefix: str = '<?xml version="2.0"?>\n'
        prefix += entityDeclarations()

        prefix_lines: int = len(prefix.splitlines())

        dom = qt.QDomDocument()
        ok, errorMsg, errorLine, errorColumn = dom.setContent(prefix + text, False)
        if not ok:
            errorLine -= prefix_lines
            return False, errorMsg, errorLine, errorColumn

        # we don't have access to line info from now on
        errorLine = -1
        errorColumn = -1

        root_node, ok, errorMsg = self.domToMml(dom)
        if not ok:
            return False, errorMsg, errorLine, errorColumn

        if root_node == 0:
            errorMsg = "empty document"
            return False, errorMsg, errorLine, errorColumn

        self.insertChild(None, root_node)
        self.layout()

        return True, errorMsg, errorLine, errorColumn

    def paint(self, p: qt.QPainter, pos: qt.QPoint) -> None:
        if self.m_root_node is None:
            return

        # p.save()
        # p.setPen(qt.Qt.GlobalColor.blue)
        # p.drawLine(pos.x() - 5, pos.y(), pos.x() + 5, pos.y())
        # p.drawLine(pos.x(), pos.y() - 5, pos.x(), pos.y() + 5)
        # p.restore()

        mr: qt.QRect = self.m_root_node.myRect()
        self.m_root_node.setRelOrigin(pos - mr.topLeft())
        self.m_root_node.paint(p)

    def dump(self):
        if self.m_root_node is None:
            return

        def _dump(node: MmlNode | None, indent: str) -> None:
            if node is None:
                return

            qt.qWarning(f"{indent + node.toStr()}")

        _dump(self.m_root_node, "")

    def size(self) -> qt.QSize:
        if self.m_root_node is None:
            return qt.QSize(0, 0)
        return self.m_root_node.deviceRect().size()

    def layout(self) -> None:
        if self.m_root_node is None:
            return

        self.m_root_node.layout()
        self.m_root_node.stretch()
        # self.dump()

    def fontName(self, type_: "int | QtMmlWidget.MmlFont") -> str:
        match type_:
            case QtMmlWidget.MmlFont.NormalFont:
                return self.m_normal_font_name
            case QtMmlWidget.MmlFont.FrakturFont:
                return self.m_fraktur_font_name
            case QtMmlWidget.MmlFont.SansSerifFont:
                return self.m_sans_serif_font_name
            case QtMmlWidget.MmlFont.ScriptFont:
                return self.m_script_font_name
            case QtMmlWidget.MmlFont.MonospaceFont:
                return self.m_monospace_font_name
            case QtMmlWidget.MmlFont.DoublestruckFont:
                return self.m_doublestruck_font_name
            case _:
                raise ValueError(f"Invalid type: {type_}")

    def setFontName(self, type_: int | QtMmlWidget.MmlFont, name: str) -> None:
        match type_:
            case QtMmlWidget.MmlFont.NormalFont:
                self.m_normal_font_name = name
            case QtMmlWidget.MmlFont.FrakturFont:
                self.m_fraktur_font_name = name
            case QtMmlWidget.MmlFont.SansSerifFont:
                self.m_sans_serif_font_name = name
            case QtMmlWidget.MmlFont.ScriptFont:
                self.m_script_font_name = name
            case QtMmlWidget.MmlFont.MonospaceFont:
                self.m_monospace_font_name = name
            case QtMmlWidget.MmlFont.DoublestruckFont:
                self.m_doublestruck_font_name = name
            case _:
                raise ValueError(f"Invalid type: {type_}")

    def baseFontPointSize(self) -> int:
        return self.m_base_font_point_size

    def setBaseFontPointSize(self, size: int):
        self.m_base_font_point_size = size

    def foregroundColor(self) -> qt.QColor:
        return self.m_foreground_color

    def setForegroundColor(self, color: qt.QColor):
        self.m_foreground_color = color

    def backgroundColor(self) -> qt.QColor:
        return self.m_background_color

    def setBackgroundColor(self, color: qt.QColor):
        self.m_background_color = color

    def _dump(self, node: MmlNode | None, indent: str = "") -> None:
        if node is None:
            return

        qt.qWarning(f"{indent + node.toStr()}")

        child: MmlNode | None = node.firstChild()
        while child is not None:
            self._dump(child, indent + "  ")
            child = child.nextSibling()

    def insertChild(
        self,
        parent: MmlNode | None,
        new_node: MmlNode | None,
    ) -> tuple[bool, str]:
        if new_node is None:
            return True, ""

        assert (
            new_node.parent() is None
            and new_node.nextSibling() is None
            and new_node.previousSibling() is None
        )

        if parent is not None:
            ok: bool
            err: str
            ok, err = mmlCheckChildType(parent.nodeType(), new_node.nodeType())
            if not ok:
                return False, err

        n: MmlNode | None
        if parent is None:
            if self.m_root_node is None:
                self.m_root_node = new_node
            else:
                n = self.m_root_node.lastSibling()
                n.m_next_sibling = new_node
                new_node.m_previous_sibling = n
        else:
            new_node.m_parent = parent
            if parent.hasChildNodes():
                n = parent.firstChild().lastSibling()
                n.m_next_sibling = new_node
                new_node.m_previous_sibling = n
            else:
                parent.m_first_child = new_node

        return True, ""

    def domToMml(self, dom_node: qt.QDomNode) -> tuple[MmlNode | None, bool, str]:
        # create the node

        mml_type: Mml.NodeType = domToMmlNodeType(dom_node)

        if mml_type == Mml.NodeType.NoNode:
            return None, True, ""

        dom_attr: qt.QDomNamedNodeMap = dom_node.attributes()

        mml_attr: MmlAttributeMap = {}

        for i in range(len(dom_attr)):
            attr_node: qt.QDomNode = dom_attr.item(i)
            assert attr_node.nodeName()
            assert attr_node.nodeValue()
            mml_attr[attr_node.nodeName()] = attr_node.nodeValue()

        mml_value: str = ""
        if mml_type == Mml.NodeType.TextNode:
            mml_value = dom_node.nodeValue()
        mml_node: MmlNode | None
        err: str
        mml_node, err = self.createNode(mml_type, mml_attr, mml_value)
        if mml_node is None:
            return None, False, err

        # create the node's children according to the child_spec

        spec: NodeSpec | None = mmlFindNodeSpec(mml_type)
        dom_child_list: qt.QDomNodeList = dom_node.childNodes()
        child_cnt: int = dom_child_list.count()
        mml_child: MmlNode | None
        ok: bool
        err: str

        separator_list: str = ""
        if mml_type == Mml.NodeType.MfencedNode:
            separator_list = mml_node.explicitAttribute("separators", ",")

        match spec.child_spec:
            case ChildSpec.ChildIgnore:
                pass

            case ChildSpec.ImplicitMrow:

                if child_cnt > 0:
                    mml_child, ok, err = self.createImplicitMrowNode(dom_node)
                    if not ok:
                        del mml_node
                        return None, False, err

                    ok, err = self.insertChild(mml_node, mml_child)
                    if not ok:
                        del mml_node
                        del mml_child
                        return None, False, err

            case _ as child_spec:
                if child_spec != ChildSpec.ChildAny:
                    # exact ammount of children specified - check...
                    if spec.child_spec != child_cnt:
                        del mml_node
                        return (
                            None,
                            False,
                            f"element {spec.tag} requires exactly {spec.child_spec} arguments, got {child_cnt}",
                        )

                #  ...and continue just as in ChildAny

                if mml_type == Mml.NodeType.MfencedNode:
                    self.insertOperator(
                        mml_node, mml_node.explicitAttribute("open", "(")
                    )

                for i in range(child_cnt):
                    dom_child: qt.QDomNode = dom_child_list.item(i)

                    mml_child, ok, err = self.domToMml(dom_child)
                    if not ok:
                        del mml_node
                        return None, ok, err

                    if (
                        mml_type == Mml.NodeType.MtableNode
                        and mml_child.nodeType() != Mml.NodeType.MtrNode
                    ):
                        mtr_node: MmlNode | None
                        mtr_node, ok = self.createNode(Mml.NodeType.MtrNode, {}, "")
                        self.insertChild(mml_node, mtr_node)
                        ok, err = self.insertChild(mtr_node, mml_child)
                        if not ok:
                            del mml_node
                            del mml_child
                            return None, False, err
                    elif (
                        mml_type == Mml.NodeType.MtrNode
                        and mml_child.nodeType() != Mml.NodeType.MtdNode
                    ):
                        mtd_node: MmlNode | None
                        mtd_node, ok = self.createNode(Mml.NodeType.MtdNode, {}, "")
                        self.insertChild(mml_node, mtd_node)
                        ok, err = self.insertChild(mtd_node, mml_child)
                        if not ok:
                            del mml_node
                            del mml_child
                            return None, False, err
                    else:
                        ok, err = self.insertChild(mml_node, mml_child)
                        if not ok:
                            del mml_node
                            del mml_child
                            return None, False, err

                    if (
                        i < child_cnt - 1
                        and mml_type == Mml.NodeType.MfencedNode
                        and separator_list
                    ):
                        separator: str
                        if i >= len(separator_list):
                            separator = separator_list[-1]
                        else:
                            separator = separator_list[i]
                        self.insertOperator(mml_node, separator)

                if mml_type == Mml.NodeType.MfencedNode:
                    self.insertOperator(
                        mml_node, mml_node.explicitAttribute("close", ")")
                    )

        return mml_node, True, ""

    def createNode(
        self,
        type_: Mml.NodeType,
        mml_attr: dict[str, str],
        mml_value: str,
    ) -> tuple[MmlNode | None, str]:
        assert type_ != Mml.NodeType.NoNode

        ok: bool
        err: str
        ok, err = mmlCheckAttributes(type_, mml_attr)
        if not ok:
            return None, err

        mml_node: MmlNode | None
        match type_:
            case Mml.NodeType.MiNode:
                from ._nodes.MmlMiNode import MmlMiNode

                mml_node = MmlMiNode(self, mml_attr)
            case Mml.NodeType.MnNode:
                from ._nodes.MmlMnNode import MmlMnNode

                mml_node = MmlMnNode(self, mml_attr)
            case Mml.NodeType.MfracNode:
                from ._nodes.MmlMfracNode import MmlMfracNode

                mml_node = MmlMfracNode(self, mml_attr)
            case Mml.NodeType.MrowNode:
                from ._nodes.MmlMrowNode import MmlMrowNode

                mml_node = MmlMrowNode(self, mml_attr)
            case Mml.NodeType.MsqrtNode:
                from ._nodes.MmlMsqrtNode import MmlMsqrtNode

                mml_node = MmlMsqrtNode(self, mml_attr)
            case Mml.NodeType.MrootNode:
                from ._nodes.MmlMrootNode import MmlMrootNode

                mml_node = MmlMrootNode(self, mml_attr)
            case Mml.NodeType.MsupNode:
                from ._nodes.MmlMsupNode import MmlMsupNode

                mml_node = MmlMsupNode(self, mml_attr)
            case Mml.NodeType.MsubNode:
                from ._nodes.MmlMsubNode import MmlMsubNode

                mml_node = MmlMsubNode(self, mml_attr)
            case Mml.NodeType.MsubsupNode:
                from ._nodes.MmlMsubsupNode import MmlMsubsupNode

                mml_node = MmlMsubsupNode(self, mml_attr)
            case Mml.NodeType.MoNode:
                from ._nodes.MmlMoNode import MmlMoNode

                mml_node = MmlMoNode(self, mml_attr)
            case Mml.NodeType.MstyleNode:
                from ._nodes.MmlMstyleNode import MmlMstyleNode

                mml_node = MmlMstyleNode(self, mml_attr)
            case Mml.NodeType.TextNode:
                from ._nodes.MmlTextNode import MmlTextNode

                mml_node = MmlTextNode(mml_value, self)
            case Mml.NodeType.MphantomNode:
                from ._nodes.MmlMphantomNode import MmlMphantomNode

                mml_node = MmlMphantomNode(self, mml_attr)
            case Mml.NodeType.MfencedNode:
                from ._nodes.MmlMfencedNode import MmlMfencedNode

                mml_node = MmlMfencedNode(self, mml_attr)
            case Mml.NodeType.MtableNode:
                from ._nodes.MmlMtableNode import MmlMtableNode

                mml_node = MmlMtableNode(self, mml_attr)
            case Mml.NodeType.MtrNode:
                from ._nodes.MmlMtrNode import MmlMtrNode

                mml_node = MmlMtrNode(self, mml_attr)
            case Mml.NodeType.MtdNode:
                from ._nodes.MmlMtdNode import MmlMtdNode

                mml_node = MmlMtdNode(self, mml_attr)
            case Mml.NodeType.MoverNode:
                from ._nodes.MmlMoverNode import MmlMoverNode

                mml_node = MmlMoverNode(self, mml_attr)
            case Mml.NodeType.MunderNode:
                from ._nodes.MmlMunderNode import MmlMunderNode

                mml_node = MmlMunderNode(self, mml_attr)
            case Mml.NodeType.MunderoverNode:
                from ._nodes.MmlMunderoverNode import MmlMunderoverNode

                mml_node = MmlMunderoverNode(self, mml_attr)
            case Mml.NodeType.MalignMarkNode:
                from ._nodes.MmlMalignMarkNode import MmlMalignMarkNode

                mml_node = MmlMalignMarkNode(self, mml_attr)
            case Mml.NodeType.MerrorNode:
                from ._nodes.MmlMerrorNode import MmlMerrorNode

                mml_node = MmlMerrorNode(self, mml_attr)
            case Mml.NodeType.MtextNode:
                from ._nodes.MmlMtextNode import MmlMtextNode

                mml_node = MmlMtextNode(self, mml_attr)
            case Mml.NodeType.MpaddedNode:
                from ._nodes.MmlMpaddedNode import MmlMpaddedNode

                mml_node = MmlMpaddedNode(self, mml_attr)
            case Mml.NodeType.MspaceNode:
                from ._nodes.MmlMspaceNode import MmlMspaceNode

                mml_node = MmlMspaceNode(self, mml_attr)
            case Mml.NodeType.UnknownNode:
                from ._nodes.MmlUnknownNode import MmlUnknownNode

                mml_node = MmlUnknownNode(self, mml_attr)
            case Mml.NodeType.NoNode:
                mml_node = None
            case _:
                raise ValueError("Invalid node type:", type_)

        return mml_node, ""

    def createImplicitMrowNode(
        self, dom_node: qt.QDomNode
    ) -> tuple[MmlNode | None, bool, str]:
        dom_child_list: qt.QDomNodeList = dom_node.childNodes()
        child_cnt: int = dom_child_list.count()

        if child_cnt == 0:
            return None, True, ""

        if child_cnt == 1:
            return self.domToMml(dom_child_list.item(0))

        mml_node: MmlNode | None
        err: str
        ok: bool
        mml_node, err = self.createNode(Mml.NodeType.MrowNode, {}, "")
        assert (
            mml_node is not None
        )  # there is no reason in heaven or hell for this to fail

        for i in range(child_cnt):
            dom_child: qt.QDomNode = dom_child_list.item(i)

            mml_child: MmlNode | None
            mml_child, ok, err = self.domToMml(dom_child)
            if not ok:
                del mml_node
                return None, ok, err

            ok, err = self.insertChild(mml_node, mml_child)
            if not ok:
                del mml_node
                del mml_child
                return None, ok, err

        return mml_node, True, ""

    def insertOperator(self, node: MmlNode, text: str) -> None:
        text_node: MmlNode | None
        mo_node: MmlNode | None
        text_node, _ = self.createNode(Mml.NodeType.TextNode, {}, text)
        mo_node, _ = self.createNode(Mml.NodeType.MoNode, {}, "")

        ok: bool
        err: str
        ok, err = self.insertChild(node, mo_node)
        assert ok
        ok, err = self.insertChild(mo_node, text_node)
        assert ok
