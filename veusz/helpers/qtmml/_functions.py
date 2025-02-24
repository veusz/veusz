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

from html import unescape
from typing import Collection, Iterator, overload, TYPE_CHECKING

from ._data import (
    g_node_spec_data,
    g_oper_spec_data,
    g_oper_spec_names,
    g_xml_entity_data,
)
from .EntitySpec import EntitySpec
from .Mml import Mml
from .NodeSpec import NodeSpec
from .OperSpec import OperSpec
from ... import qtall as qt

if TYPE_CHECKING:
    from ._nodes import MmlNode

MmlAttributeMap = dict[str, str]


# noinspection PyPep8Naming
def entityDeclarations():
    result: str = "<!DOCTYPE math [\n"

    for key, value in g_xml_entity_data:
        result += f'\t<!ENTITY {key} "{value}">\n'

    result += "]>\n"

    return result


# noinspection PyPep8Naming
def interpretSpacing(value: str, em: int, ex: int) -> tuple[bool, int]:
    if value == "thin":
        return True, 1

    if value == "medium":
        return True, 2

    if value == "thick":
        return True, 3

    g_h_spacing_data: dict[str, float] = {
        "veryverythinmathspace": 0.055555,
        "verythinmathspace": 0.111111,
        "thinmathspace": 0.166667,
        "mediummathspace": 0.222222,
        "thickmathspace": 0.277778,
        "verythickmathspace": 0.333333,
        "veryverythickmathspace": 0.388889,
    }

    if value in g_h_spacing_data:
        return True, int(em * g_h_spacing_data[value])

    factor: float
    dw: qt.QScreen | None

    if value.endswith("em"):
        value = value[:-2]
        try:
            factor: float = float(value)
        except ValueError:
            qt.qWarning(f'interpretSpacing(): could not parse "{value}em"')
            return False, 0
        else:
            if factor >= 0.0:
                return True, int(em * factor)
            return False, 0

    if value.endswith("ex"):
        value = value[:-2]
        try:
            factor = float(value)
        except ValueError:
            qt.qWarning(f'interpretSpacing(): could not parse "{value}ex"')
            return False, 0
        else:
            if factor >= 0.0:
                return True, int(ex * factor)
            return False, 0

    if value.endswith("cm"):
        value = value[:-2]
        try:
            factor = float(value)
        except ValueError:
            qt.qWarning(f'interpretSpacing(): could not parse "{value}cm"')
            return False, 0
        else:
            if factor >= 0.0:
                dw = qt.QApplication.primaryScreen()
                assert dw is not None
                assert not dw.physicalSize().isNull()
                assert not dw.virtualSize().isNull()
                return True, int(
                    factor
                    * dw.virtualSize().width()
                    / (0.1 * dw.physicalSize().width())
                )
            return False, 0

    if value.endswith("mm"):
        value = value[:-2]
        try:
            factor = float(value)
        except ValueError:
            qt.qWarning(f'interpretSpacing(): could not parse "{value}mm"')
            return False, 0
        else:
            if factor >= 0.0:
                dw = qt.QApplication.primaryScreen()
                assert dw is not None
                assert not dw.physicalSize().isNull()
                assert not dw.virtualSize().isNull()
                return True, int(
                    factor * dw.virtualSize().width() / dw.physicalSize().width()
                )
            return False, 0

    if value.endswith("in"):
        value = value[:-2]
        try:
            factor = float(value)
        except ValueError:
            qt.qWarning(f'interpretSpacing(): could not parse "{value}in"')
            return False, 0
        else:
            if factor >= 0.0:
                dw = qt.QApplication.primaryScreen()
                assert dw is not None
                assert not dw.physicalSize().isNull()
                assert not dw.virtualSize().isNull()
                return True, int(
                    factor
                    * dw.virtualSize().width()
                    / (25.4 * dw.physicalSize().width())
                )
            return False, 0

    if value.endswith("px"):
        value = value[:-2]
    # what ends with “px” and all the rest
    try:
        factor = float(value)
    except ValueError:
        qt.qWarning(f'interpretSpacing(): could not parse "{value}px"')
        return False, 0
    else:
        if factor >= 0.0:
            return True, int(factor)
        return False, 0


# noinspection PyPep8Naming
def interpretListAttr(value_list: str, idx: int, def_: str) -> str:
    items: list[str] = value_list.split()

    if not items:
        return def_

    if len(items) <= idx:
        return items[-1]
    else:
        return items[idx]


# noinspection PyPep8Naming
def interpretFrameType(value_list: str, idx: int) -> tuple[bool, Mml.FrameType]:
    value: str = interpretListAttr(value_list, idx, "none")

    if value == "none":
        return True, Mml.FrameType.FrameNone
    if value == "solid":
        return True, Mml.FrameType.FrameSolid
    if value == "dashed":
        return True, Mml.FrameType.FrameDashed

    qt.qWarning(f'interpretFrameType(): could not parse value "{value}"')
    return False, Mml.FrameType.FrameNone


# noinspection PyPep8Naming
def interpretFrameSpacing(
    value_list: str, em: int, ex: int
) -> tuple[bool, Mml.FrameSpacing]:
    fs: Mml.FrameSpacing = Mml.FrameSpacing()

    words: list[str] = value_list.split()
    if len(words) != 2:
        qt.qWarning(f'interpretFrameSpacing: could not parse value "{value_list}"')
        return False, Mml.FrameSpacing(round(0.4 * em), round(0.5 * ex))

    hor_ok: bool
    hor_ok, fs.m_hor = interpretSpacing(words[0], em, ex)
    ver_ok: bool
    ver_ok, fs.m_ver = interpretSpacing(words[1], em, ex)

    return hor_ok and ver_ok, fs


# noinspection PyPep8Naming
@overload
def mmlFindNodeSpec(type_: Mml.NodeType, /) -> NodeSpec | None: ...


# noinspection PyPep8Naming
@overload
def mmlFindNodeSpec(tag: str, /) -> NodeSpec | None: ...


# noinspection PyPep8Naming
def mmlFindNodeSpec(type_or_tag: Mml.NodeType | str, /) -> NodeSpec | None:
    if isinstance(type_or_tag, Mml.NodeType):
        type_: Mml.NodeType = type_or_tag
        for spec in g_node_spec_data:
            if type_ == spec.type:
                return spec
    elif isinstance(type_or_tag, str):
        tag: str = type_or_tag
        for spec in g_node_spec_data:
            if tag == spec.tag:
                return spec
    else:
        raise TypeError("Unsupported query type", type(type_or_tag))

    return None


# noinspection PyPep8Naming
def mmlCheckChildType(
    parent_type: Mml.NodeType,
    child_type: Mml.NodeType,
) -> tuple[bool, str]:
    if (
        parent_type == Mml.NodeType.UnknownNode
        or child_type == Mml.NodeType.UnknownNode
    ):
        return True, ""

    child_spec: NodeSpec | None = mmlFindNodeSpec(child_type)
    parent_spec: NodeSpec | None = mmlFindNodeSpec(parent_type)

    assert child_spec is not None
    assert parent_spec is not None

    allowed_child_types: Collection[str] = parent_spec.child_types
    # null list means any child type is valid
    if not allowed_child_types:
        return True, ""

    if child_spec.type_str not in allowed_child_types:
        return (
            False,
            f"illegal child {child_spec.type_str} for parent {parent_spec.type_str}",
        )

    return True, ""


# noinspection PyPep8Naming
def mmlCheckAttributes(
    child_type: Mml.NodeType, attr: MmlAttributeMap
) -> tuple[bool, str]:
    spec: NodeSpec | None = mmlFindNodeSpec(child_type)
    assert spec is not None

    allowed_attr: Collection[str] = spec.attributes
    # empty list means any attr is valid
    if not allowed_attr:
        return True, ""

    for name in attr:
        if ":" in name:
            continue

        if name not in allowed_attr:
            return False, f"illegal attribute {name} in {spec.type_str}"

    return True, ""


# noinspection PyPep8Naming
def mmlFindOperSpec(text: str, form: Mml.FormType) -> OperSpec:
    """
    :param text is a string between ``<mo>`` and ``</mo>``. It can be a character (``+``), an
    entity reference (``&infin;``) or a character reference (``&#x0221E``). Our
    job is to find an operator spec in the operator dictionary (``g_oper_spec_data``)
    that matches text. Things are further complicated by the fact, that many
    operators come in several forms (``prefix``, ``infix``, ``postfix``).

    If available, this function returns an operator spec matching text in the specified
    :param form. If such operator is not available, returns an operator spec that matches
    text, but of some other form in the preference order specified by the MathML spec.
    If that's not available either, returns the default operator spec.
    """

    # noinspection PyPep8Naming
    def searchEntitySpecData(value: str) -> Iterator[EntitySpec]:
        for name, code in g_xml_entity_data.items():
            if value == unescape(code):
                yield EntitySpec(name, code)

    # noinspection PyPep8Naming
    class OperSpecSearchResult:
        def __init__(self) -> None:
            self.prefix_form: OperSpec | None = None
            self.infix_form: OperSpec | None = None
            self.postfix_form: OperSpec | None = None

        def getForm(self, f: Mml.FormType) -> OperSpec | None:
            match f:
                case Mml.FormType.PrefixForm:
                    return self.prefix_form
                case Mml.FormType.InfixForm:
                    return self.infix_form
                case Mml.FormType.PostfixForm:
                    return self.postfix_form
            return self.postfix_form  #  just to avoid warning

        def haveForm(self, f: Mml.FormType) -> bool:
            return self.getForm(f) is not None

        # noinspection PyShadowingNames
        def addForm(self, spec: OperSpec) -> None:
            match spec.form:
                case Mml.FormType.PrefixForm:
                    self.prefix_form = spec
                case Mml.FormType.InfixForm:
                    self.infix_form = spec
                case Mml.FormType.PostfixForm:
                    self.postfix_form = spec

    # noinspection PyPep8Naming
    def searchOperSpecData(name: str) -> Iterator[OperSpec]:
        """
        Searches ``g_oper_spec_data`` and returns any instance of operator :param name. There may
        be more instances, but since the list is sorted, they will be next to each other.
        """
        for _spec in g_oper_spec_data:
            if _spec.name == name:
                yield _spec

    # noinspection PyPep8Naming
    def _mmlFindOperSpec() -> OperSpecSearchResult:
        """
        This searches ``g_oper_spec_data`` until at least one name in ``name_list`` is found with ``FormType`` ``form``,
        or until ``name_list`` is exhausted. The idea here is that if we don't find the operator in the
        specified form, we still want to use some other available form of that operator.
        """

        _result: OperSpecSearchResult = OperSpecSearchResult()

        for name in name_list:
            for _spec in searchOperSpecData(name):
                _result.addForm(_spec)
                if _result.haveForm(form):
                    break
            if _result.haveForm(form):
                break

        return _result

    name_list: list[str] = [text]

    # First, just try to find text in the operator dictionary.
    result: OperSpecSearchResult = _mmlFindOperSpec()

    if not result.haveForm(form):
        # Try to find other names for the operator represented by text.

        for ent in searchEntitySpecData(text):
            name_list.append(f"&{ent.name};")

        result = _mmlFindOperSpec()

    spec: OperSpec | None = result.getForm(form)
    if spec is not None:
        return spec

    spec = result.getForm(Mml.FormType.InfixForm)
    if spec is not None:
        return spec

    spec = result.getForm(Mml.FormType.PostfixForm)
    if spec is not None:
        return spec

    spec = result.getForm(Mml.FormType.PrefixForm)
    if spec is not None:
        return spec

    g_oper_spec_defaults: OperSpec = OperSpec(
        None,
        Mml.FormType.InfixForm,
        [
            "false",
            "false",
            "false",
            "thickmathspace",
            "1",
            "false",
            "thickmathspace",
            "false",
            "false",
        ],
        OperSpec.StretchDir.NoStretch,
    )

    return g_oper_spec_defaults


# noinspection PyPep8Naming
def mmlDictAttribute(name: str, spec: OperSpec) -> str:
    try:
        i: int = g_oper_spec_names.index(name)
    except IndexError:
        return ""
    else:
        return spec.attributes[i]


# noinspection PyPep8Naming
def rectToStr(rect: qt.QRect) -> str:
    return "[({}, {}), {}x{}]".format(
        rect.x(),
        rect.y(),
        rect.width(),
        rect.height(),
    )


# noinspection PyPep8Naming
def domToMmlNodeType(dom_node: qt.QDomNode) -> Mml.NodeType:
    mml_type: Mml.NodeType = Mml.NodeType.NoNode

    match dom_node.nodeType():
        case qt.QDomNode.NodeType.ElementNode:
            tag: str = dom_node.nodeName()
            spec: NodeSpec | None = mmlFindNodeSpec(tag)

            # treat unrecognised tags as mrow
            if spec is None:
                mml_type = Mml.NodeType.UnknownNode
            else:
                mml_type = spec.type

        case qt.QDomNode.NodeType.TextNode:
            mml_type = Mml.NodeType.TextNode

        case qt.QDomNode.NodeType.DocumentNode:
            mml_type = Mml.NodeType.MrowNode

        case qt.QDomNode.NodeType.EntityReferenceNode:
            # qt.qWarning(
            #     f'EntityReferenceNode: name="{dom_node.nodeName()}" value="{dom_node.nodeValue()}"'
            # )
            pass

        case (
            qt.QDomNode.NodeType.AttributeNode
            | qt.QDomNode.NodeType.CDATASectionNode
            | qt.QDomNode.NodeType.EntityNode
            | qt.QDomNode.NodeType.ProcessingInstructionNode
            | qt.QDomNode.NodeType.CommentNode
            | qt.QDomNode.NodeType.DocumentTypeNode
            | qt.QDomNode.NodeType.DocumentFragmentNode
            | qt.QDomNode.NodeType.NotationNode
            | qt.QDomNode.NodeType.BaseNode
            | qt.QDomNode.NodeType.CharacterDataNode
        ):
            pass

    return mml_type


# noinspection PyPep8Naming
def updateFontAttr(
    font_attr: MmlAttributeMap,
    n: "MmlNode | None",
    name: str,
    preferred_name: str = "",
) -> None:
    if preferred_name in font_attr or name in font_attr:
        return
    value: str = n.explicitAttribute(name)
    if value:
        font_attr[name] = value


# noinspection PyPep8Naming
def collectFontAttributes(node: "MmlNode | None") -> MmlAttributeMap:
    font_attr: MmlAttributeMap = {}

    n: "MmlNode | None" = node
    while n is not None:
        if n is node or n.nodeType() == Mml.NodeType.MstyleNode:
            updateFontAttr(font_attr, n, "mathvariant")
            updateFontAttr(font_attr, n, "mathsize")

            # depreciated attributes
            updateFontAttr(font_attr, n, "fontsize", "mathsize")
            updateFontAttr(font_attr, n, "fontweight", "mathvariant")
            updateFontAttr(font_attr, n, "fontstyle", "mathvariant")
            updateFontAttr(font_attr, n, "fontfamily", "mathvariant")

        n = n.parent()

    return font_attr


# noinspection PyPep8Naming
def interpretMathVariant(value: str) -> tuple[int, bool]:
    g_mv_data: dict[str, Mml.MathVariant] = {
        "normal": Mml.MathVariant.NormalMV,
        "bold": Mml.MathVariant.BoldMV,
        "italic": Mml.MathVariant.ItalicMV,
        "bold-italic": Mml.MathVariant.BoldMV | Mml.MathVariant.ItalicMV,
        "double-struck": Mml.MathVariant.DoubleStruckMV,
        "bold-fraktur": Mml.MathVariant.BoldMV | Mml.MathVariant.FrakturMV,
        "script": Mml.MathVariant.ScriptMV,
        "bold-script": Mml.MathVariant.BoldMV | Mml.MathVariant.ScriptMV,
        "fraktur": Mml.MathVariant.FrakturMV,
        "sans-serif": Mml.MathVariant.SansSerifMV,
        "bold-sans-serif": Mml.MathVariant.BoldMV | Mml.MathVariant.SansSerifMV,
        "sans-serif-italic": Mml.MathVariant.SansSerifMV | Mml.MathVariant.ItalicMV,
        "sans-serif-bold-italic": (
            Mml.MathVariant.SansSerifMV
            | Mml.MathVariant.ItalicMV
            | Mml.MathVariant.BoldMV
        ),
        "monospace": Mml.MathVariant.MonospaceMV,
    }
    return g_mv_data.get(value, Mml.MathVariant.NormalMV), value in g_mv_data


# noinspection PyPep8Naming
def interpretForm(value: str) -> tuple[Mml.FormType, bool]:
    match value:
        case "prefix":
            return Mml.FormType.PrefixForm, True
        case "infix":
            return Mml.FormType.InfixForm, True
        case "postfix":
            return Mml.FormType.PostfixForm, True
        case _:
            qt.qWarning(f'interpretForm(): could not parse value "{value}"')
            return Mml.FormType.InfixForm, False


# noinspection PyPep8Naming
def interpretColAlign(value_list: str, colnum: int) -> tuple[Mml.ColAlign, bool]:
    ok: bool
    value: str
    ok, value = interpretListAttr(value_list, colnum, "center")

    if value == "left":
        return Mml.ColAlign.ColAlignLeft, True
    if value == "right":
        return Mml.ColAlign.ColAlignRight, True
    if value == "center":
        return Mml.ColAlign.ColAlignCenter, True

    qt.qWarning(f'interpretColAlign(): could not parse value "{value}"')

    return Mml.ColAlign.ColAlignCenter, False


# noinspection PyPep8Naming
def interpretRowAlign(value_list: str, rownum: int) -> tuple[Mml.RowAlign, bool]:
    value: str = interpretListAttr(value_list, rownum, "axis")
    if value == "top":
        return Mml.RowAlign.RowAlignTop, True
    if value == "center":
        return Mml.RowAlign.RowAlignCenter, True
    if value == "bottom":
        return Mml.RowAlign.RowAlignBottom, True
    if value == "baseline":
        return Mml.RowAlign.RowAlignBaseline, True
    if value == "axis":
        return Mml.RowAlign.RowAlignAxis, True

    qt.qWarning('interpretRowAlign(): could not parse value "%s"' % value)
    return Mml.RowAlign.RowAlignAxis, False


# noinspection PyPep8Naming
def interpretPercentSpacing(value: str, base: int) -> tuple[int, bool]:
    if not value.endswith("%"):
        return 0, False

    value = value[:-1]
    # bool float_ok;
    # float factor = value.toFloat(&float_ok);
    try:
        factor = float(value)
    except ValueError:
        pass
    else:
        if factor >= 0:
            return int(base * factor / 100.0), True

    qt.qWarning(f'interpretPercentSpacing(): could not parse "{value}%"')
    return 0, False


# noinspection PyPep8Naming
def interpretPointSize(value: str) -> tuple[int, bool]:
    if not value.endswith("pt"):
        return 0, False

    value = value[:-2]
    try:
        pt_size: int = int(value)
    except ValueError:
        pass
    else:
        if pt_size > 0:
            return pt_size, True

    qt.qWarning(f'interpretPointSize(): could not parse "{value}pt"')
    return 0, False


# noinspection PyPep8Naming
def interpretDepreciatedFontAttr(
    font_attr: dict, fn: qt.QFont, em: int, ex: int
) -> qt.QFont:
    if "fontsize" in font_attr:
        value = font_attr["fontsize"]

        while True:

            ptsize, ok = interpretPointSize(value)
            if ok:
                fn.setPointSize(ptsize)
                break

            ptsize, ok = interpretPercentSpacing(value, fn.pointSize())
            if ok:
                fn.setPointSize(ptsize)
                break

            size, ok = interpretSpacing(value, em, ex)
            if ok:
                fn.setPixelSize(size)
                break

            break

    if "fontweight" in font_attr:
        value = font_attr["fontweight"]
        if value == "normal":
            fn.setBold(False)
        elif value == "bold":
            fn.setBold(True)
        else:
            qt.qWarning(
                'interpretDepreciatedFontAttr(): could not parse fontweight "%s"'
                % value
            )
    if "fontstyle" in font_attr:
        value = font_attr["fontstyle"]
        if value == "normal":
            fn.setItalic(False)
        elif value == "italic":
            fn.setItalic(True)
        else:
            qt.qWarning(
                'interpretDepreciatedFontAttr(): could not parse fontstyle "%s"' % value
            )
    if "fontfamily" in font_attr:
        value = font_attr["fontfamily"]
        fn.setFamily(value)
    return fn


# noinspection PyPep8Naming
def interpretMathSize(
    value: str, fn: qt.QFont, em: int, ex: int
) -> tuple[qt.QFont, bool]:
    ok: bool = True

    if value == "small":
        fn.setPointSize(int(fn.pointSize() * 0.7))
        return fn, ok

    if value == "normal":
        return fn, ok

    if value == "big":
        fn.setPointSize(int(fn.pointSize() * 1.5))
        return fn, ok

    ptsize, size_ok = interpretPointSize(value)
    if size_ok:
        fn.setPointSize(ptsize)
        return fn, ok

    size, size_ok = interpretSpacing(value, em, ex)
    if size_ok:
        fn.setPixelSize(size)
        return fn, ok

    ok = False
    qt.qWarning('interpretMathSize(): could not parse mathsize "%s"' % value)
    return fn, ok


# noinspection PyPep8Naming
def zeroLineThickness(s: str) -> bool:
    if not s or not s[0].isdigit():
        return False

    for c in s:
        if c.isdigit() and c != "0":
            return False
    return True
