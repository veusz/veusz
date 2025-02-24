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

import enum


class Mml:
    class NodeType(enum.Enum):
        NoNode = enum.auto()
        MiNode = enum.auto()
        MnNode = enum.auto()
        MfracNode = enum.auto()
        MrowNode = enum.auto()
        MsqrtNode = enum.auto()
        MrootNode = enum.auto()
        MsupNode = enum.auto()
        MsubNode = enum.auto()
        MsubsupNode = enum.auto()
        MoNode = enum.auto()
        MstyleNode = enum.auto()
        TextNode = enum.auto()
        MphantomNode = enum.auto()
        MfencedNode = enum.auto()
        MtableNode = enum.auto()
        MtrNode = enum.auto()
        MtdNode = enum.auto()
        MoverNode = enum.auto()
        MunderNode = enum.auto()
        MunderoverNode = enum.auto()
        MerrorNode = enum.auto()
        MtextNode = enum.auto()
        MpaddedNode = enum.auto()
        MspaceNode = enum.auto()
        MalignMarkNode = enum.auto()
        UnknownNode = enum.auto()

    class MathVariant(enum.IntFlag):
        NormalMV = 0x0000
        BoldMV = 0x0001
        ItalicMV = 0x0002
        DoubleStruckMV = 0x0004
        ScriptMV = 0x0008
        FrakturMV = 0x0010
        SansSerifMV = 0x0020
        MonospaceMV = 0x0040

    class FormType(enum.Enum):
        PrefixForm = enum.auto()
        InfixForm = enum.auto()
        PostfixForm = enum.auto()

    class ColAlign(enum.Enum):
        ColAlignLeft = enum.auto()
        ColAlignCenter = enum.auto()
        ColAlignRight = enum.auto()

    class RowAlign(enum.Enum):
        RowAlignTop = enum.auto()
        RowAlignCenter = enum.auto()
        RowAlignBottom = enum.auto()
        RowAlignAxis = enum.auto()
        RowAlignBaseline = enum.auto()

    class FrameType(enum.Enum):
        FrameNone = enum.auto()
        FrameSolid = enum.auto()
        FrameDashed = enum.auto()

    class FrameSpacing:
        def __init__(self, hor: int = 0, ver: int = 0) -> None:
            self.m_hor: int = hor
            self.m_ver: int = ver
