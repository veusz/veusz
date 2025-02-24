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

from .._functions import (
    interpretFrameSpacing,
    interpretFrameType,
    interpretListAttr,
    interpretPercentSpacing,
)
from .... import qtall as qt
from .MmlMtrNode import MmlMtrNode

from ..Mml import Mml
from .MmlTableBaseNode import MmlTableBaseNode

if TYPE_CHECKING:
    from .MmlNode import MmlNode
    from ..MmlDocument import MmlDocument

MmlAttributeMap = dict[str, str]


class MmlMtableNode(MmlTableBaseNode):
    # noinspection PyPep8Naming
    class CellSizeData:
        def __init__(self) -> None:
            self.col_widths: list[int] = []
            self.row_heights: list[int] = []

        def init(self, first_row: MmlNode | None) -> None:
            self.col_widths.clear()
            self.row_heights.clear()

            mtr: MmlNode | None = first_row
            while mtr is not None:

                assert mtr.nodeType() == Mml.NodeType.MtrNode

                col_cnt: int = 0
                mtd: MmlNode | None = mtr.firstChild()
                while mtd is not None:

                    assert mtd.nodeType() == Mml.NodeType.MtrNode

                    mtdmr: qt.QRect = mtd.myRect()

                    if col_cnt == len(self.col_widths):
                        self.col_widths.append(mtdmr.width())
                    else:
                        self.col_widths[col_cnt] = max(
                            self.col_widths[col_cnt], mtdmr.width()
                        )

                    mtd = mtd.nextSibling()

                self.row_heights.append(mtr.myRect().height())

                mtr = mtr.nextSibling()

        def numCols(self) -> int:
            return len(self.col_widths)

        def numRows(self) -> int:
            return len(self.row_heights)

        def colWidthSum(self) -> int:
            return sum(self.col_widths)

        def rowHeightSum(self) -> int:
            return sum(self.row_heights)

    def __init__(
        self,
        document: MmlDocument | None,
        attribute_map: MmlAttributeMap,
    ) -> None:
        super().__init__(Mml.NodeType.UnknownNode, document, attribute_map)

        self.m_cell_size_data: MmlMtableNode.CellSizeData = MmlMtableNode.CellSizeData()
        self.m_content_width: int = 0
        self.m_content_height: int = 0

    def rowspacing(self) -> int:
        value: str = self.explicitAttribute("rowspacing")
        if not value:
            return self.ex()

        ok: bool
        r: int
        ok, r = self.interpretSpacing(value)

        return r if ok else self.ex()

    def columnspacing(self) -> int:
        value: str = self.explicitAttribute("columnspacing")
        if not value:
            return int(0.8 * self.em())

        ok: bool
        r: int
        ok, r = self.interpretSpacing(value)

        return r if ok else int(0.8 * self.em())

    def framespacing_hor(self) -> int:
        if self.frame() == Mml.FrameType.FrameNone:
            return int(0.2 * self.em())

        value: str = self.explicitAttribute("framespacing", "0.4em 0.5ex")

        fs: Mml.FrameSpacing
        ok, fs = interpretFrameSpacing(value, self.em(), self.ex())

        if ok:
            return fs.m_hor
        else:
            return int(0.4 * self.em())

    def framespacing_ver(self) -> int:
        if self.frame() == Mml.FrameType.FrameNone:
            return int(0.2 * self.em())

        value: str = self.explicitAttribute("framespacing", "0.4em 0.5ex")

        fs: Mml.FrameSpacing
        ok, fs = interpretFrameSpacing(value, self.em(), self.ex())

        if ok:
            return fs.m_ver
        else:
            return int(0.4 * self.em())

    def frame(self) -> Mml.FrameType:
        value: str = self.explicitAttribute("frame", "none")
        return interpretFrameType(value, 0)[1]

    def columnlines(self, idx: int) -> Mml.FrameType:
        value: str = self.explicitAttribute("columnlines", "none")
        return interpretFrameType(value, idx)[1]

    def rowlines(self, idx: int) -> Mml.FrameType:
        value: str = self.explicitAttribute("rowlines", "none")
        return interpretFrameType(value, idx)[1]

    def layoutSymbol(self) -> None:
        # Obtain natural widths of columns
        self.m_cell_size_data.init(self.firstChild())

        col_spc: int = self.columnspacing()
        row_spc: int = self.rowspacing()
        frame_spc_hor: int = self.framespacing_hor()
        columnwidth_attr: str = self.explicitAttribute("columnwidth", "auto")

        ok: bool
        w: int

        # Is table width set by user? If so, set col_width_sum and never ever change it.
        col_width_sum: int = self.m_cell_size_data.colWidthSum()
        width_set_by_user: bool = False
        width_str: str = self.explicitAttribute("width", "auto")
        if width_str != "auto":
            ok, w = self.interpretSpacing(width_str)
            if ok:
                col_width_sum = (
                    w
                    - col_spc * (self.m_cell_size_data.numCols() - 1)
                    - frame_spc_hor * 2
                )
                width_set_by_user = True

        # Find out what kind of columns we are dealing with and set the widths of
        # statically sized columns.
        # sum of widths of statically sized set columns
        fixed_width_sum: int = 0
        # sum of natural widths of auto sized columns
        auto_width_sum: int = 0
        # sum of natural widths of relatively sized columns
        relative_width_sum: int = 0
        # total fraction of width taken by relatively sized columns
        relative_fraction_sum: float = 0.0

        value: str

        for i in range(self.m_cell_size_data.numCols()):
            value = interpretListAttr(columnwidth_attr, i, "auto")

            # Is it an auto sized column?
            if value in ("auto", "fit"):
                auto_width_sum += self.m_cell_size_data.col_widths[i]
                continue

            # Is it a statically sized column?
            ok, w = self.interpretSpacing(value)
            if ok:
                # Yup, sets its width to the user specified value
                self.m_cell_size_data.col_widths[i] = w
                fixed_width_sum += w
                continue

            # Is it a relatively sized column?
            if value.endswith("%"):
                value = value[:-1]
                try:
                    factor: float = float(value)
                except ValueError:
                    qt.qWarning(
                        f"MmlMtableNode::layoutSymbol(): could not parse value {value}%"
                    )
                else:
                    factor /= 100.0
                    relative_width_sum += self.m_cell_size_data.col_widths[i]
                    relative_fraction_sum += factor
                    if not width_set_by_user:
                        # If the table width was not set by the user, we are free to increase
                        # it so that the width of this column will be >= than its natural width
                        min_col_width_sum: int = round(
                            self.m_cell_size_data.col_widths[i] / factor
                        )
                        if min_col_width_sum > col_width_sum:
                            col_width_sum = min_col_width_sum
                    continue

            # Relatively sized column, but we failed to parse the factor. Treat is like an auto
            # column.
            auto_width_sum += self.m_cell_size_data.col_widths[i]

        # Work out how much space remains for the auto columns, after allocating
        # the statically sized and the relatively sized columns.
        required_auto_width_sum: int = (
            col_width_sum
            - round(relative_fraction_sum * col_width_sum)
            - fixed_width_sum
        )

        if not width_set_by_user and required_auto_width_sum < auto_width_sum:
            if relative_fraction_sum < 1:
                col_width_sum = round(
                    (fixed_width_sum + auto_width_sum) / (1 - relative_fraction_sum)
                )
            else:
                col_width_sum = fixed_width_sum + auto_width_sum + relative_width_sum
            required_auto_width_sum = auto_width_sum

        # Ratio by which we have to shrink/grow all auto sized columns to make it all fit
        auto_width_scale: float = 1.0
        if auto_width_sum > 0:
            auto_width_scale = required_auto_width_sum / auto_width_sum

        # Set correct sizes for the auto sized and the relatively sized columns.
        for i in range(self.m_cell_size_data.numCols()):
            value = interpretListAttr(columnwidth_attr, i, "auto")

            # Is it a relatively sized column?
            if value.endswith("%"):
                ok, w = interpretPercentSpacing(value, col_width_sum)
                if ok:
                    self.m_cell_size_data.col_widths[i] = w
                else:
                    # We're treating parsing errors here as auto sized columns
                    self.m_cell_size_data.col_widths[i] = round(
                        auto_width_scale * self.m_cell_size_data.col_widths[i]
                    )
            # Is it an auto sized column?
            elif value == "auto":
                self.m_cell_size_data.col_widths[i] = round(
                    auto_width_scale * self.m_cell_size_data.col_widths[i]
                )

        # qt.qWarning(
        #     "".join(
        #         "[w={} {}%]".format(
        #             col_width, 100.0 * col_width / self.m_cell_size_data.colWidthSum()
        #         )
        #         for col_width in self.m_cell_size_data.col_widths
        #     )
        # )

        self.m_content_width = self.m_cell_size_data.colWidthSum() + col_spc * (
            self.m_cell_size_data.numCols() - 1
        )
        self.m_content_height = self.m_cell_size_data.rowHeightSum() + row_spc * (
            self.m_cell_size_data.numRows() - 1
        )

        bottom: int = -self.m_content_height // 2
        child: MmlNode | None = self.firstChild()
        while child is not None:
            assert child.nodeType() == Mml.NodeType.MtrNode
            row: MmlMtrNode = cast(MmlMtrNode, child)
            row.layoutCells(self.m_cell_size_data.col_widths, col_spc)
            rmr: qt.QRect = row.myRect()
            row.setRelOrigin(qt.QPoint(0, bottom - rmr.top()))
            bottom += rmr.height() + row_spc
            child = child.nextSibling()

    def symbolRect(self) -> qt.QRect:
        frame_spc_hor: int = self.framespacing_hor()
        frame_spc_ver: int = self.framespacing_ver()

        return qt.QRect(
            -frame_spc_hor,
            -self.m_content_height / 2 - frame_spc_ver,
            self.m_content_width + 2 * frame_spc_hor,
            self.m_content_height + 2 * frame_spc_ver,
        )

    def paintSymbol(self, p: qt.QPainter) -> None:
        f: Mml.FrameType = self.frame()
        pen: qt.QPen
        if f != Mml.FrameType.FrameNone:
            p.save()

            pen = p.pen()
            if f == Mml.FrameType.FrameDashed:
                pen.setStyle(qt.Qt.PenStyle.DashLine)
            else:
                pen.setStyle(qt.Qt.PenStyle.SolidLine)
            p.setPen(pen)
            p.drawRect(self.myRect())

            p.restore()

        p.save()

        col_spc: int = self.columnspacing()
        row_spc: int = self.rowspacing()

        pen = p.pen()
        col_offset: int = 0
        for i in range(self.m_cell_size_data.numCols() - 1):
            f = self.columnlines(i)
            col_offset += self.m_cell_size_data.col_widths[i]

            if f != Mml.FrameType.FrameNone:
                if f == Mml.FrameType.FrameDashed:
                    pen.setStyle(qt.Qt.PenStyle.DashLine)
                else:
                    pen.setStyle(qt.Qt.PenStyle.SolidLine)

                p.setPen(pen)
                x: int = col_offset + col_spc // 2
                p.drawLine(
                    x, -self.m_content_height // 2, x, self.m_content_height // 2
                )

            col_offset += col_spc

        row_offset: int = 0
        for i in range(self.m_cell_size_data.numRows() - 1):
            f = self.rowlines(i)
            row_offset += self.m_cell_size_data.row_heights[i]

            if f != Mml.FrameType.FrameNone:
                if f == Mml.FrameType.FrameDashed:
                    pen.setStyle(qt.Qt.PenStyle.DashLine)
                else:
                    pen.setStyle(qt.Qt.PenStyle.SolidLine)

                p.setPen(pen)
                x: int = col_offset + col_spc // 2
                p.drawLine(x, -self.m_content_width // 2, x, self.m_content_width // 2)

            row_offset += row_spc

        p.restore()
