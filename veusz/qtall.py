#    Copyright (C) 2008 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

# ruff: noqa: F401

"""A convenience module to import used Qt symbols from."""
import typing

# noinspection PyUnresolvedReferences
from qtpy import API_NAME

# noinspection PyUnresolvedReferences
from qtpy.QtCore import (
    QAbstractItemModel,
    QAbstractListModel,
    QAbstractTableModel,
    QBuffer,
    QByteArray,
    QCoreApplication,
    QDir,
    QEvent,
    QIODevice,
    QItemSelectionModel,
    QLine,
    QLineF,
    QLocale,
    QMarginsF,
    QMimeData,
    QModelIndex,
    QMutex,
    QObject,
    QPersistentModelIndex,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QRunnable,
    QSemaphore,
    QSettings,
    QSignalMapper,
    QSize,
    QSizeF,
    QSocketNotifier,
    QStandardPaths,
    QStringListModel,
    QThread,
    QThreadPool,
    QTime,
    QTimer,
    QTranslator,
    QUrl,
    Qt,
    Signal,
    SignalInstance,
    Slot,
    qVersion,
    qWarning,
)


try:
    from qtpy.QtCore import QRegExp
except ImportError:
    from qtpy.QtCore import QRegularExpression as QRegExp


# noinspection PyUnresolvedReferences
from qtpy.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QColor,
    QCursor,
    QDesktopServices,
    QDoubleValidator,
    QFileSystemModel,
    QFont,
    QFontDatabase,
    QFontInfo,
    QFontMetrics,
    QFontMetricsF,
    QIcon,
    QIconEngine,
    QImage,
    QImageWriter,
    QIntValidator,
    QKeySequence,
    QMouseEvent,
    QPageLayout,
    QPageSize,
    QPaintDevice,
    QPaintEngine,
    QPaintEngineState,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPicture,
    QPixmap,
    QPolygonF,
    QRegion,
    QRegularExpressionValidator,
    QScreen,
    QTextCursor,
    QTextDocument,
    QTextItem,
    QTextOption,
    QTransform,
    QValidator,
    qAlpha,
    qBlue,
    qGreen,
    qRed,
    qRgba,
)

try:
    from qtpy.QtGui import QRegExpValidator
except ImportError:
    from qtpy.QtGui import QRegularExpressionValidator as QRegExpValidator

if (
    hasattr(QPaintDevice, "PaintDeviceMetric")
    and hasattr(QPaintDevice.PaintDeviceMetric, "PdmDevicePixelRatioF_EncodedA")
    and hasattr(QPaintDevice.PaintDeviceMetric, "PdmDevicePixelRatioF_EncodedB")
):
    # Qt>=6.8

    class QPaintDevice(QPaintDevice):
        if not hasattr(QPaintDevice, "encodeMetricF"):

            @staticmethod
            def encodeMetricF(
                metric: QPaintDevice.PaintDeviceMetric,
                value: float,
            ) -> int:
                from struct import pack, unpack, error

                try:
                    return unpack("<ii", pack("<d", value))[metric.value & 1]
                except error:
                    return 0

        def metric(self, metric: QPaintDevice.PaintDeviceMetric) -> int:
            if metric == QPaintDevice.PaintDeviceMetric.PdmDevicePixelRatioF_EncodedA:
                return QPaintDevice.encodeMetricF(
                    metric,
                    self.metric(QPaintDevice.PaintDeviceMetric.PdmDevicePixelRatio),
                )
            elif metric == (
                QPaintDevice.PaintDeviceMetric.PdmDevicePixelRatioF_EncodedB
            ):
                return QPaintDevice.encodeMetricF(metric, 1)
            return super().metric(metric)


try:
    QPaintEngine.User + 11
except TypeError:
    QPaintEngine.Type.__add__ = lambda a, b: QPaintEngine.Type(
        (a.value if isinstance(a, QPaintEngine.Type) else a)
        + (b.value if isinstance(b, QPaintEngine.Type) else b)
    )


# noinspection PyUnresolvedReferences
from qtpy.QtSvg import QSvgRenderer

try:
    from qtpy.QtSvg import QGraphicsSvgItem
except ImportError:
    from qtpy.QtSvgWidgets import QGraphicsSvgItem

# noinspection PyUnresolvedReferences
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFontComboBox,
    QFrame,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QItemDelegate,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QStyle,
    QStyledItemDelegate,
    QTabBar,
    QTabWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QToolButton,
    QTreeView,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QSplashScreen,
)

try:
    from qtpy.QtWidgets import QDesktopWidget
except ImportError:

    class QDesktopWidget:
        # noinspection PyPep8Naming
        @staticmethod
        def availableGeometry() -> QRect:
            return QApplication.primaryScreen().availableGeometry()

        # noinspection PyPep8Naming
        @staticmethod
        def screenGeometry() -> QRect:
            return QApplication.primaryScreen().geometry()

    QApplication.desktop = QDesktopWidget


# noinspection PyUnresolvedReferences
from qtpy.QtXml import QDomDocument, QDomNamedNodeMap, QDomNode, QDomNodeList

try:
    # noinspection PyUnresolvedReferences
    from qtpy.QtSvgWidgets import QGraphicsSvgItem
except ImportError:
    # noinspection PyUnresolvedReferences
    from qtpy.QtSvg import QGraphicsSvgItem

# noinspection PyUnresolvedReferences
from qtpy.QtPrintSupport import QAbstractPrintDialog, QPrinter, QPrintDialog

# noinspection PyUnresolvedReferences
from qtpy.uic import loadUi

if not hasattr(QPrinter, "Color"):
    QPrinter.Color = QPrinter.ColorMode.Color
if not hasattr(QPrinter, "GrayScale"):
    QPrinter.GrayScale = QPrinter.ColorMode.GrayScale
if not hasattr(QPrinter, "PdfFormat"):
    QPrinter.PdfFormat = QPrinter.OutputFormat.PdfFormat
if not hasattr(QPrinter, "HighResolution"):
    QPrinter.HighResolution = QPrinter.PrinterMode.HighResolution
if not hasattr(QPrinter, "FirstPageFirst"):
    QPrinter.FirstPageFirst = QPrinter.PageOrder.FirstPageFirst

PYQT_VERSION_STR = f"{API_NAME} {qVersion()}"
pyqtSignal = Signal
pyqtSlot = Slot


class QDirModel(QFileSystemModel):
    """
    This class is obsolete.

    See https://doc.qt.io/qt-5/qdirmodel.html.
    """

    @typing.overload
    def __init__(self, parent: QWidget | None = None) -> None: ...
    @typing.overload
    def __init__(
        self,
        nameFilters: typing.Sequence[str],
        filters: QDir.Filter,
        sort: QDir.SortFlag,
        parent: QWidget | None = None,
    ) -> None: ...
    def __init__(self, *args, **kwargs) -> None:
        if len(args) + len(kwargs) not in (1, 4):
            raise TypeError("Unsupported arguments")
        parent: QWidget | None = kwargs.pop("parent", None)
        nameFilters: typing.Sequence[str] | None = kwargs.pop("nameFilters", None)
        filters: QDir.Filter | None = kwargs.pop("filters", None)

        for arg in args:
            if filters is None and isinstance(arg, QDir.Filter):
                filters = arg
                continue
            if (
                nameFilters is None
                and isinstance(arg, typing.Sequence)
                and all(isinstance(nameFilter, str) for nameFilter in arg)
            ):
                nameFilters = arg
                continue
            if parent is None and isinstance(arg, QWidget):
                parent = arg
                continue

        super().__init__(parent)

        if nameFilters is not None:
            self.setNameFilters(nameFilters)
        if filters is not None:
            self.setFilter(filters)


try:
    from qtpy import sip
except ImportError:
    # noinspection PyPep8Naming
    class sip:
        """SIP stub for PySide2/6"""

        SIP_VERSION = 0
        SIP_VERSION_STR = "N/A"

        @staticmethod
        def isdeleted(obj: object) -> bool:
            from qtpy import shiboken

            return not shiboken.isValid(obj)


# see https://doc.qt.io/qt-5/qfontdatabase-obsolete.html#supportsThreadedFontRendering
QFontDatabase.supportsThreadedFontRendering = lambda: True


if not hasattr(QSignalMapper, "mapped"):

    class QSignalMapper(QSignalMapper):
        @property
        def mapped(self) -> dict[type, SignalInstance]:
            return {
                int: self.mappedInt,
                str: self.mappedString,
                object: self.mappedObject,
            }


if hasattr(QComboBox, "textActivated"):
    # noinspection PyTypeChecker
    QComboBox.activated = {
        int: QComboBox.activated,
        str: QComboBox.textActivated,
    }

    class QComboBox(QComboBox):

        @property
        def activated(self) -> dict[type, SignalInstance]:
            # noinspection PyTypeChecker
            return {
                int: super().activated,
                str: self.textActivated,
            }

    # noinspection PyTypeChecker
    QFontComboBox.activated = {
        int: QFontComboBox.activated,
        str: QFontComboBox.textActivated,
    }

    class QFontComboBox(QFontComboBox):

        @property
        def activated(self) -> dict[type, SignalInstance]:
            # noinspection PyTypeChecker
            return {
                int: super().activated,
                str: self.textActivated,
            }


class QMenu(QMenu):
    _orphaned_menus_to_keep_them_alive: list[QMenu] = []
    # NB: check for a memory leak

    def __init__(self, *args) -> None:
        if args:
            super().__init__(*args)
        else:
            super().__init__()
            QMenu._orphaned_menus_to_keep_them_alive.append(self)


def __getattr__(name: str) -> typing.Any:
    match name:
        case "qApp":
            return QApplication.instance()
        case _:
            raise AttributeError(name)
