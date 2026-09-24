"""PDF export DPI regression tests (requires Qt and Veusz dependencies).

Run with: python -m unittest discover -s tests -p test_export_pdf.py
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

from veusz import document, qtall as qt
from veusz.document import export


class PDFResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = qt.QApplication.instance() or qt.QApplication([])

    def test_gui_export_entry_point(self):
        # The GUI uses the class re-exported by the document package.
        self.assertIs(document.AsyncExport, export.AsyncExport)
        self.assertEqual(
            document.AsyncExport.getDPI(SimpleNamespace(pdfdpi=600), '.pdf'),
            (600, 600))

    def test_layout_uses_output_resolution(self):
        # EPS and PS are rendered through the same intermediate PDF writer.
        for extension in ('.pdf', '.eps', '.ps'):
            for dpi in (72, 150, 300, 600, 1200):
                with self.subTest(extension=extension, dpi=dpi):
                    options = SimpleNamespace(pdfdpi=dpi)
                    self.assertEqual(
                        export.AsyncExport.getDPI(options, extension),
                        (dpi, dpi))

    def test_pdf_writer_matches_layout(self):
        for dpi in (72, 150, 300, 600, 1200):
            with self.subTest(dpi=dpi), tempfile.TemporaryDirectory() as tmp:
                options = SimpleNamespace(pdfdpi=dpi, color=True)
                layoutdpi = export.AsyncExport.getDPI(options, '.pdf')
                observed = []

                def render(painter):
                    self.assertTrue(painter.isActive())
                    dev = painter.device()
                    observed.append((dev.logicalDpiX(), dev.logicalDpiY()))
                    painter.drawRect(qt.QRectF(
                        layoutdpi[0] / 2, layoutdpi[1] / 2,
                        layoutdpi[0], layoutdpi[1]))

                # Two pages also exercise the page transition path.
                helper = SimpleNamespace(
                    dpi=layoutdpi,
                    pagesize=(layoutdpi[0] * 4, layoutdpi[1] * 3),
                    renderToPainter=render)
                filename = os.path.join(tmp, 'dpi.pdf')
                export.ExportPDFRunnable(
                    options, filename, [helper, helper]).doExport()
                self.assertEqual(observed, [layoutdpi, layoutdpi])
                self.assertGreater(os.path.getsize(filename), 0)


if __name__ == '__main__':
    unittest.main()
