#    Copyright (C) 2026 Veusz developers
#
#    This file is part of Veusz.
#
#    Veusz is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    Veusz is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################

"""Windows clipboard utilities for placing Enhanced Metafile (EMF) and images on clipboard."""

import os
import sys
import time
import logging
import ctypes

logger = logging.getLogger(__name__)

CF_DIB = 8
CF_ENHMETAFILE = 14
GMEM_MOVEABLE = 0x0002

def is_windows():
    """Return True if running on Windows."""
    return sys.platform == 'win32'

def copy_images_to_clipboard(emf_bytes=None, png_bytes=None, bmp_bytes=None):
    """Place native EMF and image formats onto Windows clipboard.

    Registers:
      - CF_ENHMETAFILE (native vector format for MS Office: Word, PowerPoint, Excel)
      - CF_DIB (native bitmap format for Paint, Photoshop, image viewers)
      - 'PNG' (native PNG format for Slack, Teams, browsers, Discord)

    Returns True on success, False otherwise.
    """
    if not is_windows():
        return False

    try:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        kernel32 = ctypes.windll.kernel32

        user32.OpenClipboard.argtypes = [ctypes.c_void_p]
        user32.OpenClipboard.restype = ctypes.c_bool
        user32.CloseClipboard.argtypes = []
        user32.CloseClipboard.restype = ctypes.c_bool
        user32.EmptyClipboard.argtypes = []
        user32.EmptyClipboard.restype = ctypes.c_bool
        user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        user32.SetClipboardData.restype = ctypes.c_void_p
        user32.RegisterClipboardFormatW.argtypes = [ctypes.c_wchar_p]
        user32.RegisterClipboardFormatW.restype = ctypes.c_uint

        kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalUnlock.restype = ctypes.c_bool

        gdi32.SetEnhMetaFileBits.argtypes = [ctypes.c_uint, ctypes.c_char_p]
        gdi32.SetEnhMetaFileBits.restype = ctypes.c_void_p
    except Exception as e:
        logger.warning('Failed to load Windows DLLs for clipboard: %s', e)
        return False

    # Retry opening clipboard in case another process holds it temporarily
    opened = False
    for _ in range(10):
        if user32.OpenClipboard(None):
            opened = True
            break
        time.sleep(0.05)

    if not opened:
        logger.warning('OpenClipboard failed (clipboard locked)')
        return False

    try:
        user32.EmptyClipboard()

        # 1. Enhanced Metafile (EMF) - Vector graphics for MS Office
        if emf_bytes:
            hemf = gdi32.SetEnhMetaFileBits(len(emf_bytes), emf_bytes)
            if hemf:
                user32.SetClipboardData(CF_ENHMETAFILE, hemf)
            else:
                logger.warning('SetEnhMetaFileBits failed')

        # 2. DIB from BMP bytes (BMP header is 14 bytes, followed by DIB header + pixel data)
        if bmp_bytes and len(bmp_bytes) > 14 and bmp_bytes[:2] == b'BM':
            dib_data = bmp_bytes[14:]
            h_mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(dib_data))
            if h_mem:
                ptr = kernel32.GlobalLock(h_mem)
                if ptr:
                    ctypes.memmove(ptr, dib_data, len(dib_data))
                    kernel32.GlobalUnlock(h_mem)
                    user32.SetClipboardData(CF_DIB, h_mem)

        # 3. PNG format
        if png_bytes:
            cf_png = user32.RegisterClipboardFormatW("PNG")
            if cf_png:
                h_mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(png_bytes))
                if h_mem:
                    ptr = kernel32.GlobalLock(h_mem)
                    if ptr:
                        ctypes.memmove(ptr, png_bytes, len(png_bytes))
                        kernel32.GlobalUnlock(h_mem)
                        user32.SetClipboardData(cf_png, h_mem)

        return True
    finally:
        user32.CloseClipboard()

def copy_emf_file_to_clipboard(filepath):
    """Read an EMF file from disk and copy it to the Windows clipboard."""
    if not is_windows() or not os.path.isfile(filepath):
        return False
    with open(filepath, 'rb') as f:
        data = f.read()
    return copy_images_to_clipboard(emf_bytes=data)
