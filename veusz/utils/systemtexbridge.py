"""Render TeX text using the system TeX toolchain."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


_SYSTEM_TEX_CACHE_AVAILABLE = None


def _external_command_env(base_env=None):
    """Return an environment suitable for external TeX commands.

    When Veusz runs from an AppImage, the runtime injects its own library
    directory into LD_LIBRARY_PATH and related variables.  That is fine for
    Veusz itself, but it can break host tools such as sh, xelatex, pdftocairo,
    or dvisvgm when they fork helper programs.  Strip those AppImage-specific
    entries while leaving the rest of the user's environment intact.
    """

    env = (os.environ if base_env is None else base_env).copy()

    appdir = env.get('APPDIR')
    if appdir:
        appdir = os.path.abspath(appdir)

        ld_library_path = env.get('LD_LIBRARY_PATH')
        if ld_library_path:
            cleaned = []
            for entry in ld_library_path.split(os.pathsep):
                entry = entry.strip()
                if not entry:
                    continue
                if os.path.abspath(entry).startswith(appdir + os.sep):
                    continue
                cleaned.append(entry)
            if cleaned:
                env['LD_LIBRARY_PATH'] = os.pathsep.join(cleaned)
            else:
                env.pop('LD_LIBRARY_PATH', None)

        for name in (
                'APPDIR', 'APPIMAGE', 'APPIMAGE_EXTRACT_AND_RUN',
                'APPIMAGE_ORIGINAL_DIR', 'APPIMAGE_STARTUP_DIR',
                'APPIMAGE_UUID'):
            env.pop(name, None)

        for name in ('QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
            value = env.get(name)
            if value and appdir in value:
                env.pop(name, None)

    return env


def _command(name):
    path = Path(name)
    if (path.is_absolute() or path.parent != Path('.')) and path.exists():
        return str(path)
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required TeX command not found in PATH: {name}")
    return path


def _run(cmd, cwd, env=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        output = (proc.stdout or "") + (proc.stderr or "")
        output = output.strip()
        if len(output) > 4000:
            output = output[-4000:]
        raise RuntimeError(output or f"Command failed: {' '.join(cmd)}")


def _ensure_system_tex_cache(env=None):
    """Create the TeX Live default cache paths if possible.

    Returns True if at least one writable default cache path exists.
    """

    global _SYSTEM_TEX_CACHE_AVAILABLE
    if _SYSTEM_TEX_CACHE_AVAILABLE is not None:
        return _SYSTEM_TEX_CACHE_AVAILABLE

    kpsewhich = shutil.which('kpsewhich')
    if kpsewhich is None:
        _SYSTEM_TEX_CACHE_AVAILABLE = False
        return False

    env = _external_command_env(env)

    writable_cache = False
    for var in ('TEXMFVAR', 'TEXMFCACHE', 'TEXMFHOME', 'TEXMFCONFIG'):
        proc = subprocess.run(
            [kpsewhich, '-var-value', var],
            check=False,
            text=True,
            capture_output=True,
            env=env,
        )
        if proc.returncode != 0:
            continue

        value = proc.stdout.strip()
        if not value:
            continue

        for raw_path in value.split(os.pathsep):
            path = raw_path.strip()
            if not path:
                continue
            try:
                Path(path).expanduser().mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            if var in ('TEXMFVAR', 'TEXMFCACHE') and os.access(path, os.W_OK):
                writable_cache = True

    _SYSTEM_TEX_CACHE_AVAILABLE = writable_cache
    return writable_cache


def _fallback_tex_env(base_env=None):
    """Return a writable TeX environment rooted in the system temp dir."""

    root = Path(tempfile.gettempdir()) / 'veusz-systemtex-cache'
    env = _external_command_env(base_env)
    for name, subdir in (
            ('TEXMFVAR', 'texmf-var'),
            ('TEXMFCACHE', 'texmf-var'),
            ('TEXMFHOME', 'texmf-home'),
            ('TEXMFCONFIG', 'texmf-config'),
            ('XDG_CACHE_HOME', 'xdg-cache'),
            ('XDG_CONFIG_HOME', 'xdg-config')):
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    return env


def _pdf_to_svg(pdf_file, svg_file, cwd, env=None):
    """Convert a PDF to SVG.

    Prefer pdftocairo for PDF-based TeX engines because recent dvisvgm
    releases may depend on an older Ghostscript or mutool build.
    """

    pdftocairo = shutil.which('pdftocairo')
    if pdftocairo is not None:
        _run(
            [
                pdftocairo,
                '-svg',
                str(pdf_file),
                str(svg_file),
            ],
            cwd=cwd,
            env=env,
        )
        if svg_file.exists():
            return

    dvisvgm = _command('dvisvgm')
    _run(
        [
            dvisvgm,
            '--pdf',
            '--no-fonts',
            '--no-styles',
            '--exact-bbox',
            '-v0',
            '-o', str(svg_file),
            str(pdf_file),
        ],
        cwd=cwd,
        env=env,
    )


def _normalize_hex_color(value):
    if not value:
        return None

    value = value.strip()
    if value.lower() in ('transparent', 'none'):
        return None
    if value.startswith('#'):
        value = value[1:]

    if len(value) == 3:
        value = ''.join(ch * 2 for ch in value)
    if len(value) != 6:
        return None

    try:
        int(value, 16)
    except ValueError:
        return None
    return value.upper()


def _engine_kind(engine):
    name = Path(engine).name.lower()
    if name == 'latex':
        return 'dvi'
    return 'pdf'


def _make_document(tex, text_size, background, foreground, preamble):
    baseline = max(text_size * 1.2, text_size + 2.0)
    parts = [
        r'\documentclass{article}',
        r'\usepackage{amsmath,amssymb,bm,xcolor}',
    ]

    preamble = (preamble or '').strip()
    if preamble:
        parts.append(preamble)

    parts.extend([
        r'\pagestyle{empty}',
        r'\begin{document}',
        rf'\fontsize{{{text_size:.6f}pt}}{{{baseline:.6f}pt}}\selectfont',
    ])

    bg = _normalize_hex_color(background)
    if bg is not None:
        parts.append(rf'\pagecolor[HTML]{{{bg}}}')

    fg = _normalize_hex_color(foreground)
    if fg is not None:
        parts.append(rf'\color[HTML]{{{fg}}}')

    parts.append(rf'$\displaystyle {tex}$')
    parts.append(r'\end{document}')
    return '\n'.join(parts)


def render_svg(
        tex, text_size=20.0, width=720,
        foreground="#000000", background="transparent",
        engine='latex', preamble=''):
    """Render TeX source into SVG bytes using latex and dvisvgm."""
    del width

    engine = (engine or 'latex').strip()
    tex_engine = _command(engine)
    kind = _engine_kind(engine)
    texenv = _external_command_env()
    if not _ensure_system_tex_cache(texenv):
        texenv = _fallback_tex_env(texenv)

    with tempfile.TemporaryDirectory(prefix='veusz-systemtex-') as tmp:
        tmpdir = Path(tmp)
        texfile = tmpdir / 'textext.tex'
        texfile.write_text(
            _make_document(tex, float(text_size), background, foreground, preamble),
            encoding='utf-8'
        )

        _run(
            [
                tex_engine,
                '-interaction=nonstopmode',
                '-halt-on-error',
                '-no-shell-escape',
                '-output-directory', str(tmpdir),
                str(texfile),
            ],
            cwd=tmpdir,
            env=texenv,
        )

        if kind == 'dvi':
            svgfile = tmpdir / 'textext.svg'
            inputfile = tmpdir / 'textext.dvi'
            dvisvgm = _command('dvisvgm')
            dvisvgm_args = [
                dvisvgm,
                '--no-fonts',
                '--no-styles',
                '--exact-bbox',
                '-v0',
                '-o', str(svgfile),
                str(inputfile),
            ]
            _run(dvisvgm_args, cwd=tmpdir, env=texenv)
        else:
            inputfile = tmpdir / 'textext.pdf'
            svgfile = tmpdir / 'textext.svg'
            _pdf_to_svg(inputfile, svgfile, cwd=tmpdir, env=texenv)

        return svgfile.read_bytes()
