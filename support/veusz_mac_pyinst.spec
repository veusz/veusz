# -*- mode: python -*-

import glob

block_cipher = None

a = Analysis(
    ['../veusz/veusz_main.py'],
    binaries=[],
    datas=[],
    hiddenimports=['iminuit.iminuit_warnings','iminuit.latex','iminiuit._libiminuit','iminuit._minuit_methods'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)

# add necessary documentation, licence
binaries = a.binaries
for bin in ('VERSION', 'ChangeLog', 'AUTHORS', 'README.md', 'INSTALL', 'COPYING'):
    binaries += [ (bin, bin, 'DATA') ]

binaries += [
    ('embed.py', 'veusz/embed.py', 'DATA'),
    ('__init__.py', 'veusz/__init__.py', 'DATA'),
]

# add various required files to distribution
for f in (
        glob.glob('icons/*.png')  + glob.glob('icons/*.ico') +
        glob.glob('icons/*.svg') +
        glob.glob('examples/*.vsz') +
        glob.glob('examples/*.dat') + glob.glob('examples/*.csv') +
        glob.glob('examples/*.py') +
        glob.glob('ui/*.ui')
):
    binaries.append( (f, f, 'DATA') )

pyz = PYZ(
    a.pure, a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='veusz.exe',
    debug=False,
    strip=True,
    upx=False,
    console=False
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=False,
    name='veusz'
)

version = open('VERSION').read().strip()

app = BUNDLE(
    coll,
    name='Veusz.app',
    icon='../icons/veusz.icns',
    bundle_identifier=None,
    info_plist={
        'CFBundleShortVersionString': version,
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',

        'CFBundleDocumentTypes': [{
                'CFBundleTypeIconFile': 'veusz.icns',
                'CFBundleTypeName': 'Veusz document',
                'CFBundleTypeRole': 'Editor',
                'LSItemContentTypes': ['org.veusz.document', 'org.veusz.document_hdf5'],
                'LSHandlerRank': 'Owner',
            }],
        'UTExportedTypeDeclarations': [
            {
                'UTTypeConformsTo': ['public.data'],
                'UTTypeDescription': 'Veusz document',
                'UTTypeIconFile': 'veusz.icns',
                'UTTypeIdentifier': 'org.veusz.document',
                'UTTypeTagSpecification': {
                    'public.filename-extension': 'vsz',
                    'public.mime-type': 'application/vnd.veusz.document',
                }
            },
            {
                'UTTypeConformsTo': ['public.data'],
                'UTTypeDescription': 'Veusz HDF5 document',
                'UTTypeIconFile': 'veusz.icns',
                'UTTypeIdentifier': 'org.veusz.document_hdf5',
                'UTTypeTagSpecification': {
                    'public.filename-extension': 'vszh5',
                    'public.mime-type': 'application/vnd.veusz.document_hdf5',
                }
            },
        ]
    }
)
