# PyInstaller spec for the unsigned cell_3d_analysis desktop app.

from pathlib import Path

ROOT = Path.cwd()

a = Analysis(
    [str(ROOT / "gui" / "app.py")],
    pathex=[str(ROOT), str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "gui")],
    binaries=[],
    datas=[
        (str(ROOT / "config" / "config.yaml"), "config"),
        (str(ROOT / "docs" / "calculos_justificacion.md"), "docs"),
    ],
    hiddenimports=[
        "plot_panel",
        "pandas",
        "cellpose",
        "cellpose.models",
        "torch",
        "torchvision",
        "skimage",
        "skimage.filters",
        "skimage.color",
        "matplotlib",
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_tkagg",
        "czifile",
        "tifffile",
        "yaml",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cell3d_gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="cell3d_gui",
)
