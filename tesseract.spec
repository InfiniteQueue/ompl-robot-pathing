# -*- mode: python ; coding: utf-8 -*-
"""One self-contained weldpath.exe, carrying the tesseract_robotics native stack.

Two things here are not defaults and are the reason this is a spec file rather than a
command line.

``binaries`` names the delvewheel DLL set explicitly.  Nothing imports those 73 files --
the .pyd modules link against them and the Windows loader resolves them from a directory
added at import time -- so PyInstaller's dependency analysis cannot see them at all.  The
destination is ``tesseract_robotics.libs`` at the bundle root, which is where the wheel's
own ``__init__.py`` looks: one level up from the package.

``hiddenimports`` names the SWIG submodules, which import each other dynamically rather
than by a statement static analysis can follow.
"""
import glob
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

LIBS = os.path.join(sys.prefix, "Lib", "site-packages", "tesseract_robotics.libs")
dlls = sorted(glob.glob(os.path.join(LIBS, "*.dll")))
if not dlls:
    raise SystemExit(f"no delvewheel DLLs found in {LIBS}")
print(f"spec: carrying {len(dlls)} DLLs from {LIBS}")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[(p, "tesseract_robotics.libs") for p in dlls],
    datas=[],
    hiddenimports=collect_submodules("tesseract_robotics"),
    hookspath=[],
    runtime_hooks=["rthook_tesseract.py"],
    # Installed in the venv but unused by this project; cv2 alone is ~90 MB.
    excludes=["cv2", "aiohttp", "tkinter", "matplotlib", "PIL", "setuptools",
              "pip", "pytest", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="weldpath",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
)
