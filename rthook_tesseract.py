"""Point tesseract_robotics at the DLLs the bundle carries, before it is imported.

The wheel is delvewheel-patched: its 73 native DLLs live in a sibling directory named
``tesseract_robotics.libs``, and ``__init__.py`` finds them by walking up from its own
``__file__``.  Frozen, that walk lands in the right place only if PyInstaller reports a
``__file__`` under the extraction directory, which is not something to depend on.

``__init__.py`` also honours ``TESSERACT_PYTHON_DLL_PATH`` and calls ``add_dll_directory``
on every entry, so setting it here says the same thing without the walk.  Runtime hooks
run before user code, which is what makes this early enough to matter.
"""
import os
import sys

if hasattr(sys, "_MEIPASS"):
    os.environ["TESSERACT_PYTHON_DLL_PATH"] = os.path.join(
        sys._MEIPASS, "tesseract_robotics.libs")
