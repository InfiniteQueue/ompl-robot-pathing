"""Weld path generation for a FANUC robot using Tesseract.

Reads ``manifest.json`` from a study directory, builds a Tesseract scene from it, plans
the motion through the manifest's locators, and writes ``waypoints.json`` back into the
same directory.
"""
from __future__ import annotations

__all__ = ["cell", "manifest", "meshprep", "output", "planning", "profile", "scene",
           "toolpath"]
__version__ = "0.1.0"
