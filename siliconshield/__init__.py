"""Local training and evaluation tools for RTL security models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("siliconshield")
except PackageNotFoundError:  # source checkout without an install
    __version__ = "0.0.0+source"
