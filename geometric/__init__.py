"""
The base import for the geomeTRIC module.
"""


from . import molecule
from . import optimize
from . import engine

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
