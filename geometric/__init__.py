"""
The base import for the geomeTRIC module.
"""


from . import molecule
from . import optimize
from . import engine
from . import run_json

from ._version import get_versions as _get_versions
_versions = _get_versions()
__version__ = _versions['version']
__git_revision__ = _versions['full-revisionid']
del _get_versions, _versions

