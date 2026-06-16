"""dl_utils package.

All shared constants live in :mod:`dl_utils.constants`; they are re-exported here so
that ``from dl_utils import NUCLEUS_LABEL_KEY`` keeps working for existing callers
(including the smlm-sonata submodule and representationlearning scripts). Edit
``constants.py`` — this module only forwards.
"""

from .constants import *  # noqa: F401,F403  (re-export shared constants)
from . import constants as _constants

# Mirror the public constant names so `from dl_utils import X` resolves them even
# though `import *` would normally skip names absent from an `__all__`.
__all__ = [name for name in dir(_constants) if not name.startswith("_")]
