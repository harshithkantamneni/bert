"""python -m core.migrations entry point."""

import sys

from . import _cli

if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
