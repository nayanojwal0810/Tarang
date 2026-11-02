"""Legacy CLI entry-point delegating to scripts.prepare_data."""

from __future__ import annotations

import sys
from scripts.prepare_data import main

if __name__ == "__main__":
    sys.exit(main())
