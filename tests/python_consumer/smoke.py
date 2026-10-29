from __future__ import annotations

import baux
import baux.experimental
import baux.experimental.multi_region

assert baux.__file__
assert baux.experimental.__file__
assert baux.experimental.multi_region.__file__

try:
    import baux.baux_rust
except ImportError:
    print("baux wheel consumer: skipped (baux_rust unavailable)")
else:
    assert baux.baux_rust.__file__
    print("baux wheel consumer: ok")
