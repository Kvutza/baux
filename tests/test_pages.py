"""Unit test for two-view GitHub Pages bundle verification and generation."""

import tempfile
from pathlib import Path

from admin.ops.publish_pages import assemble_bundle
from admin.ops.verify_pages import verify_bundle

ROOT = Path(__file__).resolve().parent.parent


def test_bundle():
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir)
        assemble_bundle(dest)
        verify_bundle(dest)
