"""Guard that the package version and the installed distribution metadata agree."""

from importlib.metadata import version

import activescan


def test_version_matches_distribution_metadata() -> None:
    assert activescan.__version__ == version("activescan")
