"""Import-contract tests: every aqnet2 module imports with heavy deps HIDDEN.

BUILD_NOTES hard rule: heavy deps (torch, gpboost, statsmodels, lightgbm,
geo libs, ...) sit behind try-import guards with a printed degradation
message, and EVERY module must import cleanly with none of them installed
(the PACE venv lacks pydap and rasterio; local machines may lack all).

The heavy_dep_blocker fixture (conftest.py) installs a meta-path finder
that raises ImportError for every blocked package and purges/restores
sys.modules around each test, so importlib.import_module genuinely
re-executes the module body under the airlock. Any module that fails to
import under the blocker has an import-time heavy dependency — a contract
bug, not an environment issue.

Modules being written in parallel may not exist yet: a genuinely ABSENT
file skips with a warning; a PRESENT-but-broken module FAILS.
"""
import importlib
import os
import warnings

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
AQNET2_DIR = os.path.dirname(_TESTS_DIR)

# The full flat module namespace (INTERFACES.md / BUILD_NOTES assignments).
MODULES = ("config2", "folds2", "fetchers2", "priors", "skeleton",
           "graph_res", "field_res", "compose", "calibrate", "colocate",
           "frame2", "exceed", "uq", "validate2", "pipeline2",
           "pa_v4_ingest", "tune_deep")


def test_blocker_actually_blocks(heavy_dep_blocker):
    """Sanity: the airlock really hides every heavy dep (a blocker that
    silently passes would make the whole suite vacuous)."""
    for name in heavy_dep_blocker:
        with pytest.raises(ImportError):
            importlib.import_module(name)


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports_without_heavy_deps(heavy_dep_blocker, mod):
    path = os.path.join(AQNET2_DIR, mod + ".py")
    if not os.path.exists(path):
        warnings.warn(
            f"{mod}.py not present yet (being written in parallel) — "
            "import contract deferred, re-run once it lands",
            stacklevel=1)
        pytest.skip(f"{mod}.py absent -- parallel build")
    try:
        m = importlib.import_module(mod)
    except (Exception, SystemExit) as e:
        pytest.fail(
            f"{mod} does not import with heavy deps hidden "
            f"({type(e).__name__}: {e}) -- heavy imports must sit behind "
            "try-import guards with a printed degradation message "
            "(BUILD_NOTES hard rule)")
    assert m.__name__ == mod
