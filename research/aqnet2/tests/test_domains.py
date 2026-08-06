"""Domain-contract tests: config2 under AQNET2_DOMAIN (EXPANSION.md).

The domain switch is env-resolved at import time, and the parent test
process has already imported config2 under the default domain (conftest.py
does, and half the suite holds references to its values), so an in-process
reload would leave a torn namespace behind. Each test therefore spawns a
fresh interpreter with the env it wants and reads the resolved contract
back as JSON — the same isolation argument as the heavy_dep_blocker
airlock, applied to env instead of imports.

What is frozen here:

  * west7 resolves to the v3 artifacts namespace, 8 outer folds, a 30-site
    vault, seven state FIPS, and an AQS path that is either absent or
    west7-stemmed — NEVER the tx parquet (a tx path under west7 would
    silently build seven-state folds from Texas-only sites).
  * The default env stays bit-for-bit the shipped v2 Texas contract
    (v2, 5 outer folds, 12 vault sites, FIPS ["48"]) — the v2 run is
    shipped and its artifacts must remain reproducible.
  * PA stays TX-only in BOTH domains (Phase 1 amendment: no new PurpleAir;
    pa_states is ["48"] everywhere until the owner gate clears).
"""
import json
import os
import subprocess
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
AQNET2_DIR = os.path.dirname(_TESTS_DIR)

# Child prints the env-resolved contract as one JSON line; run with
# cwd=AQNET2_DIR so `import config2` resolves without sys.path surgery.
_PROBE = (
    "import json, config2; "
    "print(json.dumps({"
    "'domain': config2.DOMAIN, "
    "'artifacts_dir': config2.ARTIFACTS_DIR, "
    "'outer_n_folds': config2.OUTER_N_FOLDS, "
    "'vault_n_sites': config2.VAULT_N_SITES, "
    "'state_fips': config2.STATE_FIPS, "
    "'pa_state_fips': config2.PA_STATE_FIPS, "
    "'aqs_stem': config2.AQS_STEM, "
    "'bbox': config2.TX_BBOX, "
    "'aqs_path': config2.canonical_aqs_path()}))"
)


def _resolve_contract(domain=None):
    """Import config2 in a fresh interpreter; return its resolved contract.

    domain=None strips AQNET2_DOMAIN from the child env entirely (the
    default path must not depend on whatever the parent shell exported).
    """
    env = dict(os.environ)
    env.pop("AQNET2_DOMAIN", None)
    if domain is not None:
        env["AQNET2_DOMAIN"] = domain
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=AQNET2_DIR, env=env,
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_west7_domain_contract():
    c = _resolve_contract("west7")
    assert c["domain"] == "west7"
    assert os.path.basename(os.path.normpath(c["artifacts_dir"])) == "v3"
    assert c["outer_n_folds"] == 8
    assert c["vault_n_sites"] == 30
    assert len(c["state_fips"]) == 7
    assert "48" in c["state_fips"]          # TX stays in the seven
    assert c["pa_state_fips"] == ["48"]     # Phase 1: TX archive only
    assert c["aqs_stem"] == "aqs_daily_west7_v3"
    assert c["bbox"]["lon_min"] == -124.8   # Pacific edge, not the TX bbox
    p = c["aqs_path"]
    if p is not None:
        base = os.path.basename(p)
        assert base.startswith("aqs_daily_west7"), base
        assert "aqs_daily_tx" not in base, base


def test_default_domain_is_shipped_tx_contract():
    c = _resolve_contract(None)
    assert c["domain"] == "tx"
    assert os.path.basename(os.path.normpath(c["artifacts_dir"])) == "v2"
    assert c["outer_n_folds"] == 5
    assert c["vault_n_sites"] == 12
    assert c["state_fips"] == ["48"]
    assert c["pa_state_fips"] == ["48"]
    assert c["aqs_stem"] == "aqs_daily_tx_v2"
    # Frozen v2 bbox — the shipped run's geometry, digit for digit.
    assert c["bbox"] == {"lat_min": 25.6, "lat_max": 36.7,
                         "lon_min": -107.0, "lon_max": -93.3}
    p = c["aqs_path"]
    if p is not None:
        base = os.path.basename(p)
        assert base.startswith("aqs_daily_tx"), base
        assert "west7" not in base, base
