"""PurpleAir PM2.5 target corrections for AQNet.

The 'pm25' column in pipeline/purpleair_full_dataset.parquet is the raw
PurpleAir ATM estimate, which reads high relative to regulatory monitors,
increasingly so at high relative humidity. AQNet's default training target is
the U.S.-wide correction of Barkjohn et al. (2021, Atmos. Meas. Tech.
14:4617-4637):

    pm25 = 0.524 * pm25_atm - 0.0862 * RH + 5.75      (clipped at 0)

method="raw" keeps the uncorrected column as the target and is retained only
as a sensitivity option. Rows with missing humidity produce a NaN corrected
target (no RH imputation here); downstream assembly decides how to handle
them.
"""
import numpy as np

# Barkjohn et al. (2021) U.S.-wide linear PM2.5 + RH correction coefficients.
BARKJOHN_SLOPE = 0.524
BARKJOHN_RH_COEF = -0.0862
BARKJOHN_INTERCEPT = 5.75


# ── Correction functions ────────────────────────────────────────────────────

def barkjohn_correct(pm25_atm, humidity):
    """Barkjohn et al. (2021) corrected PM2.5 (ug/m3), clipped at zero.

    Parameters
    ----------
    pm25_atm : array-like
        Raw PurpleAir ATM PM2.5 (ug/m3).
    humidity : array-like
        Relative humidity (%), same length.

    Returns
    -------
    np.ndarray (float64). NaN inputs propagate to NaN outputs.
    """
    pm = np.asarray(pm25_atm, dtype=np.float64)
    rh = np.asarray(humidity, dtype=np.float64)
    corrected = BARKJOHN_SLOPE * pm + BARKJOHN_RH_COEF * rh + BARKJOHN_INTERCEPT
    return np.clip(corrected, 0.0, None)


def apply_target_correction(df, method="barkjohn"):
    """Return a copy of df with a 'target' column added.

    method="barkjohn" (default) applies barkjohn_correct(pm25, humidity);
    method="raw" sets target to the uncorrected pm25 column.
    """
    out = df.copy()
    if method == "barkjohn":
        out["target"] = barkjohn_correct(out["pm25"].to_numpy(),
                                         out["humidity"].to_numpy())
    elif method == "raw":
        out["target"] = out["pm25"].astype(np.float64)
    else:
        raise ValueError(f"Unknown correction method {method!r}; "
                         "expected 'barkjohn' or 'raw'.")
    return out
