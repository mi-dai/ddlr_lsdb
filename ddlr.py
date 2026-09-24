import numpy as np

# Galaxy shape columns per catalog (check against `galaxies.columns` in the notebook)
RUBIN_SHAPE_COLS = {"xx": "shape_xx", "yy": "shape_yy", "xy": "shape_xy"}
LEGACY_SHAPE_COLS = {"r": "SHAPE_R", "e1": "SHAPE_E1", "e2": "SHAPE_E2"}


def shape_from_rubin(xx, yy, xy):
    """Ellipse (a, b, theta) from second moments. Units of a, b follow the moments (TODO: check pixel vs arcsec)."""
    xx, yy, xy = (np.asarray(v, dtype=float) for v in (xx, yy, xy))
    mean = 0.5 * (xx + yy)
    diff = np.sqrt((0.5 * (xx - yy)) ** 2 + xy ** 2)
    a = np.sqrt(mean + diff)
    b = np.sqrt(np.clip(mean - diff, 0, None))
    theta = 0.5 * np.arctan2(2 * xy, xx - yy)
    return a, b, theta


def shape_from_legacy(r, e1, e2):
    """Ellipse (a, b, theta) from Tractor half-light radius (arcsec) and ellipticity components."""
    r, e1, e2 = (np.asarray(v, dtype=float) for v in (r, e1, e2))
    e = np.hypot(e1, e2)
    a = r
    b = r * (1 - e) / (1 + e)
    theta = 0.5 * np.arctan2(e2, e1)
    return a, b, theta


def compute_ddlr(ra_sn, dec_sn, ra_gal, dec_gal, sep_arcsec, a, b, theta):
    """Directional-light-radius-normalized separation between a transient and a candidate host.

    DLR is the radius of the galaxy's light ellipse (a, b, theta) along the direction
    from the galaxy center to the transient; DDLR = separation / DLR.

    TODO: placeholder. Currently uses a circular approximation, DDLR = sep / a.
    Replace with the directional radius, e.g. for position angle phi of the transient
    relative to the galaxy and alpha = phi - theta:
        DLR = a * b / sqrt((a * sin(alpha))**2 + (b * cos(alpha))**2)
    """
    sep_arcsec = np.asarray(sep_arcsec, dtype=float)
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ddlr = sep_arcsec / a
    return ddlr.astype(np.float32)


def add_ddlr(df, galaxy_catalog, cols, gal_suffix):
    """Partition function: add a `ddlr` column to crossmatched transient-galaxy pairs.

    cols: dict with keys ra_sn, dec_sn, ra_gal, dec_gal, sep (crossmatch output column names)
    gal_suffix: suffix lsdb appended to the galaxy columns in the crossmatch
    """
    if len(df) == 0:
        return df.assign(ddlr=np.array([], dtype=np.float32))

    if galaxy_catalog == "rubin":
        a, b, theta = shape_from_rubin(*(df[c + gal_suffix] for c in RUBIN_SHAPE_COLS.values()))
    elif galaxy_catalog == "legacy":
        a, b, theta = shape_from_legacy(*(df[c + gal_suffix] for c in LEGACY_SHAPE_COLS.values()))
    else:
        raise ValueError(f"unknown galaxy_catalog: {galaxy_catalog}")

    ddlr = compute_ddlr(df[cols["ra_sn"]], df[cols["dec_sn"]], df[cols["ra_gal"]], df[cols["dec_gal"]],
                        df[cols["sep"]], a, b, theta)
    return df.assign(ddlr=ddlr)


def select_host(df, id_col, ddlr_max=4.0):
    """Partition function: keep the smallest-DDLR candidate per transient.

    Safe per partition because crossmatch output is partitioned by the left (transient) catalog.
    Adds n_candidates, ddlr_second (next-best DDLR) and has_host (ddlr < ddlr_max).
    Transients with no galaxy inside the crossmatch radius are not in the output.
    """
    df = df.sort_values([id_col, "ddlr"], na_position="last")
    g = df.groupby(id_col, sort=False)
    df = df.assign(
        host_rank=g.cumcount(),
        n_candidates=g["ddlr"].transform("size"),
        ddlr_second=g["ddlr"].shift(-1),
    )
    hosts = df[df["host_rank"] == 0].drop(columns="host_rank")
    hosts = hosts.assign(has_host=hosts["ddlr"] < ddlr_max)
    return hosts.sort_index()
