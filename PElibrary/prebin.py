import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import hyperparameters, filteringSegm


def do_bintest(cnarr, segments=None, alpha=0.005, target_only=False):
    cnarr = cnarr.copy()

    resid = cnarr.residuals(segments)
    if not resid.index.is_unique:
        dup_idx = resid.index.duplicated(keep=False)
        logging.warning("Segments may overlap at %d bins; dropping duplicate values",
                        dup_idx.sum())
        logging.debug("Duplicated indices: %s", " ".join(map(str, resid[dup_idx].head(50))))
        resid = resid[~resid.index.duplicated()]
        cnarr = cnarr.as_dataframe(cnarr.data.loc[resid.index])
    if len(cnarr) != len(resid):
        logging.info("Segments do not cover all bins (%d), only %d of them",
                     len(cnarr), len(resid))
        cnarr = cnarr.as_dataframe(cnarr.data.loc[resid.index])

    cnarr['log2'] = resid

    if target_only:
        offTarget_idx = cnarr['gene'].isin(hyperparameters.OFFTARGET_ALIASES)
        if offTarget_idx.any():
            logging.info("Ignoring %d off-target bins", offTarget_idx.sum())

            cnarr = cnarr[~offTarget_idx]

    cnarr['p_bintest'] = z_prob(cnarr)
    is_sig = cnarr['p_bintest'] < alpha
    logging.info("Significant hits in {}/{} bins ({:.3g}%)"
                 .format(is_sig.sum(), len(is_sig),
                         100 * is_sig.sum() / len(is_sig)))

    hits = cnarr[is_sig]
    return hits


def z_prob(cnarr):
    sd = np.sqrt(1 - cnarr['weight'])

    z = cnarr['log2'] / sd

    p = 2. * norm.cdf(-np.abs(z))

    return p_adjust_bh(p)


def p_adjust_bh(p):
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


def spike_into_segments(cnarr, segments, is_sig):
    if is_sig.any():

        cnarr['is_sig'] = is_sig
        chunks = []
        for segment, seghits in cnarr.by_ranges(segments, keep_empty=True):
            if seghits['is_sig'].any():

                levels = seghits['is_sig'].cumsum() * seghits['is_sig']
                chunks.append(seghits.data
                              .assign(_levels=levels)
                              .groupby('_levels', sort=False)
                              .apply(filteringSegm.squash_region)
                              .reset_index(drop=True))
            else:

                chunks.append(pd.DataFrame.from_records([segment],
                                                        columns=segments.data.columns))
        return cnarr.as_dataframe(pd.concat(chunks,

                                            ))
    else:

        return segments
