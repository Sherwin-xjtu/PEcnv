import numpy as np
import pandas as pd

from . import measures


def do_metrics(cnarrs, segments=None, skip_low=False):
    from .cnv import CopyNumArray as CNA
    if isinstance(cnarrs, CNA):
        cnarrs = [cnarrs]
    if isinstance(segments, CNA):
        segments = [segments]
    elif segments is None:
        segments = [None]
    else:
        segments = list(segments)
    if skip_low:
        cnarrs = (cna.drop_low_coverage() for cna in cnarrs)
    rows = ((cna.meta.get("filename", cna.sample_id),
             len(seg) if seg is not None else '-'
             ) + ests_of_scale(cna.residuals(seg).values)
            for cna, seg in zip_repeater(cnarrs, segments))
    colnames = ["sample", "segments", "stdev", "mad", "iqr", "bivar"]
    return pd.DataFrame.from_records(rows, columns=colnames)


def zip_repeater(iterable, repeatable):
    rpt_len = len(repeatable)
    if rpt_len == 1:
        rpt = repeatable[0]
        for it in iterable:
            yield it, rpt
    else:
        i = -1
        for i, (it, rpt) in enumerate(zip(iterable, repeatable)):
            yield it, rpt

        if i + 1 != rpt_len:
            raise ValueError("""Number of unsegmented and segmented input files
                             did not match (%d vs. %d)""" % (i, rpt_len))


def ests_of_scale(deviations):
    std = np.std(deviations, dtype=np.float64)
    mad = measures.median_absolute_deviation(deviations)
    iqr = measures.interquartile_range(deviations)
    biw = measures.biweight_midvariance(deviations)
    return (std, mad, iqr, biw)
