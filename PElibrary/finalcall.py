import logging

import numpy as np
import pandas as pd

from . import filteringSegm


def do_finalcall(cnarr, variants=None, method="threshold", ploidy=2, purity=None,
                 is_reference_male=False, is_sample_female=False, filters=None,
                 thresholds=(-1.1, -0.25, 0.2, 0.7)):
    if method not in ("threshold", "clonal", "none"):
        raise ValueError("Argument `method` must be one of: clonal, threshold")

    outarr = cnarr.copy()
    if filters:

        for filt in ('ci', 'sem'):
            if filt in filters:
                outarr = getattr(filteringSegm, filt)(outarr)
                filters.remove(filt)

    if variants:
        outarr["baf"] = variants.baf_by_ranges(outarr)

    if purity and purity < 1.0:
        absolutes = absolute_clonal(outarr, ploidy, purity,
                                    is_reference_male, is_sample_female)

        outarr["log2"] = log2_ratios(outarr, absolutes, ploidy,
                                     is_reference_male)
        if variants:
            outarr["baf"] = rescale_baf(purity, outarr["baf"])
    elif method == "clonal":

        absolutes = absolute_pure(outarr, ploidy, is_reference_male)

    if method == "threshold":
        tokens = ["%g => %d" % (thr, i) for i, thr in enumerate(thresholds)]

        absolutes = absolute_threshold(outarr, ploidy, thresholds,
                                       is_reference_male)

    if method != 'none':
        outarr['cn'] = absolutes.round().astype('int')
        if 'baf' in outarr:
            upper_baf = ((outarr['baf'] - .5).abs() + .5).fillna(1.0).values
            outarr['cn1'] = ((absolutes * upper_baf).round()
                             .clip(0, outarr['cn'])
                             .astype('int'))
            outarr['cn2'] = outarr['cn'] - outarr['cn1']
            is_null = (outarr['baf'].isnull() & (outarr['cn'] > 0))
            outarr[is_null, 'cn1'] = np.nan
            outarr[is_null, 'cn2'] = np.nan

    if filters:

        for filt in filters:
            if not outarr.data.index.is_unique:
                outarr.data = outarr.data.reset_index(drop=True)
            outarr = getattr(filteringSegm, filt)(outarr)

    outarr.sort_columns()
    return outarr


def log2_ratios(cnarr, absolutes, ploidy, is_reference_male,
                min_abs_val=1e-3, round_to_int=False):
    if round_to_int:
        absolutes = absolutes.round()

    ratios = np.log2(np.maximum(absolutes / ploidy, min_abs_val))

    if is_reference_male:
        ratios[(cnarr.chromosome == cnarr._chr_x_label).values] += 1.0
    ratios[(cnarr.chromosome == cnarr._chr_y_label).values] += 1.0
    return ratios


def absolute_threshold(cnarr, ploidy, thresholds, is_reference_male):
    absolutes = np.zeros(len(cnarr), dtype=np.float_)
    for idx, row in enumerate(cnarr):
        ref_copies = _reference_copies_pure(row.chromosome, ploidy,
                                            is_reference_male)
        if np.isnan(row.log2):
            absolutes[idx] = ref_copies
            continue
        cnum = 0
        for cnum, thresh in enumerate(thresholds):
            if row.log2 <= thresh:
                if ref_copies != ploidy:
                    cnum = int(cnum * ref_copies / ploidy)
                break
        else:
            cnum = int(np.ceil(_log2_ratio_to_absolute_pure(row.log2,
                                                            ref_copies)))
        absolutes[idx] = cnum
    return absolutes


def absolute_clonal(cnarr, ploidy, purity, is_reference_male, is_sample_female):
    absolutes = np.zeros(len(cnarr), dtype=np.float_)
    for i, row in enumerate(cnarr):
        ref_copies, expect_copies = _reference_expect_copies(
            row.chromosome, ploidy, is_sample_female, is_reference_male)
        absolutes[i] = _log2_ratio_to_absolute(
            row.log2, ref_copies, expect_copies, purity)
    return absolutes


def absolute_pure(cnarr, ploidy, is_reference_male):
    absolutes = np.zeros(len(cnarr), dtype=np.float_)
    for i, row in enumerate(cnarr):
        ref_copies = _reference_copies_pure(row.chromosome, ploidy,
                                            is_reference_male)
        absolutes[i] = _log2_ratio_to_absolute_pure(row.log2, ref_copies)
    return absolutes


def absolute_dataframe(cnarr, ploidy, purity, is_reference_male, is_sample_female):
    absolutes = np.zeros(len(cnarr), dtype=np.float_)
    reference_copies = expect_copies = np.zeros(len(cnarr), dtype=np.int_)
    for i, row in enumerate(cnarr):
        ref_copies, exp_copies = _reference_expect_copies(
            row.chromosome, ploidy, is_sample_female, is_reference_male)
        reference_copies[i] = ref_copies
        expect_copies[i] = exp_copies
        absolutes[i] = _log2_ratio_to_absolute(
            row.log2, ref_copies, exp_copies, purity)
    return pd.DataFrame({'absolute': absolutes,
                         'reference': reference_copies,
                         'expect': expect_copies})


def absolute_expect(cnarr, ploidy, is_sample_female):
    exp_copies = np.repeat(ploidy, len(cnarr))
    is_y = (cnarr.chromosome == cnarr._chr_y_label).values
    if is_sample_female:
        exp_copies[is_y] = 0
    else:
        is_x = (cnarr.chromosome == cnarr._chr_x_label).values
        exp_copies[is_x | is_y] = ploidy // 2
    return exp_copies


def absolute_reference(cnarr, ploidy, is_reference_male):
    ref_copies = np.repeat(ploidy, len(cnarr))
    is_x = (cnarr.chromosome == cnarr._chr_x_label).values
    is_y = (cnarr.chromosome == cnarr._chr_y_label).values
    if is_reference_male:
        ref_copies[is_x] = ploidy // 2
    ref_copies[is_y] = ploidy // 2
    return ref_copies


def _reference_expect_copies(chrom, ploidy, is_sample_female, is_reference_male):
    chrom = chrom.lower()
    if chrom in ["chrx", "x"]:
        ref_copies = (ploidy // 2 if is_reference_male else ploidy)
        exp_copies = (ploidy if is_sample_female else ploidy // 2)
    elif chrom in ["chry", "y"]:
        ref_copies = ploidy // 2
        exp_copies = (0 if is_sample_female else ploidy // 2)
    else:
        ref_copies = exp_copies = ploidy
    return ref_copies, exp_copies


def _reference_copies_pure(chrom, ploidy, is_reference_male):
    chrom = chrom.lower()
    if chrom in ["chry", "y"] or (is_reference_male and chrom in ["chrx", "x"]):
        ref_copies = ploidy // 2
    else:
        ref_copies = ploidy
    return ref_copies


def _log2_ratio_to_absolute(log2_ratio, ref_copies, expect_copies, purity=None):
    if purity and purity < 1.0:
        ncopies = (ref_copies * 2 ** log2_ratio - expect_copies * (1 - purity)
                   ) / purity
    else:
        ncopies = _log2_ratio_to_absolute_pure(log2_ratio, ref_copies)
    return ncopies


def _log2_ratio_to_absolute_pure(log2_ratio, ref_copies):
    ncopies = ref_copies * 2 ** log2_ratio
    return ncopies


def rescale_baf(purity, observed_baf, normal_baf=0.5):
    tumor_baf = (observed_baf - normal_baf * (1 - purity)) / purity

    return tumor_baf
