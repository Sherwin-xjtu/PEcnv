"""Segmentation of copy number values."""
import locale
import logging
import math
import os.path
import tempfile
from io import StringIO

import numpy as np
import pandas as pd
from skgenome import tabio
from skgenome import GenomicArray
from skgenome.intersect import iter_slices

from .. import core, parallel, params, smoothing, vary
from ..cnary import CopyNumArray as CNA
from ..segfilters import squash_by_groups
from . import cbs, flasso, haar, hmm, none

SEGMENT_METHODS = ('cbs', 'flasso', 'haar', 'none',
                   'hmm', 'hmm-tumor', 'hmm-germline')


def do_segmentation(cnarr, method, threshold=None, variants=None,
                    skip_low=False, skip_outliers=10, min_weight=0,
                    save_dataframe=False, rscript_path="Rscript",
                    processes=1, smooth_cbs=False):
    """Infer copy number segments from the given coverage table."""
    if method not in SEGMENT_METHODS:
        raise ValueError("'method' must be one of: "
                         + ", ".join(SEGMENT_METHODS)
                         + "; got: " + repr(method))

    if not threshold:
        threshold = {'cbs': 0.0001,
                     'flasso': 0.0001,
                     'haar': 0.0001,
                    }.get(method)
    msg = "Segmenting with method " + repr(method)
    if threshold is not None:
        if method.startswith('hmm'):
            msg += ", smoothing window size %s," % threshold
        else:
            msg += ", significance threshold %s," % threshold
    msg += " in %s processes" % processes
    logging.info(msg)

    # NB: parallel cghFLasso segfaults in R ('memory not mapped'),
    # even when run on a single chromosome
    if method == 'flasso' or method.startswith('hmm'):
        # ENH segment p/q arms separately
        # -> assign separate identifiers via chrom name suffix?
        cna = _do_segmentation(cnarr, method, threshold, variants, skip_low,
                               skip_outliers, min_weight, save_dataframe,
                               rscript_path)
        if save_dataframe:
            cna, rstr = cna
            rstr = _to_str(rstr)

    else:
        with parallel.pick_pool(processes) as pool:
            rets = list(pool.map(_ds, ((ca, method, threshold, variants,
                                        skip_low, skip_outliers, min_weight,
                                        save_dataframe, rscript_path, smooth_cbs)
                                       for _, ca in cnarr.by_arm())))
        if save_dataframe:
            # rets is a list of (CNA, R dataframe string) -- unpack
            rets, r_dframe_strings = zip(*rets)
            # Strip the header line from all but the first dataframe, then combine
            r_dframe_strings = map(_to_str, r_dframe_strings)
            rstr = [next(r_dframe_strings)]
            rstr.extend(r[r.index('\n') + 1:] for r in r_dframe_strings)
            rstr = "".join(rstr)
        cna = cnarr.concat(rets)

    cna.sort_columns()
    if save_dataframe:
        return cna, rstr
    return cna


def _to_str(s, enc=locale.getpreferredencoding()):
    if isinstance(s, bytes):
        return s.decode(enc)
    return s


def _ds(args):
    """Wrapper for parallel map"""
    return _do_segmentation(*args)


def merge_segs(nsegs):
    newSegs = []
    tm = []
    for li in nsegs:
        if len(tm) == 0:
            tm = li
            newSegs.append(tm)
        else:
            if li[0] <= tm[1]:
                tm = [tm[0],li[1]]
                del(newSegs[-1])
                newSegs.append(tm)
            else:
                newSegs.append(li)
                tm = li
    return newSegs


def EWMA_SEG(breakpointSelects):
    csegs = []
    start = breakpointSelects[0]
    itm = 1
    edgeSize = 2
    for i in range(len(breakpointSelects)):
        if i + 1 < len(breakpointSelects):
            if itm == 1:
                start = breakpointSelects[i]
            itm += 1
            if breakpointSelects[i + 1] - breakpointSelects[i] < edgeSize:  #20 best
                if i + 1 == len(breakpointSelects) - 1:
                    if start - edgeSize > 0:
                        csegs.append([start - edgeSize, breakpointSelects[i + 1] + edgeSize])
                    else:
                        csegs.append([start, breakpointSelects[i + 1]])
            elif breakpointSelects[i + 1] - breakpointSelects[i] > edgeSize:
                end = breakpointSelects[i]
                if start - edgeSize < 0:
                    csegs.append([start, end + edgeSize])
                else:
                    csegs.append([start - edgeSize, end + edgeSize])
                itm = 1
    return csegs


def breakpoint_select(dfArr, sp, upline, dowline):
    index = 0
    breakpoints = []
    for i in dfArr.ewm(span=sp).mean():
        if i > upline or i < dowline:
            breakpoints.append(index)
        index += 1
    return breakpoints


def EWMA_model(arr):
    sp0 = 9
    mu0 = arr.mean()
    num0 = float(2 / (1 + sp0))
    std0 = arr.std()
    upline0 = mu0 + std0 * 3 * math.sqrt(num0 / (2 - num0))
    dowline0 = mu0 - std0 * 3 * math.sqrt(num0 / (2 - num0))
    breakpointSelects0 = breakpoint_select(arr, sp0, upline0, dowline0)

    arr_ = arr.iloc[::-1]
    sp1 = 9
    num1 = float(2 / (1 + sp1))
    mu1 = arr_.mean()
    num1 = float(2 / (1 + sp1))
    std1 = arr_.std()
    upline1 = mu1 + std1 * 3 * math.sqrt(num1 / (2 - num1))
    dowline1 = mu1 - std1 * 3 * math.sqrt(num1 / (2 - num1))

    breakpointSelects1 = breakpoint_select(arr_, sp1, upline1, dowline1)
    breakpointSelects2 = []
    for r in breakpointSelects1[::-1]:
        breakpointSelects2.append(np.abs(r-(len(arr)-1)))
    breakpointSelects = sorted(list(set(breakpointSelects0+breakpointSelects2)))
    if len(breakpointSelects) > 0:
        msegs = EWMA_SEG(breakpointSelects)
        return msegs
    else:
        return[[0,len(arr)-1]]



def EWMA(I):
    if len(I) > 1:
        #seq_data_arr = pd.DataFrame(I)
        segs = EWMA_model(I)
        segs = merge_segs(segs)
    else:
        segs = []
    return segs

def _do_segmentation(cnarr, method, threshold, variants=None,
                     skip_low=False, skip_outliers=10, min_weight=0,
                     save_dataframe=False,
                     rscript_path="Rscript", smooth_cbs=False):
    """Infer copy number segments from the given coverage table."""
    if not len(cnarr):
        return cnarr

    filtered_cn = cnarr.copy()
    # Filter out bins with no or near-zero sequencing coverage
    if skip_low:
        filtered_cn = filtered_cn.drop_low_coverage(verbose=False)
    # Filter by distance from rolling quantiles
    if skip_outliers:
        filtered_cn = drop_outliers(filtered_cn, 50, skip_outliers)
    # Filter by bin weights
    if min_weight:
        weight_too_low = (filtered_cn["weight"] < min_weight).fillna(True)
    else:
        weight_too_low = (filtered_cn["weight"] == 0).fillna(True)
    n_weight_too_low = weight_too_low.sum() if len(weight_too_low) else 0
    if n_weight_too_low:
        filtered_cn = filtered_cn[~weight_too_low]
        if min_weight:
            logging.debug("Dropped %d bins with weight below %s",
                          n_weight_too_low, min_weight)
        else:
            logging.debug("Dropped %d bins with zero weight",
                          n_weight_too_low)

    if len(filtered_cn) != len(cnarr):
        msg = ("Dropped %d / %d bins"
               % (len(cnarr) - len(filtered_cn), len(cnarr)))
        if cnarr["chromosome"].iat[0] == cnarr["chromosome"].iat[-1]:
            msg += " on chromosome " + str(cnarr["chromosome"].iat[0])
        logging.info(msg)
    if not len(filtered_cn):
        return filtered_cn

    seg_out = ""
    if method == 'haar':
        segarr = haar.segment_haar(filtered_cn, threshold)

    elif method == 'none':
        segarr = none.segment_none(filtered_cn)

    elif method.startswith('hmm'):
        segarr = hmm.segment_hmm(filtered_cn, method, threshold, variants)

    elif method in ('cbs', 'flasso'):
        # Run R scripts to calculate copy number segments
        rscript = {'cbs': cbs.CBS_RSCRIPT,
                   'flasso': flasso.FLASSO_RSCRIPT,
                  }[method]
        segsEWMA = EWMA(filtered_cn['log2'])
        segsNew = [x for x in segsEWMA if x != []]
        #print(filtered_cn[1:3]['log2'])
        filtered_cn['start'] += 1  # Convert to 1-indexed coordinates for R
        segarrList = []
        if segsNew == []:
            with tempfile.NamedTemporaryFile(suffix='.cnr', mode="w+t") as tmp:
                # TODO tabio.write(filtered_cn, tmp, 'seg')
                filtered_cn.data.to_csv(tmp, index=False, sep='\t',
                                        float_format='%.6g', mode="w+t")
                tmp.flush() 
                script_strings = {
                    'probes_fname': tmp.name,
                    'sample_id': cnarr.sample_id,
                    'threshold': threshold,
                    'smooth_cbs': smooth_cbs
                }
                with core.temp_write_text(rscript % script_strings,
                                          mode='w+t') as script_fname:
                    seg_out = core.call_quiet(rscript_path,
                                              "--no-restore",
                                              "--no-environ",
                                              script_fname)
            # Convert R dataframe contents (SEG) to a proper CopyNumArray
            # NB: Automatically shifts 'start' back from 1- to 0-indexed
            segarr = tabio.read(StringIO(seg_out.decode()), "seg", into=CNA)
        else:
            for seg in segsNew:
            
                with tempfile.NamedTemporaryFile(suffix='.cnr', mode="w+t") as tmp:
                    # TODO tabio.write(filtered_cn, tmp, 'seg')
                    filtered_cn[seg[0]:seg[1]].data.to_csv(tmp, index=False, sep='\t',float_format='%.6g', mode="w+t")
                    tmp.flush()
                    script_strings = {
                        'probes_fname': tmp.name,
                        'sample_id': cnarr.sample_id,
                        'threshold': threshold,
                        'smooth_cbs': smooth_cbs
                   }
                    with core.temp_write_text(rscript % script_strings, mode='w+t') as script_fname:
                        seg_out = core.call_quiet(rscript_path,
                                                  "--no-restore",
                                                  "--no-environ",
                                                  script_fname)
                # Convert R dataframe contents (SEG) to a proper CopyNumArray
                # NB: Automatically shifts 'start' back from 1- to 0-indexed
                sga = tabio.read(StringIO(seg_out.decode()), "seg", into=CNA)
                segarrList.append(sga)
            segli = [x for x in segarrList if x != []] 
            segarr = cnarr.concat(segli)
        if method == 'flasso':
            # Merge adjacent bins with same log2 value into segments
            if 'weight' in filtered_cn:
                segarr['weight'] = filtered_cn['weight']
            else:
                segarr['weight'] = 1.0
            segarr = squash_by_groups(segarr, segarr['log2'], by_arm=True)

    else:
        raise ValueError("Unknown method %r" % method)

    segarr.meta = cnarr.meta.copy()
    if variants and not method.startswith('hmm'):
        # Re-segment the variant allele freqs within each segment
        # TODO train on all segments together
        logging.info("Re-segmenting on variant allele frequency")
        newsegs = [hmm.variants_in_segment(subvarr, segment)
                   for segment, subvarr in variants.by_ranges(segarr)]
        segarr = segarr.as_dataframe(pd.concat(newsegs))
        segarr['baf'] = variants.baf_by_ranges(segarr)

    segarr = transfer_fields(segarr, cnarr)
    if save_dataframe:
        return segarr, seg_out
    else:
        return segarr


def drop_outliers(cnarr, width, factor):
    """Drop outlier bins with log2 ratios too far from the trend line.

    Outliers are the log2 values `factor` times the 90th quantile of absolute
    deviations from the rolling average, within a window of given `width`. The
    90th quantile is about 1.97 standard deviations if the log2 values are
    Gaussian, so this is similar to calling outliers `factor` * 1.97 standard
    deviations from the rolling mean. For a window size of 50, the breakdown
    point is 2.5 outliers within a window, which is plenty robust for our needs.
    """
    if not len(cnarr):
        return cnarr
    outlier_mask = np.concatenate([
        smoothing.rolling_outlier_quantile(subarr['log2'], width, .95, factor)
        for _chrom, subarr in cnarr.by_chromosome()])
    n_outliers = outlier_mask.sum()
    if n_outliers:
        logging.info("Dropped %d outlier bins:\n%s%s",
                     n_outliers,
                     cnarr[outlier_mask].data.head(20),
                     "\n..." if n_outliers > 20 else "")
    return cnarr[~outlier_mask]


def transfer_fields(segments, cnarr, ignore=params.IGNORE_GENE_NAMES):
    """Map gene names, weights, depths from `cnarr` bins to `segarr` segments.

    Segment gene name is the comma-separated list of bin gene names. Segment
    weight is the sum of bin weights, and depth is the (weighted) mean of bin
    depths.

    Also: Post-process segmentation output.

    1. Ensure every chromosome has at least one segment.
    2. Ensure first and last segment ends match 1st/last bin ends
       (but keep log2 as-is).

    """
    def make_null_segment(chrom, orig_start, orig_end):
        """Closes over 'segments'."""
        vals = {'chromosome': chrom,
                'start': orig_start,
                'end': orig_end,
                'gene': '-',
                'depth': 0.0,
                'log2': 0.0,
                'probes': 0.0,
                'weight': 0.0,
               }
        row_vals = tuple(vals[c] for c in segments.data.columns)
        return row_vals

    if not len(cnarr):
        # This Should Never Happen (TM)
        # raise RuntimeError("No bins for:\n" + str(segments.data))
        logging.warn("No bins for:\n%s", segments.data)
        return segments

    # Adjust segment endpoints to cover the chromosome arm's original bins
    # (Stretch first and last segment endpoints to match first/last bins)
    bins_chrom = cnarr.chromosome.iat[0]
    bins_start = cnarr.start.iat[0]
    bins_end = cnarr.end.iat[-1]
    if not len(segments):
        # All bins in this chromosome arm were dropped: make a dummy segment
        return make_null_segment(bins_chrom, bins_start, bins_end)
    segments.start.iat[0] = bins_start
    segments.end.iat[-1] = bins_end

    # Aggregate segment depths, weights, gene names
    # ENH refactor so that np/CNA.data access is encapsulated in skgenome
    ignore += params.ANTITARGET_ALIASES
    assert bins_chrom == segments.chromosome.iat[0]
    cdata = cnarr.data.reset_index()
    if 'depth' not in cdata.columns:
        cdata['depth'] = np.exp2(cnarr['log2'].values)
    bin_genes = cdata['gene'].values
    bin_weights = cdata['weight'].values if 'weight' in cdata.columns else None
    bin_depths = cdata['depth'].values
    seg_genes = ['-'] * len(segments)
    seg_weights = np.zeros(len(segments))
    seg_depths = np.zeros(len(segments))

    for i, bin_idx in enumerate(iter_slices(cdata, segments.data, 'outer', False)):
        if bin_weights is not None:
            seg_wt = bin_weights[bin_idx].sum()
            if seg_wt > 0:
                seg_dp = np.average(bin_depths[bin_idx],
                                    weights=bin_weights[bin_idx])
            else:
                seg_dp = 0.0
        else:
            bin_count = len(cdata.iloc[bin_idx])
            seg_wt = float(bin_count)
            seg_dp = bin_depths[bin_idx].mean()
        subgenes = [g for g in pd.unique(bin_genes[bin_idx]) if g not in ignore]
        if subgenes:
            seg_gn = ",".join(subgenes)
        else:
            seg_gn = '-'
        seg_genes[i] = seg_gn
        seg_weights[i] = seg_wt
        seg_depths[i] = seg_dp

    segments.data = segments.data.assign(
        gene=seg_genes,
        weight=seg_weights,
        depth=seg_depths)
    return segments
