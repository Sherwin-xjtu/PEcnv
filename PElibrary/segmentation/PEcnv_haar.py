"""Probe segmentation by convolving with the Haar wavelet.

The basic HaarSeg algorithm:

* Apply the undecimated discrete wavelet transform (UDWT) on the data, using the
  Haar wavelet.
* Select a set of detail subbands from the transform {LMIN, LMIN+1, ..., LMAX}.
* Find the local maxima of the selected detail subbands.
* Threshold the maxima of each subband separately, using an FDR thresholding
  procedure.
* Unify selected maxima from all the subbands to create a list of significant
  breakpoints in the data.
* Reconstruct the segmentation result from the list of significant breakpoints.

HaarSeg segmentation is based on detecting local maxima in the wavelet domain,
using Haar wavelet. The main algorithm parameter is breaksFdrQ, which controls
the sensitivity of the segmentation result. This function supports the optional
use of weights (also known as quality of measurments) and raw measurments. We
recommend using both extentions where possible, as it greatly improves the
segmentation result.

"""
#!/usr/bin/python
# coding=utf-8
from __future__ import division
import logging
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm

import numpy as np
import pandas as pd
from scipy import stats


def segment_haar(cnarr, fdr_q):
    """Do segmentation for pecnv.

    Calculate copy number segmentation by HaarSeg
    (http://haarseg.r-forge.r-project.org/)

    Parameters
    ----------
    cnarr : CopyNumArray
        Binned, normalized copy ratios.
    fdr_q : float
        False discovery rate q-value.

    Returns
    -------
    CopyNumArray
        The CBS data table as a pecnv object.
    """
    # Segment each chromosome individually
    # ENH - skip large gaps (segment chrom. arms separately)
    chrom_tables = [one_chrom(subprobes, fdr_q, chrom)
                    for chrom, subprobes in cnarr.by_arm()]
    #if isinstance(chrom_tables[0],pd.DataFrame):
     #   pass
    #else:
    #    print(chrom_tables)
    #    print('ccccc')
    #if isinstance(chrom_tables[0],pd.DataFrame):
    segarr = cnarr.as_dataframe(pd.concat(chrom_tables))
    segarr.sort_columns()
    return segarr


def one_chrom(cnarr, fdr_q, chrom):
    logging.debug("Segmenting %s", chrom)
    # results = haarSeg(cnarr.smooth_log2(),
    #                   fdr_q,
    #                   W=(cnarr['weight'].values if 'weight' in cnarr
    #                      else None))
    #results = EWMA_haar(np.array(cnarr.log2),
    results = EWMA_haar(cnarr.smooth_log2(), fdr_q, W=(cnarr['weight'].values if 'weight' in cnarr else None))
    #print(len(results['start']),len(results['end']),len(results['size']),len(results['mean'])) 
    table = pd.DataFrame({
        'chromosome': chrom,    
        'start': cnarr['start'].values.take(results['start']),
        'end': cnarr['end'].values.take(results['end']),
        'log2': results['mean'],
        'gene': '-',
         'probes': results['size'],
    })
    return table

def variants_in_segment(varr, segment, fdr_q):
    if len(varr):
        values = varr.mirrored_baf(above_half=True, tumor_boost=True)
        results = haarSeg(values, fdr_q,
                          W=None)  # ENH weight by sqrt(DP)
    else:
        values = pd.Series()
        results = None
    if results is not None and len(results['start']) > 1:
        logging.info("Segmented on allele freqs in %s:%d-%d",
                     segment.chromosome, segment.start, segment.end)
        # Ensure breakpoint locations make sense
        # - Keep original segment start, end positions
        # - Place breakpoints midway between SNVs, I guess?
        # NB: 'results' are indices, i.e. enumerated bins
        gap_rights = varr['start'].values.take(results['start'][1:])
        gap_lefts = varr['end'].values.take(results['end'][:-1])
        mid_breakpoints = (gap_lefts + gap_rights) // 2
        starts = np.concatenate([[segment.start], mid_breakpoints])
        ends = np.concatenate([mid_breakpoints, [segment.end]])
        table = pd.DataFrame({
            'chromosome': segment.chromosome,
            'start': starts,
            'end': ends,
            # 'baf': results['mean'],
            'gene': segment.gene, # '-'
            'log2': segment.log2,
            'probes': results['size'],
            # 'weight': (segment.weight * results['size']
            #            / (segment.end - segment.start)),
        })
    else:
        table = pd.DataFrame({
            'chromosome': segment.chromosome,
            'start': segment.start,
            'end': segment.end,
            # 'baf': values.median(),
            'gene': segment.gene, #'-',
            'log2': segment.log2,
            'probes': segment.probes,
            # 'weight': segment.weight,
        }, index=[0])

    return table



# ---- from HaarSeg R code -- the API ----

def haarSeg(I, breaksFdrQ,
            W=None,
            rawI=None,
            haarStartLevel=1,
            haarEndLevel=5):
    r"""Perform segmentation according to the HaarSeg algorithm.

    Parameters
    ----------
    I : array
        A 1D array of log-ratio values, sorted according to their genomic
        location.
    W : array
        Weight matrix, corresponding to quality of measurement, with values
        :math:`1/(\sigma^2)`. Must have the same size as I.
    rawI : array
        The minimum between the raw test-sample and control-sample coverages
        (before applying log ratio, but after any background reduction and/or
        normalization). These raw red / green measurments are used to detect
        low-value probes, which are more sensitive to noise.
        Used for the non-stationary variance compensation.
        Must have the same size as I.
    breaksFdrQ : float
        The FDR q parameter. This value should lie between 0 and 0.5. The
        smaller this value is, the less sensitive the segmentation result will
        be.
        For example, we will detect fewer segmentation breaks when using Q =
        1e-4, compared to when using Q = 1e-3.
        Common used values are 1e-2, 1e-3, 1e-4.
    haarStartLevel : int
        The detail subband from which we start to detect peaks. The higher this
        value is, the less sensitive we are to short segments. The default is
        value is 1, corresponding to segments of 2 probes.
    haarEndLevel : int
        The detail subband until which we use to detect peaks. The higher this
        value is, the more sensitive we are to large trends in the data. This
        value DOES NOT indicate the largest possible segment that can be
        detected.  The default is value is 5, corresponding to step of 32 probes
        in each direction.

    Returns
    -------
    dict

    Source: haarSeg.R
    """
    def med_abs_diff(diff_vals):
        """Median absolute deviation, with deviations given."""
        if len(diff_vals) == 0:
            return 0.
        return diff_vals.abs().median() * 1.4826

    diffI = pd.Series(HaarConv(I, None, 1))
    if rawI:
        # Non-stationary variance empirical threshold set to 50
        NSV_TH = 50
        varMask = (rawI < NSV_TH)
        pulseSize = 2
        diffMask = (PulseConv(varMask, pulseSize) >= .5)
        peakSigmaEst = med_abs_diff(diffI[~diffMask])
        noisySigmaEst = med_abs_diff(diffI[diffMask])
    else:
        peakSigmaEst = med_abs_diff(diffI)

    breakpoints = np.array([], dtype=np.int_)
    for level in range(haarStartLevel, haarEndLevel+1):
        stepHalfSize = 2 ** level
        convRes = HaarConv(I, W, stepHalfSize)
        peakLoc = FindLocalPeaks(convRes)
        logging.debug("Found %d peaks at level %d", len(peakLoc), level)

        if rawI:
            pulseSize = 2 * stepHalfSize
            convMask = (PulseConv(varMask, pulseSize) >= .5)
            sigmaEst = (1 - convMask) * peakSigmaEst + convMask * noisySigmaEst
            convRes /= sigmaEst
            peakSigmaEst = 1.

        T = FDRThres(convRes[peakLoc], breaksFdrQ, peakSigmaEst)
        # Keep only the peak values where the signal amplitude is large enough.
        addonPeaks = np.extract(np.abs(convRes.take(peakLoc)) >= T, peakLoc)
        breakpoints = UnifyLevels(breakpoints, addonPeaks, 2 ** (level - 1))

    logging.debug("Found %d breakpoints: %s", len(breakpoints), breakpoints)

    # Translate breakpoints to segments
    segs = SegmentByPeaks(I, breakpoints, W)
    segSt = np.insert(breakpoints, 0, 0)
    segEd = np.append(breakpoints, len(I))
    return {'start': segSt,
            'end': segEd - 1,
            'size': segEd - segSt,
            'mean': segs[segSt]}


def FDRThres(x, q, stdev):
    """False discovery rate (FDR) threshold."""
    M = len(x)
    if M < 2:
        return 0

    m = np.arange(1, M+1) / M
    x_sorted = np.sort(np.abs(x))[::-1]
    p = 2 * (1 - stats.norm.cdf(x_sorted, stdev))  # like R "pnorm"
    # Get the largest index for which p <= m*q
    indices = np.nonzero(p <= m * q)[0]
    if len(indices):
        T = x_sorted[indices[-1]]
    else:
        logging.debug("No passing p-values: min p=%.4g, min m=%.4g, q=%s",
                      p[0], m[0], q)
        T = x_sorted[0] + 1e-16  # ~= 2^-52, like MATLAB "eps"
    return T


def SegmentByPeaks(data, peaks, weights=None):
    """Average the values of the probes within each segment.

    Parameters
    ----------
    data : array
        the probe array values
    peaks : array
        Positions of copy number breakpoints in the original array

    Source: SegmentByPeaks.R
    """
    segs = np.zeros_like(data)
    for seg_start, seg_end in zip(np.insert(peaks, 0, 0),
                                  np.append(peaks, len(data))):
        if weights is not None and weights[seg_start:seg_end].sum() > 0:
            # Weighted mean of individual probe values
            val = np.average(data[seg_start:seg_end],
                             weights=weights[seg_start:seg_end])
        else:
            # Unweighted mean of individual probe values
            val = np.mean(data[seg_start:seg_end])
        segs[seg_start:seg_end] = val
    return segs



# ---- from HaarSeg C code -- the core ----

# --- HaarSeg.h
def HaarConv(signal, #const double * signal,
             weight, #const double * weight,
             stepHalfSize, #int stepHalfSize,
            ):
    """Convolve haar wavelet function with a signal, applying circular padding.

    Parameters
    ----------
    signal : const array of floats
    weight : const array of floats (optional)
    stepHalfSize : int

    Returns
    -------
    array
        Of floats, representing the convolved signal.

    Source: HaarSeg.c
    """
    signalSize = len(signal)
    if stepHalfSize > signalSize:
        # XXX TODO handle this endcase
        # raise ValueError("stepHalfSize (%s) > signalSize (%s)"
        #                  % (stepHalfSize, signalSize))
        logging.debug("Error?: stepHalfSize (%s) > signalSize (%s)",
                      stepHalfSize, signalSize)
        return np.zeros(signalSize, dtype=np.float_)

    result = np.zeros(signalSize, dtype=np.float_)
    if weight is not None:
        # Init weight sums
        highWeightSum = weight[:stepHalfSize].sum()
        # highSquareSum = np.exp2(weight[:stepHalfSize]).sum()
        highNonNormed = (weight[:stepHalfSize] * signal[:stepHalfSize]).sum()
        # Circular padding
        lowWeightSum = highWeightSum
        # lowSquareSum = highSquareSum
        lowNonNormed = -highNonNormed

    # ENH: vectorize this loop (it's the performance hotspot)
    for k in range(1, signalSize):
        highEnd = k + stepHalfSize - 1
        if highEnd >= signalSize:
            highEnd = signalSize - 1 - (highEnd - signalSize)
        lowEnd = k - stepHalfSize - 1
        if lowEnd < 0:
            lowEnd = -lowEnd - 1

        if weight is None:
            result[k] = result[k-1] + signal[highEnd] + signal[lowEnd] - 2*signal[k-1]
        else:
            
            lowNonNormed += signal[lowEnd] * weight[lowEnd] - signal[k-1] * weight[k-1]
            highNonNormed += signal[highEnd] * weight[highEnd] - signal[k-1] * weight[k-1]
            lowWeightSum += weight[k-1] - weight[lowEnd]
            highWeightSum += weight[highEnd] - weight[k-1]
            # lowSquareSum += weight[k-1] * weight[k-1] - weight[lowEnd] * weight[lowEnd]
            # highSquareSum += weight[highEnd] * weight[highEnd] - weight[k-1] * weight[k-1]
            result[k] = math.sqrt(stepHalfSize / 2) * (lowNonNormed / lowWeightSum +
                                                       highNonNormed / highWeightSum)

    if weight is None:
        stepNorm = math.sqrt(2. * stepHalfSize)
        result[1:signalSize] /= stepNorm

    return result


def FindLocalPeaks(signal, #const double * signal,
                   # peakLoc, #int * peakLoc
                  ):
    """Find local maxima on positive values, local minima on negative values.

    First and last index are never considered extramum.

    Parameters
    ----------
    signal : const array of floats

    Returns
    -------
    peakLoc : array of ints
        Locations of extrema in `signal`

    Source: HaarSeg.c
    """
    # use numpy.diff to simplify? argmax, argmin?
    maxSuspect = minSuspect = None
    peakLoc = []
    for k in range(1, len(signal) - 1):
        sig_prev, sig_curr, sig_next = signal[k-1:k+2]
        if sig_curr > 0:
            # Look for local maxima
            if (sig_curr > sig_prev) and (sig_curr > sig_next):
                peakLoc.append(k)
            elif (sig_curr > sig_prev) and (sig_curr == sig_next):
                maxSuspect = k
            elif (sig_curr == sig_prev) and (sig_curr > sig_next):
                # Take the first in a series of equal values
                if maxSuspect is not None:
                    peakLoc.append(maxSuspect)
                    maxSuspect = None
            elif (sig_curr == sig_prev) and (sig_curr < sig_next):
                maxSuspect = None

        elif sig_curr < 0:
            # Look for local maxima
            if (sig_curr < sig_prev) and (sig_curr < sig_next):
                peakLoc.append(k)
            elif (sig_curr < sig_prev) and (sig_curr == sig_next):
                minSuspect = k
            elif (sig_curr == sig_prev) and (sig_curr < sig_next):
                if minSuspect is not None:
                    peakLoc.append(minSuspect)
                    minSuspect = None
            elif (sig_curr == sig_prev) and (sig_curr > sig_next):
                minSuspect = None

    return np.array(peakLoc, dtype=np.int_)


def UnifyLevels(baseLevel, #const int * baseLevel,
                addonLevel, #const int * addonLevel,
                windowSize, #int windowSize,
                # joinedLevel, #int * joinedLevel);
               ):
    """Unify several decomposition levels.

    Merge the two lists of breakpoints, but drop addonLevel values that are too
    close to baseLevel values.

    Parameters
    ----------
    baseLevel : const array of ints
    addonLevel : const array of ints
    windowSize : int

    Returns
    -------
    joinedLevel : array of ints

    Source: HaarSeg.c
    """
    if not len(addonLevel):
        return baseLevel

    # Merge all addon items outside a window around each base item
    # ENH: do something clever with searchsorted & masks?
    joinedLevel = []
    addon_idx = 0
    for base_elem in baseLevel:
        while addon_idx < len(addonLevel):
            addon_elem = addonLevel[addon_idx]
            if addon_elem < base_elem - windowSize:
                # Addon is well before this base item -- use it
                joinedLevel.append(addon_elem)
                addon_idx += 1
            elif base_elem - windowSize <= addon_elem <= base_elem + windowSize:
                # Addon is too close to this base item -- skip it
                addon_idx += 1
            else:
                assert base_elem + windowSize < addon_elem
                # Addon is well beyond this base item -- keep for the next round
                break
        joinedLevel.append(base_elem)

    # Append the remaining addon items beyond the last base item's window
    last_pos = (baseLevel[-1] + windowSize if len(baseLevel) else -1)
    while addon_idx < len(addonLevel) and addonLevel[addon_idx] <= last_pos:
        addon_idx += 1
    if addon_idx < len(addonLevel):
        joinedLevel.extend(addonLevel[addon_idx:])

    return np.array(sorted(joinedLevel), dtype=np.int_)


def PulseConv(signal, #const double * signal,
              pulseSize, #int pulseSize,
             ):
    """Convolve a pulse function with a signal, applying circular padding to the
    signal.

    Used for non-stationary variance compensation.

    Parameters
    ----------
    signal: const array of floats
    pulseSize: int

    Returns
    -------
    array of floats

    Source: HaarSeg.c
    """
    signalSize = len(signal)
    if pulseSize > signalSize:
        # ENH: handle this endcase
        raise ValueError("pulseSize (%s) > signalSize (%s)"
                         % (pulseSize, signalSize))
    pulseHeight = 1. / pulseSize

    # Circular padding init
    result = np.zeros(signalSize, dtype=np.float_)
    for k in range((pulseSize + 1) // 2):
        result[0] += signal[k]
    for k in range(pulseSize // 2):
        result[0] += signal[k]
    result[0] *= pulseHeight

    n = 1
    for k in range(pulseSize // 2,
                   signalSize + (pulseSize // 2) - 1):
        tail = k - pulseSize
        if tail < 0:
            tail = -tail - 1
        head = k
        if head >= signalSize:
            head = signalSize - 1 - (head - signalSize)
        result[n] = result[n-1] + ((signal[head] - signal[tail]) * pulseHeight)
        n += 1

    return result


# XXX Apply afterward to the segmentation result? (not currently used)
def AdjustBreaks(signal, #const double * signal,
                 peakLoc, #const int * peakLoc,
                ):
    """Improve localization of breaks. Suboptimal, but linear-complexity.

    We try to move each break 1 sample left/right, choosing the offset which
    leads to minimum data error.

    Parameters
    ----------
    signal: const array of floats
    peakLoc: const array of ints

    Source: HaarSeg.c
    """
    newPeakLoc = peakLoc.copy()
    for k, npl_k in enumerate(newPeakLoc):
        # Calculating width of segments around the breakpoint
        n1 = (npl_k if k == 0
                else npl_k - newPeakLoc[k-1])
        n2 = (len(signal) if k+1 == len(newPeakLoc)
              else newPeakLoc[k+1])- npl_k

        # Find the best offset for current breakpoint, trying only 1 sample
        # offset
        bestScore = float("Inf")  # Smaller is better
        bestOffset = 0
        for p in (-1, 0, 1):
            # Pointless to try to remove single-sample segments
            if (n1 == 1 and p == -1) or (n2 == 1 and p == 1):
                continue

            signal_n1_to_p = signal[npl_k - n1:npl_k + p]
            s1 = signal_n1_to_p.sum() / (n1 + p)
            ss1 = ((signal_n1_to_p - s1)**2).sum()

            signal_p_to_n2 = signal[npl_k + p:npl_k + n2]
            s2 = signal_p_to_n2.sum() / (n2 - p)
            ss2 = ((signal_p_to_n2 - s2)**2).sum()

            score = ss1 + ss2
            if score < bestScore:
                bestScore = score
                bestOffset = p

        if bestOffset != 0:
            newPeakLoc[k] += bestOffset

    return newPeakLoc


# Testing

def table2coords(seg_table):
    """Return x, y arrays for plotting."""
    x = []
    y = []

    start = seg_table['start']
    size = seg_table['size']
    var = seg_table['mean']
    end = seg_table['end']
    for i in range(1, len(start)):
        for j in range(0, len(start)-i):
            if start[j] > start[j+1]:
                start[j], start[j + 1] = start[j + 1], start[j]
                size[j], size[j + 1] = size[j + 1], size[j]
                var[j], var[j + 1] = var[j + 1], var[j]
                end[j], end[j + 1] = end[j + 1], end[j]
    for i in range(len(start)):
        x.append(start[i])
        x.append(start[i]+size[i])
        y.append(var[i])
        y.append(var[i])
    # for start, size, val in seg_table:
    #     x.append(start)
    #     x.append(start + size)
    #     y.append(val)
    #     y.append(val)
    return x, y
'''
def sort_breakpoints(breakpoints):
    start = breakpoints['start']
    size = breakpoints['size']
    var = breakpoints['mean']
    end = breakpoints['end']
    newBreakpoints = {}
    for i in range(1, len(start)):
        for j in range(0, len(start) - i):
            if start[j] > start[j + 1]:
                start[j], start[j + 1] = start[j + 1], start[j]
                size[j], size[j + 1] = size[j + 1], size[j]
                var[j], var[j + 1] = var[j + 1], var[j]
                end[j], end[j + 1] = end[j + 1], end[j]
    newBreakpoints['start'] = start
    newBreakpoints['end'] = end
    newBreakpoints['size'] = size
    newBreakpoints['mean'] = var
    return newBreakpoints
'''


def sort_breakpoints(breakpoints):
    start = breakpoints['start']
    size = breakpoints['size']
    # var = breakpoints['mean']
    end = breakpoints['end']
    newBreakpoints = {}
    for i in range(1, len(start)):
        for j in range(0, len(start) - i):
            if start[j] > start[j + 1]:
                start[j], start[j + 1] = start[j + 1], start[j]
                size[j], size[j + 1] = size[j + 1], size[j]
                # var[j], var[j + 1] = var[j + 1], var[j]
                end[j], end[j + 1] = end[j + 1], end[j]
    newBreakpoints['start'] = start
    newBreakpoints['end'] = end
    newBreakpoints['size'] = size
    # newBreakpoints['mean'] = var
    return newBreakpoints


def EWMA_SEG(breakpointSelects):
    segs = []
    start = breakpointSelects[0]
    itm = 1
    edgeSize = 2
    for i in range(len(breakpointSelects)):
        if i + 1 < len(breakpointSelects):
            if itm == 1:
                start = breakpointSelects[i]
            itm += 1
            if breakpointSelects[i + 1] - breakpointSelects[i] < edgeSize:
                if i + 1 == len(breakpointSelects) - 1:
                    if start - edgeSize >0:
                        segs.append([start - edgeSize, breakpointSelects[i + 1] + edgeSize])
                    else:
                        segs.append([start, breakpointSelects[i + 1]])
            elif breakpointSelects[i + 1] - breakpointSelects[i] > edgeSize:
                end = breakpointSelects[i]
                if start - edgeSize < 0:
                    segs.append([start, end + edgeSize])
                else:
                    segs.append([start - edgeSize, end + edgeSize])
                itm = 1
    return segs
"""
def EWMA_model(arr):
    mu0 = arr[0].mean()
    num = float(2 / (1 + 10))
    #num = 0.2
    std = arr[0].std()
    upline = mu0 + std * 3 * math.sqrt(num / (2 - num))
    dowline = mu0 - std * 3 * math.sqrt(num / (2 - num))

    index = 0
    breakpoint_select = []
    for i in arr[0].ewm(span=10).mean():
        if i > upline or i < dowline:
            breakpoint_select.append(index)
        index += 1
    if len(breakpoint_select) > 0:
        segs = EWMA_SEG(breakpoint_select)
        return segs
    else:
        
        return[[0,len(arr)-1]]
"""

def breakpoint_select(dfArr,upline,dowline):
    index = 0
    breakpoints = []
    for i in dfArr[0].ewm(span=10).mean():
        if i > upline or i < dowline:
            breakpoints.append(index)
        index += 1
    return breakpoints


def EWMA_model(arr):
    # print(arr[0].ewm(span=10).mean())
    # print(arr.iloc[::-1][0].ewm(span=10).mean())
    # exit()

    mu0 = arr[0].mean()
    num0 = float(2 / (1 + 10))
    std0 = arr[0].std()
    upline0 = mu0 + std0 * 3 * math.sqrt(num0 / (2 - num0))
    dowline0 = mu0 - std0 * 3 * math.sqrt(num0 / (2 - num0))
    breakpointSelects0 = breakpoint_select(arr, upline0, dowline0)
    arr_ = arr.iloc[::-1]

    mu1 = arr_[0].mean()
    num1 = float(2 / (1 + 10))
    std1 = arr_[0].std()
    upline1 = mu1 + std1 * 3 * math.sqrt(num1 / (2 - num1))
    dowline1 = mu1 - std1 * 3 * math.sqrt(num1 / (2 - num1))

    breakpointSelects1 = breakpoint_select(arr_, upline1, dowline1)
    breakpointSelects2 = []
    for r in breakpointSelects1[::-1]:
        breakpointSelects2.append(np.abs(r-(len(arr)-1)))
    breakpointSelects = sorted(list(set(breakpointSelects0+breakpointSelects2)))
    if len(breakpointSelects) > 0:
        segs = EWMA_SEG(breakpointSelects)
        return segs
    else:
        return[[0,len(arr)-1]]

def break_points_merge(segs,I,breaksFdrQ,W,rawI,haarStartLevel,haarEndLevel):
    index_seg = 0
    dict2 = {}
    start = []
    end = []
    size = []
    mean = []
    end_index = 0
    for seg in segs:
        # results = EWMA_haar(cnarr.smooth_log2(),
        #                       fdr_q,
        #                       W=(cnarr['weight'].values if 'weight' in cnarr
        #                          else None))
        
        seg_table = haarSeg(I[seg[0]:seg[1]], breaksFdrQ, W[seg[0]:seg[1]], rawI, haarStartLevel, haarEndLevel)
        start0 = []
        end0 = []
        size0 = []
        mean0 = []
        if seg_table['start'].size < 2:
            continue
        if seg_table['start'].size == 3 or seg_table['start'].size == 2:
            for i in range(seg_table['start'].size):
                seg_table['end'][i] += seg[0]
                seg_table['start'][i] += seg[0]
                start0.append(seg_table['start'][i])
                end0.append(seg_table['end'][i])
                size0.append(seg_table['size'][i])
            if seg_table['start'].size == 3:
                if index_seg == 0:
                   mean0 = seg_table['mean'].tolist()
                   index_seg +=1
                else:
                   mean0 = seg_table['mean'][1:].tolist()
            else:
                mean0 = seg_table['mean'].tolist()
        
        if seg_table['start'].size == 2:
            if abs(seg_table['start'][-1]-seg[0]) >= abs(seg_table['start'][-1]-seg[-1]):
                start0.insert(0,seg[0]-2)
                end0.insert(0,seg[0]-1)
                size0.insert(0,2)                  
                if index_seg == 0:
                    mean0.insert(0,seg_table['mean'][0])
                    index_seg +=1
   
                #if index_seg == 0 and end_index == (len(segs)-1):
                 #   mean0.insert(0,seg_table['mean'][0])
            else:
                start0.append(seg[-1])
                end0.append(seg[-1] + 2)
                size0.append(3)
                if index_seg == 0:
                    mean0.append(seg_table['mean'][-1])
                    index_seg +=1
                #if index_seg == 0 and end_index == (len(segs)-1):
                #    mean0.insert(0,seg_table['mean'][0])
                
        start += start0[1:]
        end += end0[:-1]
        mean += mean0
        for i in range(1, len(start)):
            for j in range(0, len(start) - i):
                if start[j] > start[j + 1]:
                    start[j], start[j + 1] = start[j + 1], start[j]
                    mean[j], mean[j + 1] = mean[j + 1], mean[j]
                    end[j], end[j + 1] = end[j + 1], end[j]

        # if seg_table['start'].size > 3:
        #     start.append(seg[0] + 100)
        #     end.append(seg[-1] - 100)
        #     size.append(seg[-1]-seg[0]+1)
        #     mean.append(np.average(seg_table['mean']))
        #     print(seg_table['end'][-1]+1)
        #     start.append(seg_table['end'][-1]+1)
        #     end.append(seg_table['end'][-1]+101)
        #     size.append(101)
        #     mean.append(seg_table['mean'][-1])
    #     if seg_table['start'].size > 1:
    #         for i in range(seg_table['start'].size):
    #             if i == 0 and index_seg > 0:
    #                 seg_table['end'][i] += seg[0]
    #                 seg_table['size'][i] += seg[0]+seg_table['start'][0]
    #                 dict2['start'] = np.array([seg_table['start'][i]])
    #                 dict2['end'] = np.array([seg_table['end'][i]])
    #                 dict2['mean'] = np.array([seg_table['mean'][i]])
    #                 dict2['size'] = np.array([seg_table['size'][i]])
    #                 index_seg += 1
    #                 continue
    #             elif i == 0 and index_seg == 0:
    #                 seg_table['end'][i] += seg[0]
    #                 seg_table['size'][i] += seg[0]
    #                 dict2['start'] = np.array([seg_table['start'][i]])
    #                 dict2['end'] = np.array([seg_table['end'][i]])
    #                 dict2['mean'] = np.array([seg_table['mean'][i]])
    #                 dict2['size'] = np.array([seg_table['size'][i]])
    #                 continue
    #             elif i == seg_table['start'].size-1:
    #                 continue
    #             seg_table['start'][i] += seg[0]
    #             seg_table['end'][i] += seg[0]
    #             if i == 1:
    #                 seg_table['end'][i] += seg[0]
    #                 seg_table['size'][i] += seg[0]
    #                 dict2['start'] = np.append(dict2['start'], [seg_table['start'][i]])
    #                 dict2['end'] = np.append(dict2['end'], [seg_table['end'][i]])
    #                 dict2['mean'] = np.append(dict2['mean'], [seg_table['mean'][i]])
    #                 dict2['size'] = np.append(dict2['size'], [seg_table['size'][i]])
    #             else:
    #                 dict2['start'] = np.append(dict2['start'], [seg_table['start'][i]])
    #                 dict2['end'] = np.append(dict2['end'], [seg_table['end'][i]])
    #                 dict2['mean'] = np.append(dict2['mean'], [seg_table['mean'][i]])
    #                 dict2['size'] = np.append(dict2['size'], [seg_table['size'][i]])
    #     else:
    #         if index_seg == 0:
    #             index_seg += 1
    #             continue
    if len(start) > 0: 
        start.insert(0,0)
        end.append(len(I)-1)
        size = list(map(lambda x, y: x - y+1,end,start))
    else:
        dict2['start'] = np.array(start)
        dict2['end'] = np.array(end)
        dict2['size'] = np.array(size)
        dict2['mean'] = np.array(mean)
        return dict2
    #mean = mean[:-1]
    if end[0] <= 0:
        del(end[0])
        del(start[0])
        del(mean[0])
        del(size[0])
    if start[-1] >= len(I)-1:
       # print(len(I))
       # print(start)
       # print(end)
       # print(mean)
        del(end[-1])
        del(start[-1])
        del(mean[-1])
        del(size[-1])

    dict2['start'] = np.array(start)
    dict2['end'] = np.array(end)
    dict2['size'] = np.array(size)
    #dict2['mean'] = np.array(mean)
    newdict = sort_breakpoints(dict2)
    means = []
    for i in range(len(newdict['start'])):
        means.append(np.mean(I[newdict['start'][i]:newdict['end'][i]]))
    newdict['mean'] = np.array(means)

    return newdict


# results = EWMA_haar(cnarr.smooth_log2(),
#                       fdr_q,
#                       W=(cnarr['weight'].values if 'weight' in cnarr
#                          else None))


def merge_segs(segs):
    newSegs = []
    tm = []
    for li in segs:
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


def EWMA_haar(I, breaksFdrQ,
            W=None,
            rawI=None,
            haarStartLevel=1,
            haarEndLevel=5):
    if len(I) > 1:
        seq_data_arr = pd.DataFrame(I)
        segs = EWMA_model(seq_data_arr)
        segs = merge_segs(segs)
        try:
            breakPoints = break_points_merge(segs, I, breaksFdrQ, W, rawI, haarStartLevel, haarEndLevel)
        except:
            breakPoints = {'start':np.array([0]),'end':np.array([1]),'size':np.array([2]),'mean':np.array([0])}
        if breakPoints['start'].size == 0:
            try:
                breakPoints = haarSeg(I, breaksFdrQ, W, rawI, haarStartLevel, haarEndLevel)
            except:
                breakPoints = {'start':np.array([0]),'end':np.array([1]),'size':np.array([2]),'mean':np.array([0])}
    else:
        try:
            breakPoints = haarSeg(I, breaksFdrQ, W, rawI, haarStartLevel, haarEndLevel)
        except:
            breakPoints = {'start':np.array([0]),'end':np.array([1]),'size':np.array([2]),'mean':np.array([0])}
        
    return breakPoints

if __name__ == '__main__':
    real_data = np.concatenate((np.zeros(800), np.ones(200),
                                np.zeros(800), .8*np.ones(200), np.zeros(800)))
    # np.random.seed(0x5EED)
    noisy_data = real_data + np.random.standard_normal(len(real_data)) * .2
    noisy_data_arr = pd.DataFrame(noisy_data)
    # print(noisy_data_arr)
    # segs = EWMA_model(noisy_data_arr)

    # Run using default parameters
    seg_table = haarSeg(noisy_data, .005)
    breakPoints = EWMA_haar(noisy_data)

    # logging.info("%s", seg_table)
    # #
    from matplotlib import pyplot
    indices = np.arange(len(noisy_data))
    pyplot.scatter(indices, noisy_data, alpha=0.2, color='gray')
    x, y = table2coords(seg_table)
    x1, y1 = table2coords(breakPoints)

    pyplot.plot(x, y, color='r', marker='x', lw=2)
    pyplot.plot(x1, y1, color='b', marker='x', lw=2)
    pyplot.show()
