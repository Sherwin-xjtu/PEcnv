# !/usr/bin/python
# coding=utf-8
from __future__ import division

import logging
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy import stats

from PElibrary.adjustingCoverage import adajustingBaseDepth, adajustedEdge


def segment_haar(cnarr, fdr_q):
    chrom_tables = [one_chrom(subprobes, fdr_q, chrom)
                    for chrom, subprobes in cnarr.by_arm()]

    segarr = cnarr.as_dataframe(pd.concat(chrom_tables))
    segarr.sort_columns()
    return segarr


def one_chrom(cnarr, fdr_q, chrom):
    logging.debug("Segmenting %s", chrom)

    results = PE_haar(cnarr.smooth_log2(),
                      fdr_q,
                      W=(cnarr['weight'].values if 'weight' in cnarr
                         else None))
    print(results)
    # print(len(results['start']),len(results['end']),len(results['size']),len(results['mean'])) 
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
            'gene': segment.gene,  # '-'
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
            'gene': segment.gene,  # '-',
            'log2': segment.log2,
            'probes': segment.probes,
            # 'weight': segment.weight,
        }, index=[0])

    return table


def haarSeg(I, breaksFdrQ,
            W=None,
            rawI=None,
            haarStartLevel=1,
            haarEndLevel=5):
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
    for level in range(haarStartLevel, haarEndLevel + 1):
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

    m = np.arange(1, M + 1) / M
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


def HaarConv(signal,  # const double * signal,
             weight,  # const double * weight,
             stepHalfSize,  # int stepHalfSize,
             ):
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
            result[k] = result[k - 1] + signal[highEnd] + signal[lowEnd] - 2 * signal[k - 1]
        else:

            lowNonNormed += signal[lowEnd] * weight[lowEnd] - signal[k - 1] * weight[k - 1]
            highNonNormed += signal[highEnd] * weight[highEnd] - signal[k - 1] * weight[k - 1]
            lowWeightSum += weight[k - 1] - weight[lowEnd]
            highWeightSum += weight[highEnd] - weight[k - 1]

            result[k] = math.sqrt(stepHalfSize / 2) * (lowNonNormed / lowWeightSum +
                                                       highNonNormed / highWeightSum)

    if weight is None:
        stepNorm = math.sqrt(2. * stepHalfSize)
        result[1:signalSize] /= stepNorm

    return result


def FindLocalPeaks(signal,  # const double * signal,
                   # peakLoc, #int * peakLoc
                   ):
    # use numpy.diff to simplify? argmax, argmin?
    maxSuspect = minSuspect = None
    peakLoc = []
    for k in range(1, len(signal) - 1):
        sig_prev, sig_curr, sig_next = signal[k - 1:k + 2]
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


def UnifyLevels(baseLevel,  # const int * baseLevel,
                addonLevel,  # const int * addonLevel,
                windowSize,  # int windowSize,
                # joinedLevel, #int * joinedLevel);
                ):
    if not len(addonLevel):
        return baseLevel

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

 
def PulseConv(signal,  # const double * signal,
              pulseSize,  # int pulseSize,
              ):
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
        result[n] = result[n - 1] + ((signal[head] - signal[tail]) * pulseHeight)
        n += 1

    return result


def AdjustBreaks(signal,  # const double * signal,
                 peakLoc,  # const int * peakLoc,
                 ):
    newPeakLoc = peakLoc.copy()
    for k, npl_k in enumerate(newPeakLoc):
        # Calculating width of segments around the breakpoint
        n1 = (npl_k if k == 0
              else npl_k - newPeakLoc[k - 1])
        n2 = (len(signal) if k + 1 == len(newPeakLoc)
              else newPeakLoc[k + 1]) - npl_k

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
            ss1 = ((signal_n1_to_p - s1) ** 2).sum()

            signal_p_to_n2 = signal[npl_k + p:npl_k + n2]
            s2 = signal_p_to_n2.sum() / (n2 - p)
            ss2 = ((signal_p_to_n2 - s2) ** 2).sum()

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
        for j in range(0, len(start) - i):
            if start[j] > start[j + 1]:
                start[j], start[j + 1] = start[j + 1], start[j]
                size[j], size[j + 1] = size[j + 1], size[j]
                var[j], var[j + 1] = var[j + 1], var[j]
                end[j], end[j + 1] = end[j + 1], end[j]
    for i in range(len(start)):
        x.append(start[i])
        x.append(start[i] + size[i])
        y.append(var[i])
        y.append(var[i])

    return x, y


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


def selectingSegments(breakpoint_select):
    segs = []
    start = breakpoint_select[0]
    itm = 1
    for i in range(len(breakpoint_select)):
        if i + 1 < len(breakpoint_select):
            if itm == 1:
                start = breakpoint_select[i]
            itm += 1
            if breakpoint_select[i + 1] - breakpoint_select[i] < 20:
                if i + 1 == len(breakpoint_select) - 1:
                    if start - 2 > 0:
                        segs.append([start - 2, breakpoint_select[i + 1] + 2])
                    else:
                        segs.append([start, breakpoint_select[i + 1]])
            elif breakpoint_select[i + 1] - breakpoint_select[i] > 10:
                end = breakpoint_select[i]
                if start - 2 < 0:
                    segs.append([start, end + 2])
                else:
                    segs.append([start - 2, end + 2])
                itm = 1
    return segs


def breakpoint_select(dfArr, upline, dowline):
    index = 0
    breakpoints = []
    for i in dfArr:
        if i > upline or i < dowline:
            breakpoints.append(index)
        index += 1
    return breakpoints


def indentifyingRegions(arr):
    mu0 = arr[0].mean()
    # num0 = float(2 / (1 + 10))
    num0 = 0.2
    std0 = arr[0].std()
    upline0 = mu0 + std0 * 3 * math.sqrt(num0 / (2 - num0))
    dowline0 = mu0 - std0 * 3 * math.sqrt(num0 / (2 - num0))
    breakpointSelects0 = breakpoint_select(arr, upline0, dowline0)
    arr_ = arr.iloc[::-1]

    mu1 = arr_[0].mean()
    # num1 = float(2 / (1 + 10))
    num1 = 0.2
    std1 = arr_[0].std()
    upline1 = mu1 + std1 * 3 * math.sqrt(num1 / (2 - num1))
    dowline1 = mu1 - std1 * 3 * math.sqrt(num1 / (2 - num1))

    breakpointSelects1 = breakpoint_select(arr_, upline1, dowline1)
    breakpointSelects2 = []
    for r in breakpointSelects1[::-1]:
        breakpointSelects2.append(np.abs(r - (len(arr) - 1)))
    breakpointSelects = sorted(list(set(breakpointSelects0 + breakpointSelects2)))
    if len(breakpointSelects) > 0:
        segs = selectingSegments(breakpointSelects)
        return segs
    else:
        return [[0, len(arr) - 1]]


def break_points_merge(segs, I, breaksFdrQ, W, rawI, haarStartLevel, haarEndLevel):
    index_seg = 0
    dict2 = {}
    start = []
    end = []
    size = []
    mean = []
    end_index = 0
    seq_data_arr = pd.DataFrame(I)
    adjusted_data = seq_data_arr
    I0 = np.array(adjusted_data)
    indices = np.arange(len(I0))
    pyplot.scatter(indices, adjusted_data, alpha=0.2, c='r')
    for seg in segs:

        seg_table = haarSeg(I0[seg[0]:seg[1]], breaksFdrQ)

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
                    index_seg += 1
                else:
                    mean0 = seg_table['mean'][1:].tolist()
            else:
                mean0 = seg_table['mean'].tolist()

        if seg_table['start'].size == 2:
            if abs(seg_table['start'][-1] - seg[0]) >= abs(seg_table['start'][-1] - seg[-1]):
                start0.insert(0, seg[0] - 2)
                end0.insert(0, seg[0] - 1)
                size0.insert(0, 2)
                if index_seg == 0:
                    mean0.insert(0, seg_table['mean'][0])
                    index_seg += 1

                # if index_seg == 0 and end_index == (len(segs)-1):
                #   mean0.insert(0,seg_table['mean'][0])
            else:
                start0.append(seg[-1])
                end0.append(seg[-1] + 2)
                size0.append(3)
                if index_seg == 0:
                    mean0.append(seg_table['mean'][-1])
                    index_seg += 1
                # if index_seg == 0 and end_index == (len(segs)-1):
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
        start.insert(0, 0)
        end.append(len(I) - 1)
        size = list(map(lambda x, y: x - y + 1, end, start))
    else:
        dict2['start'] = np.array(start)
        dict2['end'] = np.array(end)
        dict2['size'] = np.array(size)
        dict2['mean'] = np.array(mean)
        return dict2
    # mean = mean[:-1]
    if end[0] <= 0:
        del (end[0])
        del (start[0])
        del (mean[0])
        del (size[0])
    if start[-1] >= len(I) - 1:
        del (end[-1])
        del (start[-1])
        del (mean[-1])
        del (size[-1])

    dict2['start'] = np.array(start)
    dict2['end'] = np.array(end)
    dict2['size'] = np.array(size)
    # dict2['mean'] = np.array(mean)
    newdict = sort_breakpoints(dict2)
    means = []
    for i in range(len(newdict['start'])):
        means.append(np.mean(I[newdict['start'][i]:newdict['end'][i]]))
    newdict['mean'] = np.array(means)

    return newdict


def PE_haar(I, breaksFdrQ,
            W=None,
            rawI=None,
            haarStartLevel=1,
            haarEndLevel=5):
    if len(I) > 1:

        seq_data_arr = pd.DataFrame(I)
        segs = indentifyingRegions(seq_data_arr)
        breakPoints = break_points_merge(segs, I, breaksFdrQ, W, rawI, haarStartLevel, haarEndLevel)
        if breakPoints['start'].size == 0:
            try:
                breakPoints = haarSeg(I, breaksFdrQ, W, rawI, haarStartLevel, haarEndLevel)
            except:
                breakPoints = {'start': np.array([0]), 'end': np.array([1]), 'size': np.array([2]),
                               'mean': np.array([0])}
    else:
        try:
            breakPoints = haarSeg(I, breaksFdrQ, W, rawI, haarStartLevel, haarEndLevel)
        except:
            breakPoints = {'start': np.array([0]), 'end': np.array([1]), 'size': np.array([2]), 'mean': np.array([0])}

    return breakPoints


if __name__ == '__main__':

    real_data = np.concatenate((np.zeros(800), np.ones(200),
                                np.zeros(800), .8 * np.ones(200), np.zeros(800)))
    # np.random.seed(0x5EED)
    noisy_data = real_data + np.random.standard_normal(len(real_data)) * .2
    noisy_data_arr = pd.DataFrame(noisy_data)
    noisy_data_li = noisy_data_arr[0].tolist()
    bases_depth = noisy_data_li

    for i in range(len(bases_depth) - 1):
        n1, n2 = 1, 1
        print(bases_depth[i], adajustingBaseDepth(i, bases_depth, n1, n2))
        print(bases_depth[i], adajustedEdge(i, bases_depth))

    real_data = np.concatenate((np.zeros(800), np.ones(200),
                                np.zeros(800), .8 * np.ones(200), np.zeros(800)))


    noisy_data = real_data + np.random.standard_normal(len(real_data)) * .2
    noisy_data_arr = pd.DataFrame(noisy_data)
    # print(noisy_data_arr)


    # reader = pd.read_csv(file_path, sep="\t")
    file_path = 'F:/CNV/PEcnv/cnvCalling/PEcnv/3.tsv'
    reader = pd.read_csv(file_path, sep='\t')
    reader_chr1 = reader[reader['chromosome'] == '2']

    # data0 = reader['log2']
    # data = np.array(data0)
    data0 = reader_chr1['log2']
    data = np.array(data0)

    # data0_ = np.diff(data0)
    #
    # for i in range(2):
    #     data0_ = np.diff(data0_)
    # X = [i for i in data0_ if i != 0]
    # print(np.log2(X))

    # plt.plot(np.log2(data0))

    # y1 = data0.rolling(200, 1, center=True).median()
    # data = np.array(y1)
    # data_log2 = np.log2(data)
    # y2 = savgol_filter(data, 7, 3, mode='nearest')
    # Run using default parameters

    seg_table = haarSeg(data, 0.0001)

    # logging.info("%s", seg_table)
    # #

    indices = np.arange(len(data))
    seq_data_arr = pd.DataFrame(data)
    adjusted_data = seq_data_arr

    pyplot.scatter(indices, data, alpha=0.2, color='gray')
    # print(breakPoints)

    x, y = table2coords(seg_table)
    # x1, y1 = table2coords(breakPoints)

    pyplot.plot(x, y, color='b', marker='x', lw=2)
    # pyplot.plot(x1, y1, color='b', marker='x', lw=2)
    pyplot.show()
