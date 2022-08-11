import logging
import math

import numpy as np
import pandas as pd
from scipy import stats


def segment_haar(cnarr, fdr_q):
    chrom_tables = [one_chrom(subprobes, fdr_q, chrom)
                    for chrom, subprobes in cnarr.by_arm()]
    segarr = cnarr.as_dataframe(pd.concat(chrom_tables))
    segarr.sort_columns()
    return segarr


def one_chrom(cnarr, fdr_q, chrom):
    logging.debug("Segmenting %s", chrom)
    results = haarSeg(cnarr.smooth_log2(),
                      fdr_q,
                      W=(cnarr['weight'].values if 'weight' in cnarr
                         else None))
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
                          W=None)
    else:
        values = pd.Series()
        results = None
    if results is not None and len(results['start']) > 1:
        logging.info("Segmented on allele freqs in %s:%d-%d",
                     segment.chromosome, segment.start, segment.end)

        gap_rights = varr['start'].values.take(results['start'][1:])
        gap_lefts = varr['end'].values.take(results['end'][:-1])
        mid_breakpoints = (gap_lefts + gap_rights) // 2
        starts = np.concatenate([[segment.start], mid_breakpoints])
        ends = np.concatenate([mid_breakpoints, [segment.end]])
        table = pd.DataFrame({
            'chromosome': segment.chromosome,
            'start': starts,
            'end': ends,

            'gene': segment.gene,
            'log2': segment.log2,
            'probes': results['size'],

        })
    else:
        table = pd.DataFrame({
            'chromosome': segment.chromosome,
            'start': segment.start,
            'end': segment.end,

            'gene': segment.gene,
            'log2': segment.log2,
            'probes': segment.probes,

        }, index=[0])

    return table


def haarSeg(I, breaksFdrQ,
            W=None,
            rawI=None,
            haarStartLevel=1,
            haarEndLevel=5):
    def med_abs_diff(diff_vals):
        if len(diff_vals) == 0:
            return 0.
        return diff_vals.abs().median() * 1.4826

    diffI = pd.Series(HaarConv(I, None, 1))
    if rawI:
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
        addonPeaks = np.extract(np.abs(convRes.take(peakLoc)) >= T, peakLoc)
        breakpoints = UnifyLevels(breakpoints, addonPeaks, 2 ** (level - 1))

    logging.debug("Found %d breakpoints: %s", len(breakpoints), breakpoints)

    segs = SegmentByPeaks(I, breakpoints, W)
    segSt = np.insert(breakpoints, 0, 0)
    segEd = np.append(breakpoints, len(I))
    return {'start': segSt,
            'end': segEd - 1,
            'size': segEd - segSt,
            'mean': segs[segSt]}


def FDRThres(x, q, stdev):
    M = len(x)
    if M < 2:
        return 0

    m = np.arange(1, M + 1) / M
    x_sorted = np.sort(np.abs(x))[::-1]
    p = 2 * (1 - stats.norm.cdf(x_sorted, stdev))
    indices = np.nonzero(p <= m * q)[0]
    if len(indices):
        T = x_sorted[indices[-1]]
    else:
        logging.debug("No passing p-values: min p=%.4g, min m=%.4g, q=%s",
                      p[0], m[0], q)
        T = x_sorted[0] + 1e-16
    return T


def SegmentByPeaks(data, peaks, weights=None):
    segs = np.zeros_like(data)
    for seg_start, seg_end in zip(np.insert(peaks, 0, 0),
                                  np.append(peaks, len(data))):
        if weights is not None and weights[seg_start:seg_end].sum() > 0:

            val = np.average(data[seg_start:seg_end],
                             weights=weights[seg_start:seg_end])
        else:

            val = np.mean(data[seg_start:seg_end])
        segs[seg_start:seg_end] = val
    return segs


def HaarConv(signal,
             weight,
             stepHalfSize,
             ):
    signalSize = len(signal)
    if stepHalfSize > signalSize:
        logging.debug("Error?: stepHalfSize (%s) > signalSize (%s)",
                      stepHalfSize, signalSize)
        return np.zeros(signalSize, dtype=np.float_)

    result = np.zeros(signalSize, dtype=np.float_)
    if weight is not None:
        highWeightSum = weight[:stepHalfSize].sum()

        highNonNormed = (weight[:stepHalfSize] * signal[:stepHalfSize]).sum()

        lowWeightSum = highWeightSum

        lowNonNormed = -highNonNormed

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


def FindLocalPeaks(signal,

                   ):
    maxSuspect = minSuspect = None
    peakLoc = []
    for k in range(1, len(signal) - 1):
        sig_prev, sig_curr, sig_next = signal[k - 1:k + 2]
        if sig_curr > 0:

            if (sig_curr > sig_prev) and (sig_curr > sig_next):
                peakLoc.append(k)
            elif (sig_curr > sig_prev) and (sig_curr == sig_next):
                maxSuspect = k
            elif (sig_curr == sig_prev) and (sig_curr > sig_next):

                if maxSuspect is not None:
                    peakLoc.append(maxSuspect)
                    maxSuspect = None
            elif (sig_curr == sig_prev) and (sig_curr < sig_next):
                maxSuspect = None

        elif sig_curr < 0:

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


def UnifyLevels(baseLevel,
                addonLevel,
                windowSize,

                ):
    if not len(addonLevel):
        return baseLevel

    joinedLevel = []
    addon_idx = 0
    for base_elem in baseLevel:
        while addon_idx < len(addonLevel):
            addon_elem = addonLevel[addon_idx]
            if addon_elem < base_elem - windowSize:

                joinedLevel.append(addon_elem)
                addon_idx += 1
            elif base_elem - windowSize <= addon_elem <= base_elem + windowSize:

                addon_idx += 1
            else:
                assert base_elem + windowSize < addon_elem

                break
        joinedLevel.append(base_elem)

    last_pos = (baseLevel[-1] + windowSize if len(baseLevel) else -1)
    while addon_idx < len(addonLevel) and addonLevel[addon_idx] <= last_pos:
        addon_idx += 1
    if addon_idx < len(addonLevel):
        joinedLevel.extend(addonLevel[addon_idx:])

    return np.array(sorted(joinedLevel), dtype=np.int_)


def PulseConv(signal,
              pulseSize,
              ):
    signalSize = len(signal)
    if pulseSize > signalSize:
        raise ValueError("pulseSize (%s) > signalSize (%s)"
                         % (pulseSize, signalSize))
    pulseHeight = 1. / pulseSize

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


def AdjustBreaks(signal,
                 peakLoc,
                 ):
    newPeakLoc = peakLoc.copy()
    for k, npl_k in enumerate(newPeakLoc):

        n1 = (npl_k if k == 0
              else npl_k - newPeakLoc[k - 1])
        n2 = (len(signal) if k + 1 == len(newPeakLoc)
              else newPeakLoc[k + 1]) - npl_k

        bestScore = float("Inf")
        bestOffset = 0
        for p in (-1, 0, 1):

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


def table2coords(seg_table):
    """Return x, y arrays for plotting."""
    x = []
    y = []
    for start, size, val in seg_table:
        x.append(start)
        x.append(start + size)
        y.append(val)
        y.append(val)
    return x, y


if __name__ == '__main__':
    real_data = np.concatenate((np.zeros(800), np.ones(200),
                                np.zeros(800), .8 * np.ones(200), np.zeros(800)))

    noisy_data = real_data + np.random.standard_normal(len(real_data)) * .2

    seg_table = haarSeg(noisy_data, .005)

    logging.info("%s", seg_table)

    from matplotlib import pyplot

    indices = np.arange(len(noisy_data))
    pyplot.scatter(indices, noisy_data, alpha=0.2, color='gray')
    x, y = table2coords(seg_table)
    pyplot.plot(x, y, color='r', marker='x', lw=2)
    pyplot.show()
