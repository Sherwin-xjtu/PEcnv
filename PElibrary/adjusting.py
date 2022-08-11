import logging
import math

import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs, savgol_filter
from . import measures


def check_inputs(x, width, as_series=True, weights=None):
    x = np.asfarray(x)
    wing = _width2wing(width, x)
    signal = _pad_array(x, wing)
    if as_series:
        signal = pd.Series(signal)
    if weights is None:
        return x, wing, signal

    weights = _pad_array(weights, wing)

    weights[:wing] *= np.linspace(1 / wing, 1, wing)
    weights[-wing:] *= np.linspace(1, 1 / wing, wing)
    if as_series:
        weights = pd.Series(weights)
    return x, wing, signal, weights


def _width2wing(width, x, min_wing=3):
    if 0 < width < 1:
        wing = int(math.ceil(len(x) * width * 0.5))
    elif width >= 2 and int(width) == width:

        width = min(width, len(x) - 1)
        wing = int(width // 2)
    else:
        raise ValueError("width must be either a fraction between 0 and 1 "
                         "or an integer greater than 1 (got %s)" % width)
    wing = max(wing, min_wing)
    wing = min(wing, len(x) - 1)
    assert wing >= 1, "Wing must be at least 1 (got %s)" % wing
    return wing


def _pad_array(x, wing):
    return np.concatenate((x[wing - 1::-1],
                           x,
                           x[:-wing - 1:-1]))


def merge_segs(segs):
    newSegs = []
    tm = []
    for li in segs:
        if len(tm) == 0:
            tm = li
            newSegs.append(tm)
        else:
            if li[0] <= tm[1]:
                tm = [tm[0], li[1]]
                del (newSegs[-1])
                newSegs.append(tm)
            else:
                newSegs.append(li)
                tm = li
    return newSegs


def EWMA_SEG(breakpointSelects):
    segs = []
    start = breakpointSelects[0]
    itm = 1
    edgeSize = 1
    for i in range(len(breakpointSelects)):
        if i + 1 < len(breakpointSelects):
            if itm == 1:
                start = breakpointSelects[i]
            itm += 1
            if breakpointSelects[i + 1] - breakpointSelects[i] < edgeSize:
                if i + 1 == len(breakpointSelects) - 1:
                    if start - edgeSize > 0:
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


def breakpoint_select(dfArr, sp, upline, dowline):
    index = 0
    breakpoints = []
    for i in dfArr.ewm(span=1).mean():
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
        breakpointSelects2.append(np.abs(r - (len(arr) - 1)))
    breakpointSelects = sorted(list(set(breakpointSelects0 + breakpointSelects2)))
    if len(breakpointSelects) > 0:
        segs = EWMA_SEG(breakpointSelects)
        return segs
    else:
        return [[0, len(arr) - 1]]


def rolling_median(x, width):
    seq_data_arr = x
    segs = EWMA_model(seq_data_arr)
    newSegs = merge_segs(segs)
    interm = 0
    tmPos = 0
    winSize = 2
    if len(newSegs) > 0:
        for seg in newSegs:
            """Rolling median with mirrored edges."""

            seg_x = x[seg[0]:seg[1]]

            if seg_x.size > winSize + 1:
                seg_x, wing, signal = check_inputs(seg_x, width)
                seg_rolled = signal.rolling(2 * wing + 1, 1, center=True).median()
                seg_rolled = np.asfarray(seg_rolled[wing:-wing])
                seg_rolled_toli = seg_rolled.tolist()
            else:
                seg_rolled_toli = seg_x.tolist()

            normal_x = x[tmPos:seg[0]]
            tmPos = seg[1]
            if normal_x.size > winSize + 1:
                normal_x, wing, signal = check_inputs(normal_x, width)
                rolled = signal.rolling(winSize * wing + 1, 1, center=True).median()
                rolled = np.asfarray(rolled[wing:-wing])
                rolled_toli = rolled.tolist()

                rolled_toli = rolled_toli + seg_rolled_toli
                if interm == 0:

                    rolledAll = rolled_toli
                    interm += 1
                else:

                    rolledAll += rolled_toli
            else:
                rolled_toli = normal_x.tolist()

                rolled_toli = rolled_toli + seg_rolled_toli
                if interm == 0:

                    rolledAll = rolled_toli
                    interm += 1
                else:

                    rolledAll += rolled_toli

        if tmPos != x[x.size - 1]:
            endList = x[tmPos:]
            if endList.size > winSize + 1:
                endList, wing, signal = check_inputs(endList, width)
                rolled = signal.rolling(winSize * wing + 1, 1, center=True).median()
                rolled = np.asfarray(rolled[wing:-wing])
                rolled_toli = rolled.tolist()
                if interm == 0:

                    rolledAll = rolled_toli
                    interm += 1
                else:

                    rolledAll += rolled_toli
            else:
                endList = endList.tolist()

                rolled_toli = endList
                if interm == 0:

                    rolledAll = rolled_toli
                    interm += 1
                else:

                    rolledAll += rolled_toli
        rolledAll_ = np.asfarray(rolledAll)


    else:
        """Rolling median with mirrored edges."""
        x, wing, signal = check_inputs(x, width)
        rolled = signal.rolling(winSize * wing + 1, 1, center=True).median()
        rolledAll_ = np.asfarray(rolled[wing:-wing])
    return rolledAll_


def rolling_quantile(x, width, quantile):
    x, wing, signal = check_inputs(x, width)
    rolled = signal.rolling(2 * wing + 1, 2, center=True).quantile(quantile)
    return np.asfarray(rolled[wing:-wing])


def rolling_std(x, width):
    x, wing, signal = check_inputs(x, width)
    rolled = signal.rolling(2 * wing + 1, 2, center=True).std()
    return np.asfarray(rolled[wing:-wing])


def convolve_weighted(window, signal, weights, n_iter=1):
    assert len(weights) == len(signal), (
            "len(weights) = %d, len(signal) = %d, window_size = %s"
            % (len(weights), len(signal), len(window)))
    y, w = signal, weights
    window /= window.sum()
    for _i in range(n_iter):
        logging.debug("Iteration %d: len(y)=%d, len(w)=%d",
                      _i, len(y), len(w))
        D = np.convolve(w * y, window, mode='same')
        N = np.convolve(w, window, mode='same')
        y = D / N

        w = np.convolve(window, w, mode='same')
    return y, w


def convolve_unweighted(window, signal, wing, n_iter=1):
    window /= window.sum()
    y = signal
    for _i in range(n_iter):
        y = np.convolve(window, y, mode='same')

    y = y[wing:-wing]
    return y


def guess_window_size(x, weights=None):
    if weights is None:
        sd = measures.biweight_midvariance(x)
    else:
        sd = measures.weighted_std(x, weights)
    width = 4 * sd * len(x) ** (4 / 5)
    width = max(3, int(round(width)))
    width = min(len(x), width)
    return width


def kaiser(x, width=None, weights=None, do_fit_edges=False):
    if len(x) < 2:
        return x
    if width is None:
        width = guess_window_size(x, weights)
    x, wing, *padded = check_inputs(x, width, False, weights)

    window = np.kaiser(2 * wing + 1, 14)
    if weights is None:
        signal, = padded
        y = convolve_unweighted(window, signal, wing)
    else:
        signal, weights = padded
        y, _w = convolve_weighted(window, signal, weights)
    if do_fit_edges:
        _fit_edges(x, y, wing)
    return y


def savgol(x, total_width=None, weights=None,
           window_width=7, order=3, n_iter=1):
    if len(x) < 2:
        return x

    if total_width:
        n_iter = max(1, min(1000, total_width // window_width))
    else:
        total_width = n_iter * window_width
    logging.debug("adjusting in %d iterations for effective bandwidth %d",
                  n_iter, total_width)

    if weights is None:
        x, total_wing, signal = check_inputs(x, total_width, False)
        y = signal
        for _i in range(n_iter):
            y = savgol_filter(y, window_width, order, mode='interp')

    else:

        x, total_wing, signal, weights = check_inputs(x, total_width, False, weights)
        window = savgol_coeffs(window_width, order)
        y, w = convolve_weighted(window, signal, weights, n_iter)

    bad_idx = (y > x.max()) | (y < x.min())
    if bad_idx.any():
        logging.warning("adjusting overshot at {} / {} indices: "
                        "({}, {}) vs. original ({}, {})"
                        .format(bad_idx.sum(), len(bad_idx),
                                y.min(), y.max(),
                                x.min(), x.max()))
    return y[total_wing:-total_wing]


def _fit_edges(x, y, wing, polyorder=3):
    window_length = 2 * wing + 1
    n = len(x)

    _fit_edge(x, y, 0, window_length, 0, wing, polyorder)
    _fit_edge(x, y, n - window_length, n, n - wing, n, polyorder)


def _fit_edge(x, y, window_start, window_stop, interp_start, interp_stop,
              polyorder):
    x_edge = x[window_start:window_stop]

    poly_coeffs = np.polyfit(np.arange(0, window_stop - window_start),
                             x_edge, polyorder)

    i = np.arange(interp_start - window_start, interp_stop - window_start)
    values = np.polyval(poly_coeffs, i)

    y[interp_start:interp_stop] = values


def outlier_iqr(a, c=3.0):
    a = np.asarray(a)
    dists = np.abs(a - np.median(a))
    iqr = measures.interquartile_range(a)
    return dists > (c * iqr)


def outlier_mad_median(a):
    K = 2.24

    a = np.asarray(a)
    dists = np.abs(a - np.median(a))
    mad = measures.median_absolute_deviation(a)
    return (dists / mad) > K


def rolling_outlier_iqr(x, width, c=3.0):
    if len(x) <= width:
        return np.zeros(len(x), dtype=np.bool_)
    dists = x - savgol(x, width)
    q_hi = rolling_quantile(dists, width, .75)
    q_lo = rolling_quantile(dists, width, .25)
    iqr = q_hi - q_lo
    outliers = (np.abs(dists) > iqr * c)
    return outliers


def rolling_outlier_quantile(x, width, q, m):
    if len(x) <= width:
        return np.zeros(len(x), dtype=np.bool_)
    dists = np.abs(x - savgol(x, width))
    quants = rolling_quantile(dists, width, q)
    outliers = (dists > quants * m)
    return outliers


def rolling_outlier_std(x, width, stdevs):
    if len(x) <= width:
        return np.zeros(len(x), dtype=np.bool_)
    dists = x - savgol(x, width)
    x_std = rolling_std(dists, width)
    outliers = (np.abs(dists) > x_std * stdevs)
    return outliers
