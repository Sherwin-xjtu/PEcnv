import sys
from functools import wraps

import numpy as np
from scipy import stats


def on_array(default=None):
    def outer(f):
        @wraps(f)
        def wrapper(a, **kwargs):
            a = np.asfarray(a)
            a = a[~np.isnan(a)]
            if not len(a):
                return np.nan
            if len(a) == 1:
                if default is None:
                    return a[0]
                return default
            return f(a, **kwargs)

        return wrapper

    return outer


def on_weighted_array(default=None):
    def outer(f):
        @wraps(f)
        def wrapper(a, w, **kwargs):
            if len(a) != len(w):
                raise ValueError("Unequal array lengths: a=%d, w=%d"
                                 % (len(a), len(w)))
            if not len(a):
                return np.nan
            a = np.asfarray(a)
            w = np.asfarray(w)

            a_nan = np.isnan(a)
            if a_nan.any():
                a = a[~a_nan]
                if not len(a):
                    return np.nan
                w = w[~a_nan]
            if len(a) == 1:
                if default is None:
                    return a[0]
                return default

            w_nan = np.isnan(w)
            if w_nan.any():
                w[w_nan] = 0.0
            return f(a, w, **kwargs)

        return wrapper

    return outer


@on_array()
def biweight_location(a, initial=None, c=6.0, epsilon=1e-3, max_iter=5):
    def biloc_iter(a, initial):

        d = a - initial
        mad = np.median(np.abs(d))
        w = d / max(c * mad, epsilon)
        w = (1 - w ** 2) ** 2

        mask = (w < 1)
        weightsum = w[mask].sum()
        if weightsum == 0:
            return initial
        return initial + (d[mask] * w[mask]).sum() / weightsum

    if initial is None:
        initial = np.median(a)
    for _i in range(max_iter):
        result = biloc_iter(a, initial)
        if abs(result - initial) <= epsilon:
            break
        initial = result
    return result


@on_array()
def modal_location(a):
    sarr = np.sort(a)
    kde = stats.gaussian_kde(sarr)
    y = kde.evaluate(sarr)
    peak = sarr[y.argmax()]
    return peak


@on_weighted_array()
def weighted_median(a, weights):
    order = a.argsort()
    a = a[order]
    weights = weights[order]
    midpoint = 0.5 * weights.sum()
    if (weights > midpoint).any():
        return a[weights.argmax()]
    cumulative_weight = weights.cumsum()
    midpoint_idx = cumulative_weight.searchsorted(midpoint)
    if (midpoint_idx > 0 and
            cumulative_weight[midpoint_idx - 1] - midpoint < sys.float_info.epsilon):
        return a[midpoint_idx - 1: midpoint_idx + 1].mean()
    return a[midpoint_idx]


@on_array(0)
def biweight_midvariance(a, initial=None, c=9.0, epsilon=1e-3):
    if initial is None:
        initial = biweight_location(a)

    d = a - initial

    mad = np.median(np.abs(d))
    w = d / max(c * mad, epsilon)

    mask = np.abs(w) < 1
    if w[mask].sum() == 0:
        return mad * 1.4826
    n = mask.sum()
    d_ = d[mask]
    w_ = (w ** 2)[mask]
    return np.sqrt((n * (d_ ** 2 * (1 - w_) ** 4).sum())
                   / (((1 - w_) * (1 - 5 * w_)).sum() ** 2))


@on_array(0)
def gapper_scale(a):
    gaps = np.diff(np.sort(a))
    n = len(a)
    idx = np.arange(1, n)
    weights = idx * (n - idx)
    return (gaps * weights).sum() * np.sqrt(np.pi) / (n * (n - 1))


@on_array(0)
def interquartile_range(a):
    return np.percentile(a, 75) - np.percentile(a, 25)


@on_array(0)
def median_absolute_deviation(a, scale_to_sd=True):
    a_median = np.median(a)
    mad = np.median(np.abs(a - a_median))
    if scale_to_sd:
        mad *= 1.4826
    return mad


@on_weighted_array()
def weighted_mad(a, weights, scale_to_sd=True):
    a_median = weighted_median(a, weights)
    mad = weighted_median(np.abs(a - a_median), weights)
    if scale_to_sd:
        mad *= 1.4826
    return mad


@on_weighted_array()
def weighted_std(a, weights):
    mean = np.average(a, weights=weights)
    var = np.average((a - mean) ** 2, weights=weights)
    return np.sqrt(var)


@on_array(0)
def mean_squared_error(a, initial=None):
    if initial is None:
        initial = a.mean()
    if initial:
        a = a - initial
    return (a ** 2).mean()


@on_array(0)
def q_n(a):
    vals = []
    for i, x_i in enumerate(a):
        for x_j in a[i + 1:]:
            vals.append(abs(x_i - x_j))
    quartile = np.percentile(vals, 25)

    n = len(a)
    if n <= 10:

        scale = 1.392
    elif 10 < n < 400:

        scale = 1.0 + (4 / n)
    else:
        scale = 1.0

    return quartile / scale
