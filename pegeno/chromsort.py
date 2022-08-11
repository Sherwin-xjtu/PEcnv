from itertools import takewhile

import numpy as np
import pandas as pd


def detect_big_chroms(sizes):

    sizes = pd.Series(sizes).sort_values(ascending=False)
    reldiff = sizes.diff().abs().values[1:] / sizes.values[:-1]
    changepoints = np.nonzero(reldiff > .5)[0]
    if changepoints.any():
        n_big = changepoints[0] + 1
        thresh = sizes.iat[n_big - 1]
    else:
        n_big = len(sizes)
        thresh = sizes[-1]
    return n_big, thresh


def sorter_chrom(label):

    chrom = (label[3:] if label.lower().startswith('chr')
             else label)
    if chrom in ('X', 'Y'):
        key = (1000, chrom)
    else:
        # Separate numeric and special chromosomes
        nums = ''.join(takewhile(str.isdigit, chrom))
        chars = chrom[len(nums):]
        nums = int(nums) if nums else 0
        if not chars:
            key = (nums, '')
        elif len(chars) == 1:
            key = (2000 + nums, chars)
        else:
            key = (3000 + nums, chars)
    return key
