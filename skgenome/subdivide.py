import logging

import pandas as pd

from .merge import merge


def subdivide(table, avg_size, min_size=0, verbose=False):
    return pd.DataFrame.from_records(
        _split_targets(table, avg_size, min_size, verbose),
        columns=table.columns)


def _split_targets(regions, avg_size, min_size, verbose):

    for row in merge(regions).itertuples(index=False):
        span = row.end - row.start
        if span >= min_size:
            nbins = int(round(span / avg_size)) or 1
            if nbins == 1:
                yield row
            else:
                # Divide the region into equal-sized bins
                bin_size = span / nbins
                bin_start = row.start
                if verbose:
                    label = (row.gene if 'gene' in regions else
                             "%s:%d-%d" % (row.chromosome, row.start, row.end))
                    logging.info("Splitting: {:30} {:7} / {} = {:.2f}"
                                 .format(label, span, nbins, bin_size))
                for i in range(1, nbins):
                    bin_end = row.start + int(i * bin_size)
                    yield row._replace(start=bin_start, end=bin_end)
                    bin_start = bin_end
                yield row._replace(start=bin_start)
