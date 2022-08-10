import itertools

import numpy as np
import pandas as pd

from .chromsort import sorter_chrom
from .combiners import get_combiners, first_of


def flatten(table, combine=None, split_columns=None):
    if not len(table):
        return table
    if (table.start.values[1:] >= table.end.cummax().values[:-1]).all():
        return table
    # NB: Input rows and columns should already be sorted like this
    table = table.sort_values(['chromosome', 'start', 'end'])
    cmb = get_combiners(table, False, combine)
    out = (table.groupby(by='chromosome',
                         as_index=False, group_keys=False, sort=False)
           .apply(_flatten_overlapping, cmb, split_columns)
           .reset_index(drop=True))
    return out.reindex(out.chromosome.apply(sorter_chrom)
                       .sort_values(kind='mergesort').index)


def _flatten_overlapping(table, combine, split_columns):
    """Merge overlapping regions within a chromosome/strand.

    Assume chromosome and (if relevant) strand are already identical, so only
    start and end coordinates are considered.
    """
    if split_columns:
        row_groups = (tuple(_flatten_tuples_split(row_group, combine,
                                                  split_columns))
                      for row_group in _nonoverlapping_groups(table, 0))
    else:
        row_groups = (tuple(_flatten_tuples(row_group, combine))
                      for row_group in _nonoverlapping_groups(table, 0))
    all_row_groups = itertools.chain(*row_groups)
    return pd.DataFrame.from_records(list(all_row_groups),
                                     columns=table.columns)


def _flatten_tuples(keyed_rows, combine):

    rows = [kr[1] for kr in keyed_rows]
    first_row = rows[0]
    if len(rows) == 1:
        yield first_row
    else:
        # TODO speed this up! Bottleneck is in dictcomp
        extra_cols = [x for x in first_row._fields[3:] if x in combine]
        breaks = sorted(set(itertools.chain(*[(r.start, r.end) for r in rows])))
        for bp_start, bp_end in zip(breaks[:-1], breaks[1:]):
            # Find the row(s) overlapping this region
            # i.e. any not already seen and not already passed
            rows_in_play = [row for row in rows
                            if row.start <= bp_start and row.end >= bp_end]
            # Combine the extra fields of the overlapping regions
            extra_fields = {key: combine[key]([getattr(r, key)
                                               for r in rows_in_play])
                            for key in extra_cols}
            yield first_row._replace(start=bp_start, end=bp_end,
                                     **extra_fields)


def _flatten_tuples_split(keyed_rows, combine, split_columns):

    rows = [kr[1] for kr in keyed_rows]
    first_row = rows[0]
    if len(rows) == 1:
        yield first_row
    else:
        # TODO - use split_columns
        extra_cols = [x for x in first_row._fields[3:] if x in combine]
        breaks = sorted(set(itertools.chain(*[(r.start, r.end) for r in rows])))
        for bp_start, bp_end in zip(breaks[:-1], breaks[1:]):
            # Find the row(s) overlapping this region
            # i.e. any not already seen and not already passed
            rows_in_play = [row for row in rows
                            if row.start <= bp_start and row.end >= bp_end]
            # Combine the extra fields of the overlapping regions
            extra_fields = {key: combine[key]([getattr(r, key)
                                               for r in rows_in_play])
                            for key in extra_cols}
            yield first_row._replace(start=bp_start, end=bp_end,
                                     **extra_fields)



def merge(table, bp=0, stranded=False, combine=None):
    """Merge overlapping rows in a DataFrame."""
    if not len(table):
        return table
    gap_sizes = table.start.values[1:] - table.end.cummax().values[:-1]
    if (gap_sizes > -bp).all():
        return table
    if stranded:
        groupkey = ['chromosome', 'strand']
    else:
        # NB: same gene name can appear on alt. contigs
        groupkey = ['chromosome']
    table = table.sort_values(groupkey + ['start', 'end'])
    cmb = get_combiners(table, stranded, combine)
    out = (table.groupby(by=groupkey,
                         as_index=False, group_keys=False, sort=False)
           .apply(_merge_overlapping, bp, cmb)
           .reset_index(drop=True))
    # Re-sort chromosomes cleverly instead of lexicographically
    return out.reindex(out.chromosome.apply(sorter_chrom)
                       .sort_values(kind='mergesort').index)


def _merge_overlapping(table, bp, combine):

    merged_rows = [_squash_tuples(row_group, combine)
                   for row_group in _nonoverlapping_groups(table, bp)]
    return pd.DataFrame.from_records(merged_rows,
                                     columns=merged_rows[0]._fields)


def _nonoverlapping_groups(table, bp):

    gap_sizes = table.start.values[1:] - table.end.cummax().values[:-1]
    group_keys = np.r_[False, gap_sizes > (-bp)].cumsum()

    keyed_groups = zip(group_keys, table.itertuples(index=False))
    return (row_group
            for _key, row_group in itertools.groupby(keyed_groups, first_of))


def _squash_tuples(keyed_rows, combine):

    rows = [kr[1] for kr in keyed_rows] #list(rows)
    firsttup = rows[0]
    if len(rows) == 1:
        return firsttup
    newfields = {key: combiner([getattr(r, key) for r in rows])
                 for key, combiner in combine.items()}
    return firsttup._replace(**newfields)
