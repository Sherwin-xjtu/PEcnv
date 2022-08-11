import collections
import math
import sys

import numpy as np
import pandas as pd

from . import hyperparameters
from .segmIndicators import segment_mean


def do_breaks(probes, segments, min_probes=1):
    intervals = get_gene_intervals(probes)
    bpoints = get_breakpoints(intervals, segments, min_probes)
    return pd.DataFrame.from_records(bpoints,
                                     columns=['gene', 'chromosome',
                                              'location', 'change',
                                              'probes_left', 'probes_right'])


def get_gene_intervals(all_probes, ignore=hyperparameters.IGNORE_GENE_NAMES):
    ignore += hyperparameters.OFFTARGET_ALIASES

    gene_probes = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in all_probes:
        gname = str(row.gene)
        if gname not in ignore:
            gene_probes[row.chromosome][gname].append(row)

    intervals = collections.defaultdict(list)
    for chrom, gp in gene_probes.items():
        for gene, probes in gp.items():
            starts = sorted(row.start for row in probes)
            end = max(row.end for row in probes)
            intervals[chrom].append((gene, starts, end))
        intervals[chrom].sort(key=lambda gse: gse[1])
    return intervals


def get_breakpoints(intervals, segments, min_probes):
    breakpoints = []
    for i, curr_row in enumerate(segments[:-1]):
        curr_chrom = curr_row.chromosome
        curr_end = curr_row.end
        next_row = segments[i + 1]

        if next_row.chromosome != curr_chrom:
            continue
        for gname, gstarts, gend in intervals[curr_chrom]:
            if gstarts[0] < curr_end < gend:
                probes_left = sum(s < curr_end for s in gstarts)
                probes_right = sum(s >= curr_end for s in gstarts)
                if probes_left >= min_probes and probes_right >= min_probes:
                    breakpoints.append(
                        (gname, curr_chrom, int(math.ceil(curr_end)),
                         next_row.log2 - curr_row.log2,
                         probes_left, probes_right))
    breakpoints.sort(key=lambda row: (min(row[4], row[5]), abs(row[3])),
                     reverse=True)
    return breakpoints


def do_genemetrics(cnarr, segments=None, threshold=0.2, min_probes=3,
                   skip_low=False, male_reference=False, is_sample_female=None):
    if is_sample_female is None:
        is_sample_female = cnarr.guess_xx(male_reference=male_reference)
    cnarr = cnarr.shift_xx(male_reference, is_sample_female)
    if segments:
        segments = segments.shift_xx(male_reference, is_sample_female)
        rows = gene_metrics_by_segment(cnarr, segments, threshold, skip_low)
    else:
        rows = gene_metrics_by_gene(cnarr, threshold, skip_low)
    rows = list(rows)
    columns = (rows[0].index if len(rows) else cnarr._required_columns)
    columns = ["gene"] + [col for col in columns if col != "gene"]
    table = pd.DataFrame.from_records(rows).reindex(columns=columns)
    if min_probes and len(table):
        n_probes = (table.segment_probes
                    if 'segment_probes' in table.columns
                    else table.n_bins)
        table = table[n_probes >= min_probes]
    return table


def gene_metrics_by_gene(cnarr, threshold, skip_low=False):
    for row in group_by_genes(cnarr, skip_low):
        if abs(row.log2) >= threshold and row.gene:
            yield row


def gene_metrics_by_segment(cnarr, segments, threshold, skip_low=False):
    extra_cols = [col for col in segments.data.columns
                  if col not in cnarr.data.columns
                  and col not in ('depth', 'probes', 'weight')]
    for colname in extra_cols:
        cnarr[colname] = np.nan
    for segment, subprobes in cnarr.by_ranges(segments):
        if abs(segment.log2) >= threshold:
            for row in group_by_genes(subprobes, skip_low):
                row["log2"] = segment.log2
                if hasattr(segment, 'weight'):
                    row['segment_weight'] = segment.weight
                if hasattr(segment, 'probes'):
                    row['segment_probes'] = segment.probes
                for colname in extra_cols:
                    row[colname] = getattr(segment, colname)
                yield row


def group_by_genes(cnarr, skip_low):
    ignore = ('', np.nan) + hyperparameters.OFFTARGET_ALIASES
    for gene, rows in cnarr.by_gene():
        if not rows or gene in ignore:
            continue
        segmean = segment_mean(rows, skip_low)
        if segmean is None:
            continue
        outrow = rows[0].copy()
        outrow["end"] = rows.end.iat[-1]
        outrow["gene"] = gene
        outrow["log2"] = segmean
        outrow["n_bins"] = len(rows)
        if "weight" in rows:
            outrow["weight"] = rows["weight"].sum()
            if "depth" in rows:
                outrow["depth"] = np.average(rows["depth"],
                                             weights=rows["weight"])
        elif "depth" in rows:
            outrow["depth"] = rows["depth"].mean()
        yield outrow
