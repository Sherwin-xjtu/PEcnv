import logging
import os
import tempfile

import numpy as np
import pandas as pd
from pegeno import tabio, GenomicArray as GA

from . import coverInfo, parsingBam
from .offTarget import compare_chrom_names
from .measures import weighted_median


def midsize_file(fnames):
    return sorted(fnames, key=lambda f: os.stat(f).st_size
                  )[len(fnames) // 2 - 1]


def do_autobin(bam_fname, method, targets=None, access=None,
               bp_per_bin=100000., target_min_size=20, target_max_size=50000,
               offTarget_min_size=500, offTarget_max_size=1000000, fasta=None):
    if method in ('amplicon', 'hybrid'):
        if targets is None:
            raise ValueError("Target regions are required for method %r "
                             "but were not provided." % method)
        if not len(targets):
            raise ValueError("Target regions are required for method %r "
                             "but were not provided." % method)

    def depth2binsize(depth, min_size, max_size):
        if depth:
            bin_size = int(round(bp_per_bin / depth))
            if bin_size < min_size:
                logging.info("Limiting est. bin size %d to given min. %d",
                             bin_size, min_size)
                bin_size = min_size
            elif bin_size > max_size:
                logging.info("Limiting est. bin size %d to given max. %d",
                             bin_size, max_size)
                bin_size = max_size
            return bin_size

    parsingBam.ensure_bam_index(bam_fname)
    rc_table = parsingBam.idxstats(bam_fname, drop_unmapped=True, fasta=fasta)
    read_len = parsingBam.get_read_length(bam_fname, fasta=fasta)
    logging.info("Estimated read length %s", read_len)

    if method == 'amplicon':

        tgt_depth = sample_region_cov(bam_fname, targets, fasta=fasta)
        anti_depth = None
    elif method == 'hybrid':
        tgt_depth, anti_depth = hybrid(rc_table, read_len, bam_fname, targets,
                                       access, fasta)
    elif method == 'wgs':
        if access is not None and len(access):
            rc_table = update_chrom_length(rc_table, access)
        tgt_depth = average_depth(rc_table, read_len)
        anti_depth = None

    tgt_bin_size = depth2binsize(tgt_depth, target_min_size, target_max_size)
    anti_bin_size = depth2binsize(anti_depth, offTarget_min_size,
                                  offTarget_max_size)
    return ((tgt_depth, tgt_bin_size),
            (anti_depth, anti_bin_size))


def hybrid(rc_table, read_len, bam_fname, targets, access=None, fasta=None):
    if access is None:
        access = idxstats2ga(rc_table, bam_fname)

        compare_chrom_names(access, targets)
    offTargets = access.subtract(targets)

    rc_table, targets, offTargets = shared_chroms(rc_table, targets,
                                                  offTargets)

    target_depth = sample_region_cov(bam_fname, targets, fasta=fasta)

    target_length = region_size_by_chrom(targets)['length']
    target_reads = (target_length * target_depth / read_len).values
    anti_table = update_chrom_length(rc_table, offTargets)
    anti_table = anti_table.assign(mapped=anti_table.mapped - target_reads)
    anti_depth = average_depth(anti_table, read_len)
    return target_depth, anti_depth


def average_depth(rc_table, read_length):
    mean_depths = read_length * rc_table.mapped / rc_table.length
    return weighted_median(mean_depths, rc_table.length)


def idxstats2ga(table, bam_fname):
    return GA(table.assign(start=0, end=table.length)
              .loc[:, ('chromosome', 'start', 'end')],
              meta_dict={'filename': bam_fname})


def sample_region_cov(bam_fname, regions, max_num=100, fasta=None):
    midsize_regions = sample_midsize_regions(regions, max_num)
    with tempfile.NamedTemporaryFile(suffix='.bed', mode='w+t') as f:
        tabio.write(regions.as_dataframe(midsize_regions), f, 'bed4')
        f.flush()
        table = coverInfo.bedcov(f.name, bam_fname, 0, fasta)

    return table.basecount.sum() / (table.end - table.start).sum()


def sample_midsize_regions(regions, max_num):
    sizes = regions.end - regions.start
    lo_size, hi_size = np.percentile(sizes[sizes > 0], [25, 75])
    midsize_regions = regions.data[(sizes >= lo_size) & (sizes <= hi_size)]
    if len(midsize_regions) > max_num:
        midsize_regions = midsize_regions.sample(max_num, random_state=0xA5EED)
    return midsize_regions


def shared_chroms(*tables):
    chroms = tables[0].chromosome.drop_duplicates()
    for tab in tables[1:]:
        if tab is not None:
            new_chroms = tab.chromosome.drop_duplicates()
            chroms = chroms[chroms.isin(new_chroms)]
    return [None if tab is None else tab[tab.chromosome.isin(chroms)]
            for tab in tables]


def update_chrom_length(rc_table, regions):
    if regions is not None and len(regions):
        chrom_sizes = region_size_by_chrom(regions)
        rc_table = rc_table.merge(chrom_sizes, on='chromosome', how='inner')
        rc_table['length'] = rc_table['length_y']
        rc_table = rc_table.drop(['length_x', 'length_y'], axis=1)
    return rc_table


def region_size_by_chrom(regions):
    chromgroups = regions.data.groupby('chromosome', sort=False)

    sizes = [total_region_size(g) for _key, g in chromgroups]
    return pd.DataFrame({'chromosome': regions.chromosome.drop_duplicates(),
                         'length': sizes})


def total_region_size(regions):
    return (regions.end - regions.start).sum()
