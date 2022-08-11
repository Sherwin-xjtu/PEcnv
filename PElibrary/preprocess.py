import logging

import numpy as np
from pegeno import tabio, GenomicArray as GA


def do_preprocess(fa_fname, exclude_fnames=(), min_gap_size=5000,
                  skip_noncanonical=True):
    fa_regions = get_regions(fa_fname)
    if skip_noncanonical:
        fa_regions = drop_noncanonical_contigs(fa_regions)
    preprocess_regions = GA.from_rows(fa_regions)
    for ex_fname in exclude_fnames:
        excluded = tabio.read(ex_fname, 'bed3')
        preprocess_regions = preprocess_regions.subtract(excluded)
    return GA.from_rows(join_regions(preprocess_regions, min_gap_size))


def drop_noncanonical_contigs(region_tups):
    from .offTarget import is_canonical_contig_name
    return (tup for tup in region_tups
            if is_canonical_contig_name(tup[0]))


def get_regions(fasta_fname):
    with open(fasta_fname) as infile:
        chrom = cursor = run_start = None
        for line in infile:
            if line.startswith('>'):

                if run_start is not None:
                    yield log_this(chrom, run_start, cursor)

                chrom = line.split(None, 1)[0][1:]
                run_start = None
                cursor = 0
            else:
                line = line.rstrip()
                if 'N' in line:
                    if all(c == 'N' for c in line):

                        if run_start is not None:
                            yield log_this(chrom, run_start, cursor)
                            run_start = None
                    else:

                        line_chars = np.array(line, dtype='c')
                        n_indices = np.where(line_chars == b'N')[0]

                        if run_start is not None:
                            yield log_this(chrom, run_start, cursor + n_indices[0])
                        elif n_indices[0] != 0:
                            yield log_this(chrom, cursor, cursor + n_indices[0])

                        gap_mask = np.diff(n_indices) > 1
                        if gap_mask.any():
                            ok_starts = n_indices[:-1][gap_mask] + 1 + cursor
                            ok_ends = n_indices[1:][gap_mask] + cursor
                            for start, end in zip(ok_starts, ok_ends):
                                yield log_this(chrom, start, end)

                        if n_indices[-1] + 1 < len(line_chars):
                            run_start = cursor + n_indices[-1] + 1
                        else:
                            run_start = None
                else:
                    if run_start is None:
                        run_start = cursor
                cursor += len(line)

        if run_start is not None:
            yield log_this(chrom, run_start, cursor)


def log_this(chrom, run_start, run_end):
    return (chrom, run_start, run_end)


def join_regions(regions, min_gap_size):
    min_gap_size = min_gap_size or 0
    for chrom, rows in regions.by_chromosome():

        coords = iter(zip(rows['start'], rows['end']))
        prev_start, prev_end = next(coords)
        for start, end in coords:
            gap = start - prev_end
            assert gap > 0, ("Impossible gap between %s %d-%d and %d-%d (=%d)"
                             % (chrom, prev_start, prev_end, start, end, gap))
            if gap < min_gap_size:

                prev_end = end
            else:

                yield (chrom, prev_start, prev_end)
                prev_start, prev_end = start, end
        yield (chrom, prev_start, prev_end)
