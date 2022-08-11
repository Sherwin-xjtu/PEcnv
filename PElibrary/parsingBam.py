import logging
import os
from itertools import islice

import numpy as np
import pandas as pd
import pysam
from io import StringIO
from pathlib import Path, PurePath


def idxstats(bam_fname, drop_unmapped=False, fasta=None):
    handle = StringIO(pysam.idxstats(bam_fname, split_lines=False, reference_filename=fasta))
    table = pd.read_csv(handle, sep='\t', header=None,
                        names=['chromosome', 'length', 'mapped', 'unmapped'])
    if drop_unmapped:
        table = table[table.mapped != 0].drop('unmapped', axis=1)
    return table


def bam_total_reads(bam_fname, fasta=None):
    table = idxstats(bam_fname, drop_unmapped=True, fasta=fasta)
    return table.mapped.sum()


def ensure_bam_index(bam_fname):
    if PurePath(bam_fname).suffix == ".cram":
        if os.path.isfile(bam_fname + '.crai'):

            bai_fname = bam_fname + '.crai'
        else:

            bai_fname = bam_fname[:-1] + 'i'
        if not is_newer_than(bai_fname, bam_fname):
            logging.info("Indexing CRAM file %s", bam_fname)
            pysam.index(bam_fname)
            bai_fname = bam_fname + '.crai'
        assert os.path.isfile(bai_fname), \
            "Failed to generate cram index " + bai_fname
    else:
        if os.path.isfile(bam_fname + '.bai'):

            bai_fname = bam_fname + '.bai'
        else:

            bai_fname = bam_fname[:-1] + 'i'
        if not is_newer_than(bai_fname, bam_fname):
            logging.info("Indexing BAM file %s", bam_fname)
            pysam.index(bam_fname)
            bai_fname = bam_fname + '.bai'
        assert os.path.isfile(bai_fname), \
            "Failed to generate bam index " + bai_fname
    return bai_fname


def ensure_bam_sorted(bam_fname, by_name=False, span=50, fasta=None):
    if by_name:

        def out_of_order(read, prev):
            return not (prev is None or
                        prev.qname <= read.qname)
    else:

        def out_of_order(read, prev):
            return not (prev is None or
                        read.tid != prev.tid or
                        prev.pos <= read.pos)

    bam = pysam.Samfile(bam_fname, 'rb', reference_filename=fasta)
    last_read = None
    for read in islice(bam, span):
        if out_of_order(read, last_read):
            return False
        last_read = read
    bam.close()
    return True


def is_newer_than(target_fname, orig_fname):
    if not os.path.isfile(target_fname):
        return False
    return (os.stat(target_fname).st_mtime >= os.stat(orig_fname).st_mtime)


def get_read_length(bam, span=1000, fasta=None):
    was_open = False
    if isinstance(bam, str):
        bam = pysam.Samfile(bam, 'rb', reference_filename=fasta)
    else:
        was_open = True
    lengths = [read.query_length for read in islice(bam, span)
               if read.query_length > 0]
    if was_open:
        bam.seek(0)
    else:
        bam.close()
    return np.median(lengths)
