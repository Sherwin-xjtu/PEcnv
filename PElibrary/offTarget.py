import logging
import re

from pegeno import GenomicArray as GA

from .hyperparameters import INSERT_SIZE, MIN_REF_COVERAGE, OFFTARGET_NAME


def do_offTarget(targets, preprocess=None, avg_bin_size=150000,
                 min_bin_size=None):
    if not min_bin_size:
        min_bin_size = 2 * int(avg_bin_size * (2 ** MIN_REF_COVERAGE))
    return get_offTargets(targets, preprocess, avg_bin_size, min_bin_size)


def get_offTargets(targets, preprocessible, avg_bin_size, min_bin_size):
    if preprocessible:
        preprocessible = drop_noncanonical_contigs(preprocessible, targets)
    else:
        TELOMERE_SIZE = 150000
        preprocessible = guess_chromosome_regions(targets, TELOMERE_SIZE)
    pad_size = 2 * INSERT_SIZE
    bg_arr = (preprocessible.resize_ranges(-pad_size)
              .subtract(targets.resize_ranges(pad_size))
              .subdivide(avg_bin_size, min_bin_size))
    bg_arr['gene'] = OFFTARGET_NAME
    return bg_arr


def drop_noncanonical_contigs(preprocessible, targets, verbose=True):
    preprocess_chroms, target_chroms = compare_chrom_names(preprocessible, targets)

    untgt_chroms = preprocess_chroms - target_chroms

    if any(is_canonical_contig_name(c) for c in target_chroms):
        chroms_to_skip = [c for c in untgt_chroms
                          if not is_canonical_contig_name(c)]
    else:

        max_tgt_chr_name_len = max(map(len, target_chroms))
        chroms_to_skip = [c for c in untgt_chroms
                          if len(c) > max_tgt_chr_name_len]
    if chroms_to_skip:
        skip_idx = preprocessible.chromosome.isin(chroms_to_skip)
        preprocessible = preprocessible[~skip_idx]
    return preprocessible


def compare_chrom_names(a_regions, b_regions):
    a_chroms = set(a_regions.chromosome.unique())
    b_chroms = set(b_regions.chromosome.unique())
    if a_chroms and a_chroms.isdisjoint(b_chroms):
        msg = "Chromosome names do not match between files"
        a_fname = a_regions.meta.get('filename')
        b_fname = b_regions.meta.get('filename')
        if a_fname and b_fname:
            msg += " {} and {}".format(a_fname, b_fname)
        msg += ": {} vs. {}".format(', '.join(map(repr, sorted(a_chroms)[:3])),
                                    ', '.join(map(repr, sorted(b_chroms)[:3])))
        raise ValueError(msg)
    return a_chroms, b_chroms


def guess_chromosome_regions(targets, telomere_size):
    endpoints = [subarr.end.iat[-1] for _c, subarr in targets.by_chromosome()]
    whole_chroms = GA.from_columns({
        'chromosome': targets.chromosome.drop_duplicates(),
        'start': telomere_size,
        'end': endpoints})
    return whole_chroms


re_canonical = re.compile(r"(chr)?(\d+|[XYxy])$")

re_noncanonical = re.compile("|".join((r"^chrEBV$",
                                       r"^NC|_random$",
                                       r"Un_",
                                       r"^HLA\-",
                                       r"_alt$",
                                       r"hap\d$",
                                       r"chrM",
                                       r"MT")))


def is_canonical_contig_name(name):
    return not re_noncanonical.search(name)


def _drop_short_contigs(garr):
    from .plots import chromosome_sizes
    from pegeno.chromsort import detect_big_chroms

    chrom_sizes = chromosome_sizes(garr)
    n_big, thresh = detect_big_chroms(chromosome_sizes.values())
    chrom_names_to_keep = {c for c, s in chrom_sizes.items()
                           if s >= thresh}
    assert len(chrom_names_to_keep) == n_big
    return garr[garr.chromosome.isin(chrom_names_to_keep)]
