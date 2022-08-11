import collections
import itertools
import logging
import math

import numpy as np

from . import kernel, hyperparameters
from pegeno.rangelabel import unpack_range, Region

MB = 1e-6


def plot_x_dividers(axis, chrom_sizes, pad=None):
    assert isinstance(chrom_sizes, collections.OrderedDict)
    if pad is None:
        pad = 0.003 * sum(chrom_sizes.values())
    x_dividers = []
    x_centers = []
    x_starts = collections.OrderedDict()
    curr_offset = pad
    for label, size in list(chrom_sizes.items()):
        x_starts[label] = curr_offset
        x_centers.append(curr_offset + 0.5 * size)
        x_dividers.append(curr_offset + size + pad)
        curr_offset += size + 2 * pad

    axis.set_xlim(0, curr_offset)
    for xposn in x_dividers[:-1]:
        axis.axvline(x=xposn, color='k')

    axis.set_xticks(x_centers)
    axis.set_xticklabels(list(chrom_sizes.keys()), rotation=60)
    axis.tick_params(labelsize='small')
    axis.tick_params(axis='x', length=0)
    axis.get_yaxis().tick_left()

    return x_starts


def chromosome_sizes(probes, to_mb=False):
    chrom_sizes = collections.OrderedDict()
    for chrom, rows in probes.by_chromosome():
        chrom_sizes[chrom] = rows['end'].max()
        if to_mb:
            chrom_sizes[chrom] *= MB
    return chrom_sizes


def translate_region_to_bins(region, bins):
    if region is None:
        return Region(None, None, None)
    chrom, start, end = unpack_range(region)
    if start is None and end is None:
        return Region(chrom, start, end)
    if start is None:
        start = 0
    if end is None:
        end = float("inf")

    c_bin_starts = bins.data.loc[bins.data.chromosome == chrom, "start"].values
    r_start, r_end = np.searchsorted(c_bin_starts, [start, end])
    return Region(chrom, r_start, r_end)


def translate_segments_to_bins(segments, bins):
    if "probes" in segments and segments["probes"].sum() == len(bins):

        return update_binwise_positions_simple(segments)
    else:
        logging.warning("Segments %s 'probes' sum does not match the number "
                        "of bins in %s", segments.sample_id, bins.sample_id)

        _x, segments, _v = update_binwise_positions(bins, segments)
        return segments


def update_binwise_positions_simple(cnarr):
    start_chunks = []
    end_chunks = []
    is_segment = ("probes" in cnarr)
    if is_segment:
        cnarr = cnarr[cnarr["probes"] > 0]
    for _chrom, c_arr in cnarr.by_chromosome():
        if is_segment:

            ends = c_arr["probes"].values.cumsum()
            starts = np.r_[0, ends[:-1]]
        else:

            n_bins = len(c_arr)
            starts = np.arange(n_bins)
            ends = np.arange(1, n_bins + 1)
        start_chunks.append(starts)
        end_chunks.append(ends)
    return cnarr.as_dataframe(
        cnarr.data.assign(start=np.concatenate(start_chunks),
                          end=np.concatenate(end_chunks)))


def update_binwise_positions(cnarr, segments=None, variants=None):
    cnarr = cnarr.copy()
    if segments:
        segments = segments.copy()
        seg_chroms = set(segments.chromosome.unique())
    if variants:
        variants = variants.copy()
        var_chroms = set(variants.chromosome.unique())

    for chrom in cnarr.chromosome.unique():

        c_idx = (cnarr.chromosome == chrom)
        c_bins = cnarr[c_idx]
        if segments and chrom in seg_chroms:
            c_seg_idx = (segments.chromosome == chrom).values
            seg_starts = np.searchsorted(c_bins.start.values,
                                         segments.start.values[c_seg_idx])
            seg_ends = np.r_[seg_starts[1:], len(c_bins)]
            segments.data.loc[c_seg_idx, "start"] = seg_starts
            segments.data.loc[c_seg_idx, "end"] = seg_ends

        if variants and chrom in var_chroms:

            c_varr_idx = (variants.chromosome == chrom).values
            c_varr_df = variants.data[c_varr_idx]
            v_starts = np.searchsorted(c_bins.start.values,
                                       c_varr_df.start.values)

            for idx, size in list(get_repeat_slices(v_starts)):
                v_starts[idx] += np.arange(size) / size
            variant_sizes = c_varr_df.end - c_varr_df.start
            variants.data.loc[c_varr_idx, "start"] = v_starts
            variants.data.loc[c_varr_idx, "end"] = v_starts + variant_sizes

        c_starts = np.arange(len(c_bins))
        c_ends = np.arange(1, len(c_bins) + 1)
        cnarr.data.loc[c_idx, "start"] = c_starts
        cnarr.data.loc[c_idx, "end"] = c_ends

    return cnarr, segments, variants


def get_repeat_slices(values):
    offset = 0
    for idx, (_val, rpt) in enumerate(itertools.groupby(values)):
        size = len(list(rpt))
        if size > 1:
            i = idx + offset
            slc = slice(i, i + size)
            yield slc, size
            offset += size - 1


def cvg2rgb(cvg, desaturate):
    cutoff = 1.33
    x = min(abs(cvg) / cutoff, 1.0)
    if desaturate:

        x = ((1. - math.cos(x * math.pi)) / 2.) ** 0.8
        s = x ** 1.2
    else:
        s = x
    if cvg < 0:
        rgb = (1 - s, 1 - s, 1 - .25 * x)
    else:
        rgb = (1 - .25 * x, 1 - s, 1 - s)
    return rgb


def gene_coords_by_name(probes, names):
    names = list(filter(None, set(names)))
    if not names:
        return {}

    gene_index = collections.defaultdict(set)
    for i, gene in enumerate(probes['gene']):
        for gene_name in gene.split(','):
            if gene_name in names:
                gene_index[gene_name].add(i)

    all_coords = collections.defaultdict(lambda: collections.defaultdict(set))
    for name in names:
        gene_probes = probes.data.take(sorted(gene_index.get(name, [])))
        if not len(gene_probes):
            raise ValueError("No targeted gene named '%s' found" % name)

        start = gene_probes['start'].min()
        end = gene_probes['end'].max()
        chrom = kernel.check_unique(gene_probes['chromosome'], name)

        uniq_names = set()
        for oname in set(gene_probes['gene']):
            uniq_names.update(oname.split(','))
        all_coords[chrom][start, end].update(uniq_names)

    uniq_coords = {}
    for chrom, hits in all_coords.items():
        uniq_coords[chrom] = [(start, end, ",".join(sorted(gene_names)))
                              for (start, end), gene_names in hits.items()]
    return uniq_coords


def gene_coords_by_range(probes, chrom, start, end,
                         ignore=hyperparameters.IGNORE_GENE_NAMES):
    """Find the chromosomal position of all genes in a range.

    Returns
    -------
    dict
        Of: {chromosome: [(start, end, gene), ...]}
    """
    ignore += hyperparameters.OFFTARGET_ALIASES

    genes = collections.OrderedDict()
    for row in probes.in_range(chrom, start, end):
        name = str(row.gene)
        if name in genes:
            genes[name][1] = row.end
        elif name not in ignore:
            genes[name] = [row.start, row.end]

    return {chrom: [(gstart, gend, name)
                    for name, (gstart, gend) in list(genes.items())]}
