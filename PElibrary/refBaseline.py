import collections
import logging

import numpy as np
import pandas as pd
import pyfaidx
from pegeno import tabio, GenomicArray as GA

from . import kernel, correct, measures, hyperparameters
from .implementation import read_cna
from .cnv import CopyNumArray as CNA


def do_refBaseline_flat(targets, offTargets=None, fa_fname=None,
                        male_refBaseline=False):
    ref_probes = bed2probes(targets)
    if offTargets:
        ref_probes.add(bed2probes(offTargets))

    ref_probes['log2'] = ref_probes.expect_flat_log2(male_refBaseline)
    ref_probes['depth'] = np.exp2(ref_probes['log2'])

    if fa_fname:
        gc, rmask = get_fasta_stats(ref_probes, fa_fname)
        ref_probes['gc'] = gc
        ref_probes['rmask'] = rmask
    else:
        logging.info("No FASTA refBaseline genome provided; "
                     "skipping GC, RM calculations")
    ref_probes.sort_columns()
    return ref_probes


def bed2probes(bed_fname):
    regions = tabio.read_auto(bed_fname)
    table = regions.data.loc[:, ("chromosome", "start", "end")]
    table["gene"] = (regions.data["gene"] if "gene" in regions.data else '-')
    table["log2"] = 0.0
    table["spread"] = 0.0
    return CNA(table, {"sample_id": kernel.fbase(bed_fname)})


def do_refBaseline(target_fnames, offTarget_fnames=None, fa_fname=None,
                   male_refBaseline=False, female_samples=None,
                   do_gc=True, do_edge=True, do_rmask=True, do_cluster=False,
                   min_cluster_size=4):
    if offTarget_fnames:
        kernel.assert_equal("Unequal number of target and offTarget files given",
                            targets=len(target_fnames),
                            offTargets=len(offTarget_fnames))
    if not fa_fname:
        logging.info("No FASTA refBaseline genome provided; "
                     "skipping GC, RM calculations")

    if female_samples is None:

        sexes = infer_sexes(target_fnames, False)
        if offTarget_fnames:
            a_sexes = infer_sexes(offTarget_fnames, False)
            for sid, a_is_xx in a_sexes.items():
                t_is_xx = sexes.get(sid)
                if t_is_xx is None:
                    sexes[sid] = a_is_xx
                elif t_is_xx != a_is_xx and a_is_xx is not None:

                    sexes[sid] = a_is_xx
    else:
        sexes = collections.defaultdict(lambda: female_samples)

    ref_probes = combine_probes(target_fnames, offTarget_fnames, fa_fname,
                                male_refBaseline, sexes, do_gc, do_edge, do_rmask,
                                do_cluster, min_cluster_size)
    warn_bad_bins(ref_probes)
    return ref_probes


def infer_sexes(cnn_fnames, is_haploid_x):
    sexes = {}
    for fname in cnn_fnames:
        cnarr = read_cna(fname)
        if cnarr:
            is_xx = cnarr.guess_xx(is_haploid_x)
            if is_xx is not None:
                sexes[cnarr.sample_id] = is_xx
    return sexes


def combine_probes(filenames, offTarget_fnames, fa_fname,
                   is_haploid_x, sexes, correct_gc, correct_edge, correct_rmask,
                   do_cluster, min_cluster_size):
    ref_df, all_logr, all_depths = load_sample_block(filenames, fa_fname, is_haploid_x, sexes, True, correct_gc,
                                                     correct_edge, False)
    if offTarget_fnames:
        anti_ref_df, anti_logr, anti_depths = load_sample_block(offTarget_fnames, fa_fname, is_haploid_x, sexes, False,
                                                                correct_gc, False, correct_rmask)
        ref_df = ref_df.append(anti_ref_df, ignore_index=True, sort=False)
        all_logr = np.hstack([all_logr, anti_logr])
        all_depths = np.hstack([all_depths, anti_depths])

    stats_all = summarize_info(all_logr, all_depths)
    ref_df = ref_df.assign(**stats_all)

    if do_cluster:

        sample_ids = [kernel.fbase(f) for f in filenames]
        if len(sample_ids) != len(all_logr) - 1:
            raise ValueError("Expected %d target coverage files (.cnn), got %d"
                             % (len(all_logr) - 1, len(sample_ids)))
        clustered_cols = create_clusters(all_logr, min_cluster_size, sample_ids)
        if clustered_cols:
            try:
                ref_df = ref_df.assign(**clustered_cols)
            except ValueError as exc:
                print("refBaseline:", len(ref_df.index))
                for cl_key, cl_col in clustered_cols.items():
                    print(cl_key, ":", len(cl_col))
                raise exc
        else:
            print("** Why weren't there any clustered cols?")

    ref_cna = CNA(ref_df, meta_dict={'sample_id': 'refBaseline'})

    ref_cna.sort()
    ref_cna.sort_columns()

    return ref_cna


def load_sample_block(filenames, fa_fname,
                      is_haploid_x, sexes, skip_low,
                      correct_gc, correct_edge, correct_rmask):
    filenames = sorted(filenames, key=kernel.fbase)
    cnarr1 = read_cna(filenames[0])
    if not len(cnarr1):

        col_names = ['chromosome', 'start', 'end', 'gene', 'log2', 'depth']
        if 'gc' in cnarr1 or fa_fname:
            col_names.append('gc')
        if fa_fname:
            col_names.append('rmask')
        col_names.append('spread')
        empty_df = pd.DataFrame.from_records([], columns=col_names)
        empty_logr = np.array([[]] * (len(filenames) + 1))
        empty_dp = np.array([[]] * len(filenames))
        return empty_df, empty_logr, empty_dp

    ref_columns = {
        'chromosome': cnarr1.chromosome,
        'start': cnarr1.start,
        'end': cnarr1.end,
        'gene': cnarr1['gene'],
    }
    if fa_fname and (correct_rmask or correct_gc):
        gc, rmask = get_fasta_stats(cnarr1, fa_fname)
        if correct_gc:
            ref_columns['gc'] = gc
        if correct_rmask:
            ref_columns['rmask'] = rmask
    elif 'gc' in cnarr1 and correct_gc:

        gc = cnarr1['gc']
        ref_columns['gc'] = gc

    is_chr_x = (cnarr1.chromosome == cnarr1._chr_x_label)
    is_chr_y = (cnarr1.chromosome == cnarr1._chr_y_label)
    ref_flat_logr = cnarr1.expect_flat_log2(is_haploid_x)
    ref_edge_bias = correct.get_edge_bias(cnarr1, hyperparameters.INSERT_SIZE)

    all_depths = [cnarr1['depth'] if 'depth' in cnarr1
                  else np.exp2(cnarr1['log2'])]
    all_logr = [
        ref_flat_logr,
        bias_correct_logr(cnarr1, ref_columns, ref_edge_bias,
                          ref_flat_logr, sexes, is_chr_x, is_chr_y,
                          correct_gc, correct_edge, correct_rmask, skip_low)]

    for fname in filenames[1:]:
        cnarrx = read_cna(fname)

        if not np.array_equal(
                cnarr1.data.loc[:, ('chromosome', 'start', 'end', 'gene')].values,
                cnarrx.data.loc[:, ('chromosome', 'start', 'end', 'gene')].values):
            raise RuntimeError("%s bins do not match those in %s"
                               % (fname, filenames[0]))
        all_depths.append(cnarrx['depth'] if 'depth' in cnarrx
                          else np.exp2(cnarrx['log2']))
        all_logr.append(
            bias_correct_logr(cnarrx, ref_columns, ref_edge_bias, ref_flat_logr,
                              sexes, is_chr_x, is_chr_y,
                              correct_gc, correct_edge, correct_rmask, skip_low))
    all_logr = np.vstack(all_logr)
    all_depths = np.vstack(all_depths)
    ref_df = pd.DataFrame.from_dict(ref_columns)
    return ref_df, all_logr, all_depths


def bias_correct_logr(cnarr, ref_columns, ref_edge_bias,
                      ref_flat_logr, sexes, is_chr_x, is_chr_y,
                      correct_gc, correct_edge, correct_rmask, skip_low):
    """Perform bias corrections on the sample."""
    cnarr.center_all(skip_low=skip_low)
    shift_sex_chroms(cnarr, sexes, ref_flat_logr, is_chr_x, is_chr_y)

    if (cnarr['log2'] > hyperparameters.NULL_LOG2_coverInfo - hyperparameters.MIN_REF_COVERAGE
    ).sum() <= len(cnarr) // 2:
        logging.warning("WARNING: most bins have no or very low coverage; "
                        "check that the right BED file was used")
    else:
        if 'gc' in ref_columns and correct_gc:
            cnarr = correct.center_by_window(cnarr, .1, ref_columns['gc'])
        if 'rmask' in ref_columns and correct_rmask:
            cnarr = correct.center_by_window(cnarr, .1, ref_columns['rmask'])
        if correct_edge:
            cnarr = correct.center_by_window(cnarr, .1, ref_edge_bias)
    return cnarr['log2']


def shift_sex_chroms(cnarr, sexes, ref_flat_logr, is_chr_x, is_chr_y):
    is_xx = sexes.get(cnarr.sample_id)
    cnarr['log2'] += ref_flat_logr
    if is_xx:

        cnarr[is_chr_y, 'log2'] = -1.0
    else:

        cnarr[is_chr_x | is_chr_y, 'log2'] += 1.0


def summarize_info(all_logr, all_depths):
    print(all_logr)

    cvg_centers = np.apply_along_axis(measures.biweight_location, 0,
                                      all_logr)
    depth_centers = np.apply_along_axis(measures.biweight_location, 0,
                                        all_depths)
    spreads = np.array([measures.biweight_midvariance(a, initial=i)
                        for a, i in zip(all_logr.T, cvg_centers)])
    print(cvg_centers)
    result = {
        'log2': cvg_centers,
        'depth': depth_centers,
        'spread': spreads,
    }

    return result


def create_clusters(logr_matrix, min_cluster_size, sample_ids):
    from .correlation import markov, kmeans

    logr_matrix = logr_matrix[1:, :]
    print("Clustering", len(logr_matrix), "samples...")

    clusters = kmeans(logr_matrix)
    cluster_cols = {}
    sample_ids = np.array(sample_ids)
    for i, clust_idx in enumerate(clusters):
        i += 1

        if len(clust_idx) < min_cluster_size:
            continue

        samples = sample_ids[clust_idx]

        clust_matrix = logr_matrix[clust_idx, :]

        clust_info = summarize_info(clust_matrix, [])
        cluster_cols.update({
            'log2_%d' % i: clust_info['log2'],
            'spread_%d' % i: clust_info['spread'],
        })
    return cluster_cols


def warn_bad_bins(cnarr, max_name_width=50):
    bad_bins = cnarr[correct.mask_bad_bins(cnarr)]
    fg_index = ~bad_bins['gene'].isin(hyperparameters.OFFTARGET_ALIASES)
    fg_bad_bins = bad_bins[fg_index]
    if len(fg_bad_bins) > 0:
        bad_pct = (100 * len(fg_bad_bins)
                   / sum(~cnarr['gene'].isin(hyperparameters.OFFTARGET_ALIASES)))

        if len(fg_bad_bins) < 500:
            gene_cols = min(max_name_width, max(map(len, fg_bad_bins['gene'])))
            labels = fg_bad_bins.labels()
            chrom_cols = max(labels.apply(len))
            last_gene = None
            for label, probe in zip(labels, fg_bad_bins):
                if probe.gene == last_gene:
                    gene = '  "'
                else:
                    gene = probe.gene
                    last_gene = gene
                if len(gene) > max_name_width:
                    gene = gene[:max_name_width - 3] + '...'
                if 'rmask' in cnarr:
                    logging.info("  %s  %s  log2=%.3f  spread=%.3f  rmask=%.3f",
                                 gene.ljust(gene_cols),
                                 label.ljust(chrom_cols),
                                 probe.log2, probe.spread, probe.rmask)
                else:
                    logging.info("  %s  %s  log2=%.3f  spread=%.3f",
                                 gene.ljust(gene_cols),
                                 label.ljust(chrom_cols),
                                 probe.log2, probe.spread)

    bg_bad_bins = bad_bins[~fg_index]
    if len(bg_bad_bins) > 0:
        bad_pct = (100 * len(bg_bad_bins)
                   / sum(cnarr['gene'].isin(hyperparameters.OFFTARGET_ALIASES)))


def get_fasta_stats(cnarr, fa_fname):
    gc_rm_vals = [calculate_gc_lo(subseq)
                  for subseq in fasta_extract_regions(fa_fname, cnarr)]
    gc_vals, rm_vals = zip(*gc_rm_vals)
    return np.asfarray(gc_vals), np.asfarray(rm_vals)


def calculate_gc_lo(subseq):
    cnt_at_lo = subseq.count('a') + subseq.count('t')
    cnt_at_up = subseq.count('A') + subseq.count('T')
    cnt_gc_lo = subseq.count('g') + subseq.count('c')
    cnt_gc_up = subseq.count('G') + subseq.count('C')
    tot = float(cnt_gc_up + cnt_gc_lo + cnt_at_up + cnt_at_lo)
    if not tot:
        return 0.0, 0.0
    frac_gc = (cnt_gc_lo + cnt_gc_up) / tot
    frac_lo = (cnt_at_lo + cnt_gc_lo) / tot
    return frac_gc, frac_lo


def fasta_extract_regions(fa_fname, intervals):
    with pyfaidx.Fasta(fa_fname, as_raw=True) as fa_file:
        for chrom, subarr in intervals.by_chromosome():
            for _chrom, start, end in subarr.coords():
                yield fa_file[_chrom][int(start):int(end)]


def refBaseline2regions(refarr):
    is_bg = (refarr['gene'].isin(hyperparameters.OFFTARGET_ALIASES))
    regions = GA(refarr.data.loc[:, ('chromosome', 'start', 'end', 'gene')],
                 {'sample_id': 'refBaseline'})
    targets = regions[~is_bg]
    offTargets = regions[is_bg]
    return targets, offTargets
