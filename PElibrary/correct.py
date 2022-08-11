import logging

import numpy as np
import pandas as pd

from . import measures, hyperparameters, adjusting


def do_correct(target_raw, offTarget_raw, reference,
               do_gc=True, do_edge=True, do_rmask=True, do_cluster=False):
    logging.info("Processing target: %s", target_raw.sample_id)
    cnarr, ref_matched = load_adjust_coverages(target_raw, reference,
                                               True, do_gc, do_edge, False)
    logging.info("Processing offTarget: %s", offTarget_raw.sample_id)
    anti_cnarr, ref_anti = load_adjust_coverages(offTarget_raw, reference,
                                                 False, do_gc, False, do_rmask)
    if len(anti_cnarr):
        cnarr.add(anti_cnarr)
        ref_matched.add(ref_anti)

    log2_key = 'log2'
    spread_key = 'spread'
    if do_cluster:
        ref_log2_cols = [col for col in ref_matched.data.columns
                         if col == "log2"
                         or col.startswith("log2")]
        if len(ref_log2_cols) == 1:
            logging.info("Reference does not contain any sub-clusters; "
                         "using %s", log2_key)
        else:

            corr_coefs = np.array([cnarr.log2.corr(ref_matched[ref_col])
                                   for ref_col in ref_log2_cols])
            ordered = [(k, r) for r, k in sorted(zip(corr_coefs, ref_log2_cols),
                                                 reverse=True)]
            logging.info("Correlations with each cluster:\n\t%s",
                         "\n\t".join(["{}\t: {}".format(k, r)
                                      for k, r in ordered]))
            log2_key = ordered[0][0]
            if log2_key.startswith('log2_'):
                sufcorrect = log2_key.split('_', 1)[1]
                spread_key = 'spread_' + sufcorrect
            logging.info(" -> Choosing columns %r and %r", log2_key, spread_key)

    cnarr.data['log2'] -= ref_matched[log2_key]
    cnarr = apply_weights(cnarr, ref_matched, log2_key, spread_key)
    cnarr.center_all(skip_low=True)
    return cnarr


def load_adjust_coverages(cnarr, ref_cnarr, skip_low,
                          correct_gc, correct_edge, correct_rmask):
    if 'gc' in cnarr:
        cnarr = cnarr.keep_columns(cnarr._required_columns + ('depth',))

    if not len(cnarr):
        return cnarr, ref_cnarr[:0]

    ref_matched = match_ref_to_sample(ref_cnarr, cnarr)

    ok_cvg_indices = ~mask_bad_bins(ref_matched)
    cnarr = cnarr[ok_cvg_indices]
    ref_matched = ref_matched[ok_cvg_indices]

    if (cnarr['log2'] > hyperparameters.NULL_LOG2_coverInfo - hyperparameters.MIN_REF_COVERAGE
    ).sum() <= len(cnarr) // 2:
        logging.warning("WARNING: most bins have no or very low coverage; "
                        "check that the right BED file was used")
    else:
        cnarr_index_reset = False
        if correct_gc:
            if 'gc' in ref_matched:
                logging.info("Correcting for GC bias...")
                cnarr = center_by_window(cnarr, .1, ref_matched['gc'])
                cnarr_index_reset = True
            else:
                logging.warning("WARNING: Skipping correction for GC bias")
        if correct_edge:
            logging.info("Correcting for density bias...")
            edge_bias = get_edge_bias(cnarr, hyperparameters.INSERT_SIZE)
            cnarr = center_by_window(cnarr, .1, edge_bias)
            cnarr_index_reset = True
        if correct_rmask:
            if 'rmask' in ref_matched:
                logging.info("Correcting for RepeatMasker bias...")
                cnarr = center_by_window(cnarr, .1, ref_matched['rmask'])
                cnarr_index_reset = True
            else:
                logging.warning("WARNING: Skipping correction for "
                                "RepeatMasker bias")
        if cnarr_index_reset:
            ref_matched.data.reset_index(drop=True, inplace=True)
    return cnarr, ref_matched


def mask_bad_bins(cnarr):
    mask = ((cnarr['log2'] < hyperparameters.MIN_REF_COVERAGE) |
            (cnarr['log2'] > -hyperparameters.MIN_REF_COVERAGE) |
            (cnarr['spread'] > hyperparameters.MAX_REF_SPREAD))
    if 'depth' in cnarr:
        mask |= cnarr['depth'] == 0
    if 'gc' in cnarr:
        mask |= (cnarr['gc'] > .7) | (cnarr['gc'] < .3)
    return mask


def match_ref_to_sample(ref_cnarr, samp_cnarr):
    samp_labeled = samp_cnarr.data.set_index(pd.Index(samp_cnarr.coords()))
    ref_labeled = ref_cnarr.data.set_index(pd.Index(ref_cnarr.coords()))
    for dset, name in ((samp_labeled, "sample"),
                       (ref_labeled, "reference")):
        dupes = dset.index.duplicated()
        if dupes.any():
            raise ValueError("Duplicated genomic coordinates in " + name +
                             " set:\n" + "\n".join(map(str, dset.index[dupes])))

    ref_matched = ref_labeled.reindex(index=samp_labeled.index)

    num_missing = pd.isnull(ref_matched.start).sum()
    if num_missing > 0:
        raise ValueError("Reference is missing %d bins found in %s"
                         % (num_missing, samp_cnarr.sample_id))
    x = ref_cnarr.as_dataframe(ref_matched.reset_index(drop=True)
                               .set_index(samp_cnarr.data.index))
    return x


def center_by_window(cnarr, fraction, sort_key):
    df = cnarr.data.reset_index(drop=True)
    np.random.seed(0xA5EED)
    shuffle_order = np.random.permutation(df.index)

    df = df.iloc[shuffle_order]

    if isinstance(sort_key, pd.Series):
        sort_key = sort_key.values
    sort_key = sort_key[shuffle_order]

    order = np.argsort(sort_key, kind='mergesort')
    df = df.iloc[order]
    biases = adjusting.rolling_median(df['log2'], fraction)

    df['log2'] -= biases
    correctarr = cnarr.as_dataframe(df)
    correctarr.sort()
    return correctarr


def get_edge_bias(cnarr, margin):
    output_by_chrom = []
    for _chrom, subarr in cnarr.by_chromosome():
        tile_starts = subarr['start'].values
        tile_ends = subarr['end'].values
        tgt_sizes = tile_ends - tile_starts

        losses = edge_losses(tgt_sizes, margin)

        gap_sizes = tile_starts[1:] - tile_ends[:-1]
        ok_gaps_mask = (gap_sizes < margin)
        ok_gaps = gap_sizes[ok_gaps_mask]
        left_gains = edge_gains(tgt_sizes[1:][ok_gaps_mask], ok_gaps, margin)
        right_gains = edge_gains(tgt_sizes[:-1][ok_gaps_mask], ok_gaps, margin)
        gains = np.zeros(len(subarr))
        gains[np.concatenate([[False], ok_gaps_mask])] += left_gains
        gains[np.concatenate([ok_gaps_mask, [False]])] += right_gains
        output_by_chrom.append(gains - losses)
    return pd.Series(np.concatenate(output_by_chrom), index=cnarr.data.index)


def edge_losses(target_sizes, insert_size):
    losses = insert_size / (2 * target_sizes)

    small_mask = (target_sizes < insert_size)
    t_small = target_sizes[small_mask]
    losses[small_mask] -= ((insert_size - t_small) ** 2
                           / (2 * insert_size * t_small))
    return losses


def edge_gains(target_sizes, gap_sizes, insert_size):
    if not (gap_sizes <= insert_size).all():
        raise ValueError("Gaps greater than insert size:\n" +
                         gap_sizes[gap_sizes > insert_size].head())
    gap_sizes = np.maximum(0, gap_sizes)
    gains = ((insert_size - gap_sizes) ** 2
             / (4 * insert_size * target_sizes))

    past_other_side_mask = (target_sizes + gap_sizes < insert_size)
    g_past = gap_sizes[past_other_side_mask]
    t_past = target_sizes[past_other_side_mask]
    gains[past_other_side_mask] -= ((insert_size - t_past - g_past) ** 2
                                    / (4 * insert_size * t_past))
    return gains


def apply_weights(cnarr, ref_matched, log2_key, spread_key, epsilon=1e-4):
    logging.debug("Weighting bins by size and overall variance in sample")
    simple_wt = np.zeros(len(cnarr))

    is_anti = cnarr['gene'].isin(hyperparameters.OFFTARGET_ALIASES)
    tgt_cna = cnarr[~is_anti]
    tgt_var = measures.biweight_midvariance(tgt_cna
                                                .drop_low_coverage()
                                                .residuals()) ** 2
    bin_sz = np.sqrt(tgt_cna['end'] - tgt_cna['start'])
    tgt_simple_wts = 1 - tgt_var / (bin_sz / bin_sz.mean())
    simple_wt[~is_anti] = tgt_simple_wts

    if is_anti.any():

        anti_cna = cnarr[is_anti]
        anti_ok = anti_cna.drop_low_coverage()
        frac_anti_low = 1 - (len(anti_ok) / len(anti_cna))
        if frac_anti_low > .5:
            logging.warning("WARNING: Most offTarget bins ({:.2f}%, {:d}/{:d})"
                            " have low or no coverage; is this amplicon/WGS?"
                            .format(100 * frac_anti_low,
                                    len(anti_cna) - len(anti_ok),
                                    len(anti_cna)))

        anti_var = measures.biweight_midvariance(anti_ok.residuals()) ** 2
        anti_bin_sz = np.sqrt(anti_cna['end'] - anti_cna['start'])
        anti_simple_wts = 1 - anti_var / (anti_bin_sz / anti_bin_sz.mean())
        simple_wt[is_anti] = anti_simple_wts

        var_ratio = max(tgt_var, .01) / max(anti_var, .01)
        if var_ratio > 1:
            logging.info("Targets are %.2f x more variable than offTargets",
                         var_ratio)
        else:
            logging.info("offTargets are %.2f x more variable than targets",
                         1. / var_ratio)

    if ((ref_matched[spread_key] > epsilon).any() and
            (np.abs(np.mod(ref_matched[log2_key], 1)) > epsilon).any()):

        logging.debug("Weighting bins by coverage spread in reference")

        fancy_wt = 1.0 - ref_matched[spread_key] ** 2

        x = .9
        weights = (x * fancy_wt
                   + (1 - x) * simple_wt)
    else:

        weights = simple_wt

    return cnarr.add_columns(weight=weights.clip(epsilon, 1.0))
