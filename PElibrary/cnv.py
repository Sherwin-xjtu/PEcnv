import logging

import numpy as np
import pandas as pd
from scipy.stats import median_test
from pegeno import GenomicArray

from . import kernel, measures, hyperparameters, adjusting
from .segmIndicators import segment_mean


class CopyNumArray(GenomicArray):
    _required_columns = ("chromosome", "start", "end", "gene", "log2")
    _required_dtypes = (str, int, int, str, float)

    def __init__(self, data_table, meta_dict=None):
        GenomicArray.__init__(self, data_table, meta_dict)

    @property
    def log2(self):
        return self.data["log2"]

    @log2.setter
    def log2(self, value):
        self.data["log2"] = value

    @property
    def _chr_x_label(self):
        if 'chr_x' in self.meta:
            return self.meta['chr_x']
        if len(self):
            chr_x = ('chrX' if self.chromosome.iat[0].startswith('chr') else 'X')
            self.meta['chr_x'] = chr_x
            return chr_x
        return ''

    @property
    def _chr_y_label(self):
        if 'chr_y' in self.meta:
            return self.meta['chr_y']
        if len(self):
            chr_y = ('chrY' if self._chr_x_label.startswith('chr') else 'Y')
            self.meta['chr_y'] = chr_y
            return chr_y
        return ''

    def by_gene(self, ignore=hyperparameters.IGNORE_GENE_NAMES):

        ignore += hyperparameters.OFFTARGET_ALIASES
        start_idx = end_idx = None
        for _chrom, subgary in self.by_chromosome():
            prev_idx = 0
            for gene, gene_idx in subgary._get_gene_map().items():
                if gene not in ignore:
                    if not len(gene_idx):
                        logging.warning("Specified gene name somehow missing: "
                                        "%s", gene)
                        continue
                    start_idx = gene_idx[0]
                    end_idx = gene_idx[-1] + 1
                    if prev_idx < start_idx:
                        yield hyperparameters.OFFTARGET_NAME, subgary.as_dataframe(
                            subgary.data.iloc[prev_idx:start_idx])
                    yield gene, subgary.as_dataframe(
                        subgary.data.iloc[start_idx:end_idx])
                    prev_idx = end_idx
            if prev_idx < len(subgary) - 1:
                yield hyperparameters.OFFTARGET_NAME, subgary.as_dataframe(
                    subgary.data.iloc[prev_idx:])

    def center_all(self, estimator=pd.Series.mean, by_chrom=True,
                   skip_low=False, verbose=False):

        est_funcs = {
            "mean": pd.Series.mean,
            "median": pd.Series.median,
            "mode": measures.modal_location,
            "biweight": measures.biweight_location,
        }
        if isinstance(estimator, str):
            if estimator in est_funcs:
                estimator = est_funcs[estimator]
            else:
                raise ValueError("Estimator must be a function or one of: %s"
                                 % ", ".join(map(repr, est_funcs)))
        cnarr = (self.drop_low_coverage(verbose=verbose) if skip_low else self
                 ).autosomes()
        if cnarr:
            if by_chrom:
                values = pd.Series([estimator(subarr['log2'])
                                    for _c, subarr in cnarr.by_chromosome()
                                    if len(subarr)])
            else:
                values = cnarr['log2']
            shift = -estimator(values)
            if verbose:
                logging.info("Shifting log2 values by %f", shift)
            self.data['log2'] += shift

    def drop_low_coverage(self, verbose=False):

        min_cvg = hyperparameters.NULL_LOG2_coverInfo - hyperparameters.MIN_REF_COVERAGE
        drop_idx = self.data['log2'] < min_cvg
        if 'depth' in self:
            drop_idx |= self.data['depth'] == 0
        if verbose and drop_idx.any():
            logging.info("Dropped %d low-coverage bins",
                         drop_idx.sum())
        return self[~drop_idx]

    def squash_genes(self, summary_func=measures.biweight_location,
                     squash_offTarget=False, ignore=hyperparameters.IGNORE_GENE_NAMES):

        def squash_rows(name, rows):
            if len(rows) == 1:
                return tuple(rows.iloc[0])
            chrom = kernel.check_unique(rows.chromosome, 'chromosome')
            start = rows.start.iat[0]
            end = rows.end.iat[-1]
            cvg = summary_func(rows.log2)
            outrow = [chrom, start, end, name, cvg]

            for xfield in ('depth', 'gc', 'rmask', 'spread', 'weight'):
                if xfield in self:
                    outrow.append(summary_func(rows[xfield]))
            if 'probes' in self:
                outrow.append(sum(rows['probes']))
            return tuple(outrow)

        outrows = []
        for name, subarr in self.by_gene(ignore):
            if not len(subarr):
                continue
            if name in hyperparameters.OFFTARGET_ALIASES and not squash_offTarget:
                outrows.extend(subarr.data.itertuples(index=False))
            else:
                outrows.append(squash_rows(name, subarr.data))
        return self.as_rows(outrows)

    def shift_xx(self, male_reference=False, is_xx=None):

        outprobes = self.copy()
        if is_xx is None:
            is_xx = self.guess_xx(male_reference=male_reference)
        if is_xx and male_reference:

            outprobes[outprobes.chromosome == self._chr_x_label, 'log2'] -= 1.0

        elif not is_xx and not male_reference:

            outprobes[outprobes.chromosome == self._chr_x_label, 'log2'] += 1.0

        return outprobes

    def guess_xx(self, male_reference=False, verbose=True):

        is_xy, stats = self.compare_sex_chromosomes(male_reference)
        if is_xy is None:
            return None
        elif verbose:
            logging.info("Relative log2 coverage of %s=%.3g, %s=%.3g "
                         "(maleness=%.3g x %.3g = %.3g) --> assuming %s",
                         self._chr_x_label, stats['chrx_ratio'],
                         self._chr_y_label, stats['chry_ratio'],
                         stats['chrx_male_lr'], stats['chry_male_lr'],
                         stats['chrx_male_lr'] * stats['chry_male_lr'],
                         'male' if is_xy else 'female')
        return ~is_xy

    def compare_sex_chromosomes(self, male_reference=False, skip_low=False):

        if not len(self):
            return None, {}

        chrx = self[self.chromosome == self._chr_x_label]
        if not len(chrx):
            logging.warning("No %s found in sample; is the input truncated?",
                            self._chr_x_label)
            return None, {}

        auto = self.autosomes()
        if skip_low:
            chrx = chrx.drop_low_coverage()
            auto = auto.drop_low_coverage()
        auto_l = auto['log2'].values
        use_weight = ('weight' in self)
        auto_w = auto['weight'].values if use_weight else None

        def compare_to_auto(vals, weights):

            try:
                stat, _p, _med, cont = median_test(auto_l, vals, ties='ignore',
                                                   lambda_='log-likelihood')
            except ValueError:

                stat = None
            else:
                if stat == 0 and 0 in cont:
                    stat = None

            if use_weight:
                med_diff = abs(measures.weighted_median(auto_l, auto_w) -
                               measures.weighted_median(vals, weights))
            else:
                med_diff = abs(np.median(auto_l) - np.median(vals))
            return (stat, med_diff)

        def compare_chrom(vals, weights, female_shift, male_shift):

            female_stat, f_diff = compare_to_auto(vals + female_shift, weights)
            male_stat, m_diff = compare_to_auto(vals + male_shift, weights)

            if female_stat is not None and male_stat is not None:
                return female_stat / max(male_stat, 0.01)

            return f_diff / max(m_diff, 0.01)

        female_x_shift, male_x_shift = (-1, 0) if male_reference else (0, +1)
        chrx_male_lr = compare_chrom(chrx['log2'].values,
                                     (chrx['weight'].values if use_weight
                                      else None),
                                     female_x_shift, male_x_shift)
        combined_score = chrx_male_lr

        chry = self[self.chromosome == self._chr_y_label]
        if len(chry):
            chry_male_lr = compare_chrom(chry['log2'].values,
                                         (chry['weight'].values if use_weight
                                          else None),
                                         +3, 0)
            if np.isfinite(chry_male_lr):
                combined_score *= chry_male_lr
        else:

            chry_male_lr = np.nan

        auto_mean = segment_mean(auto, skip_low=skip_low)
        chrx_mean = segment_mean(chrx, skip_low=skip_low)
        chry_mean = segment_mean(chry, skip_low=skip_low)
        return (combined_score > 1.0,
                dict(chrx_ratio=chrx_mean - auto_mean,
                     chry_ratio=chry_mean - auto_mean,
                     combined_score=combined_score,

                     chrx_male_lr=chrx_male_lr,
                     chry_male_lr=chry_male_lr,
                     ))

    def expect_flat_log2(self, is_male_reference=None):
        if is_male_reference is None:
            is_male_reference = not self.guess_xx(verbose=False)
        cvg = np.zeros(len(self), dtype=np.float_)
        if is_male_reference:

            idx = ((self.chromosome == self._chr_x_label).values |
                   (self.chromosome == self._chr_y_label).values)
        else:

            idx = (self.chromosome == self._chr_y_label).values
        cvg[idx] = -1.0
        return cvg

    def residuals(self, segments=None):

        if not segments:
            resids = [subcna.log2 - subcna.log2.median()
                      for _chrom, subcna in self.by_chromosome()]
        elif "log2" in segments:
            resids = [bins_lr - seg_lr
                      for seg_lr, bins_lr in zip(
                    segments['log2'],
                    self.iter_ranges_of(segments, 'log2', mode='inner',
                                        keep_empty=True))
                      if len(bins_lr)]
        else:
            resids = [lr - lr.median()
                      for lr in self.iter_ranges_of(segments, 'log2',
                                                    keep_empty=False)]
        return pd.concat(resids) if resids else pd.Series([])

    def smooth_log2(self, bandwidth=None, by_arm=True):

        if bandwidth is None:
            bandwidth = adjusting.guess_window_size(self.log2,
                                                    weights=(self['weight']
                                                             if 'weight' in self
                                                             else None))

        if by_arm:
            parts = self.by_arm()
        else:
            parts = self.by_chromosome()
        if 'weight' in self:
            out = [adjusting.savgol(subcna['log2'].values, bandwidth,
                                    weights=subcna['weight'].values)
                   for _chrom, subcna in parts]
        else:
            out = [adjusting.savgol(subcna['log2'].values, bandwidth)
                   for _chrom, subcna in parts]
        return np.concatenate(out)

    def _guess_average_depth(self, segments=None, window=100):

        cnarr = self.autosomes()
        if not len(cnarr):
            cnarr = self

        y_log2 = cnarr.residuals(segments)
        if segments is None and window:
            y_log2 -= adjusting.savgol(y_log2, window)

        y = np.exp2(y_log2)

        loc = measures.biweight_location(y)
        spread = measures.biweight_midvariance(y, loc)
        if spread > 0:
            return loc / spread ** 2
        return loc
