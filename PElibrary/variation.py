import logging

import numpy as np
import pandas as pd
from pegeno import GenomicArray


class VariantArray(GenomicArray):
    _required_columns = ("chromosome", "start", "end", "ref", "alt")
    _required_dtypes = (str, int, int, str, str)

    def __init__(self, data_table, meta_dict=None):
        GenomicArray.__init__(self, data_table, meta_dict)

    def baf_by_ranges(self, ranges, summary_func=np.nanmedian, above_half=None,
                      tumor_boost=False):
        if 'alt_freq' not in self:
            return pd.Series(np.repeat(np.nan, len(ranges)), index=self.data.index)

        def summarize(vals):
            return summary_func(_mirrored_baf(vals, above_half))

        cnarr = self.heterozygous()
        if tumor_boost and 'n_alt_freq' in self:
            cnarr = cnarr.add_columns(alt_freq=cnarr.tumor_boost())
        return cnarr.into_ranges(ranges, 'alt_freq', np.nan, summarize)

    def het_frac_by_ranges(self, ranges):

        if 'zygosity' not in self and 'n_zygosity' not in self:
            return self.as_series(np.repeat(np.nan, len(ranges)))

        zygosity = self['n_zygosity' if 'n_zygosity' in self
        else 'zygosity']
        het_idx = (zygosity != 0.0) & (zygosity != 1.0)
        cnarr = self.add_columns(is_het=het_idx)
        het_frac = cnarr.into_ranges(ranges, 'is_het', np.nan, np.nanmean)
        return het_frac

    def zygosity_from_freq(self, het_freq=0.0, hom_freq=1.0):
        assert 0.0 <= het_freq <= hom_freq <= 1.0
        self = self.copy()
        for freq_key, zyg_key in (('alt_freq', 'zygosity'),
                                  ('n_alt_freq', 'n_zygosity')):
            if zyg_key in self:
                zyg = np.repeat(0.5, len(self))
                vals = self[freq_key].values
                zyg[vals >= hom_freq] = 1.0
                zyg[vals < het_freq] = 0.0
                self[zyg_key] = zyg
        return self

    def heterozygous(self):
        if 'zygosity' in self:

            zygosity = self['n_zygosity' if 'n_zygosity' in self
            else 'zygosity']
            het_idx = (zygosity != 0.0) & (zygosity != 1.0)
            if het_idx.any():
                self = self[het_idx]
        return self

    def mirrored_baf(self, above_half=None, tumor_boost=False):

        if tumor_boost and "n_alt_freq" in self:
            alt_freq = self.tumor_boost()
        else:
            alt_freq = self["alt_freq"]
        return _mirrored_baf(alt_freq, above_half)

    def tumor_boost(self):

        if not ("alt_freq" in self and "n_alt_freq" in self):
            raise ValueError("TumorBoost requires a matched tumor and normal "
                             "pair of samples in the VCF.")
        return _tumor_boost(self["alt_freq"].values, self["n_alt_freq"].values)


def _mirrored_baf(vals, above_half=None):
    shift = (vals - .5).abs()
    if above_half is None:
        above_half = (vals.median() > .5)
    if above_half:
        return .5 + shift
    else:
        return .5 - shift


def _tumor_boost(t_freqs, n_freqs):
    lt_mask = (t_freqs < n_freqs)
    lt_idx = np.nonzero(lt_mask)[0]
    gt_idx = np.nonzero(~lt_mask)[0]
    out = pd.Series(np.zeros_like(t_freqs))
    out[lt_idx] = 0.5 * t_freqs.take(lt_idx) / n_freqs.take(lt_idx)
    out[gt_idx] = 1 - 0.5 * (1 - t_freqs.take(gt_idx)
                             ) / (1 - n_freqs.take(gt_idx))
    return out


def _allele_specific_copy_numbers(segarr, varr, ploidy=2):
    seg_depths = ploidy * np.exp2(segarr["log2"])
    seg_bafs = varr.baf_by_ranges(segarr, above_half=True)
    cn1 = 0.5 * (1 - seg_bafs) * seg_depths
    cn2 = seg_depths - cn1

    return pd.DataFrame({"baf": seg_bafs, "cn1": cn1, "cn2": cn2})
