import logging
import time
from collections import OrderedDict as OD

import numpy as np
import pandas as pd
from pegeno import tabio

from . import finalcall
from .implementation import read_cna
from ._version import __version__


def merge_samples(filenames):
    def label_with_gene(cnarr):
        row2label = lambda row: "{}:{}-{}:{}".format(
            row.chromosome, row.start, row.end, row.gene)
        return cnarr.data.apply(row2label, axis=1)

    if not filenames:
        return []
    first_cnarr = read_cna(filenames[0])
    out_table = first_cnarr.data.reindex(columns=["chromosome", "start", "end", "gene"])
    out_table["label"] = label_with_gene(first_cnarr)
    out_table[first_cnarr.sample_id] = first_cnarr["log2"]
    for fname in filenames[1:]:
        cnarr = read_cna(fname)
        if not (len(cnarr) == len(out_table)
                and (label_with_gene(cnarr) == out_table["label"]).all()):
            raise ValueError("Mismatched row coordinates in %s" % fname)

        if cnarr.sample_id in out_table.columns:
            raise ValueError("Duplicate sample ID: %s" % cnarr.sample_id)
        out_table[cnarr.sample_id] = cnarr["log2"]
        del cnarr
    return out_table


def fmt_cdt(sample_ids, table):
    outheader = ['GID', 'CLID', 'NAME', 'GWEIGHT'] + sample_ids
    header2 = ['AID', '', '', '']
    header2.extend(['ARRY' + str(i).zfill(3) + 'X'
                    for i in range(len(sample_ids))])
    header3 = ['EWEIGHT', '', '', ''] + ['1'] * len(sample_ids)
    outrows = [header2, header3]
    outtable = pd.concat([
        pd.DataFrame.from_dict(OD([
            ("GID", pd.Series(table.index).apply(lambda x: "GENE%dX" % x)),
            ("CLID", pd.Series(table.index).apply(lambda x: "IMAGE:%d" % x)),
            ("NAME", table["label"]),
            ("GWEIGHT", 1),
        ])),
        table.drop(["chromosome", "start", "end", "gene", "label"],
                   axis=1)],
        axis=1)
    outrows.extend(outtable.itertuples(index=False))
    return outheader, outrows


def fmt_gct(sample_ids, table):
    return NotImplemented


def fmt_jtv(sample_ids, table):
    outheader = ["CloneID", "Name"] + sample_ids
    outtable = pd.concat([
        pd.DataFrame({
            "CloneID": "IMAGE:",
            "Name": table["label"],
        }),
        table.drop(["chromosome", "start", "end", "gene", "label"],
                   axis=1)],
        axis=1)
    outrows = outtable.itertuples(index=False)
    return outheader, outrows


def export_nexus_basic(cnarr):
    out_table = cnarr.data.reindex(columns=['chromosome', 'start', 'end', 'gene', 'log2'])
    out_table['probe'] = cnarr.labels()
    return out_table


def export_nexus_ogt(cnarr, varr, min_weight=0.0):
    if min_weight and "weight" in cnarr:
        mask_low_weight = (cnarr["weight"] < min_weight)
        cnarr.data = cnarr.data[~mask_low_weight]
    bafs = varr.baf_by_ranges(cnarr)

    out_table = cnarr.data.reindex(columns=['chromosome', 'start', 'end', 'log2'])
    out_table = out_table.rename(columns={
        "chromosome": "Chromosome",
        "start": "Position",
        "end": "Position",
        "log2": "Log R Ratio",
    })
    out_table["B-Allele Frequency"] = bafs
    return out_table


def export_seg(sample_fnames, chrom_ids=False):
    dframes, sample_ids = zip(*(_load_seg_dframe_id(fname)
                                for fname in sample_fnames))
    out_table = tabio.seg.write_seg(dframes, sample_ids, chrom_ids)
    return out_table


def _load_seg_dframe_id(fname):
    segarr = read_cna(fname)
    assert segarr is not None
    assert segarr.data is not None
    assert segarr.sample_id is not None
    return segarr.data, segarr.sample_id


def export_bed(segments, ploidy, is_reference_male, is_sample_female,
               label, show):
    out = segments.data.reindex(columns=["chromosome", "start", "end"])
    out["label"] = label if label else segments["gene"]
    out["ncopies"] = (segments["cn"] if "cn" in segments
                      else finalcall.absolute_pure(segments, ploidy, is_reference_male)
                      .round().astype('int'))
    if show == "ploidy":

        out = out[out["ncopies"] != ploidy]
    elif show == "variant":

        exp_copies = finalcall.absolute_expect(segments, ploidy, is_sample_female)
        out = out[out["ncopies"] != exp_copies]
    return out


VCF_HEADER = """\
##fileformat=VCFv4.2
##fileDate={date}
##source=CNVkit v{version}
##INFO=<ID=CIEND,Number=2,Type=Integer,Description="Confidence interval around END for imprecise variants">
##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval around POS for imprecise variants">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant described in this record">
##INFO=<ID=IMPRECISE,Number=0,Type=Flag,Description="Imprecise structural variation">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Difference in length between REF and ALT alleles">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=FOLD_CHANGE,Number=1,Type=Float,Description="Fold change">
##INFO=<ID=FOLD_CHANGE_LOG,Number=1,Type=Float,Description="Log fold change">
##INFO=<ID=PROBES,Number=1,Type=Integer,Description="Number of probes in CNV">
##ALT=<ID=DEL,Description="Deletion">
##ALT=<ID=DUP,Description="Duplication">
##ALT=<ID=CNV,Description="Copy number variable region">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Float,Description="Genotype quality">
##FORMAT=<ID=CN,Number=1,Type=Integer,Description="Copy number genotype for imprecise events">
##FORMAT=<ID=CNQ,Number=1,Type=Float,Description="Copy number genotype quality for imprecise events">
""".format(date=time.strftime("%Y%m%d"), version=__version__)


def export_vcf(segments, ploidy, is_reference_male, is_sample_female,
               sample_id=None, cnarr=None):
    vcf_columns = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
                   "INFO", "FORMAT", sample_id or segments.sample_id]
    if cnarr:
        segments = assign_ci_start_end(segments, cnarr)
    vcf_rows = segments2vcf(segments, ploidy, is_reference_male,
                            is_sample_female)
    table = pd.DataFrame.from_records(vcf_rows, columns=vcf_columns)
    vcf_body = table.to_csv(sep='\t', header=True, index=False,
                            float_format="%.3g")
    return VCF_HEADER, vcf_body


def assign_ci_start_end(segarr, cnarr):
    lefts_rights = ((bins.end.iat[0], bins.start.iat[-1])
                    for _seg, bins in cnarr.by_ranges(segarr, mode="outer"))
    ci_lefts, ci_rights = zip(*lefts_rights)
    return segarr.as_dataframe(
        segarr.data.assign(ci_left=ci_lefts, ci_right=ci_rights))


def segments2vcf(segments, ploidy, is_reference_male, is_sample_female):
    out_dframe = segments.data.reindex(columns=["chromosome", "end", "log2", "probes"])
    out_dframe["start"] = segments.start.replace(0, 1)

    if "cn" in segments:
        out_dframe["ncopies"] = segments["cn"]
        abs_expect = finalcall.absolute_expect(segments, ploidy, is_sample_female)
    else:
        abs_dframe = finalcall.absolute_dataframe(segments, ploidy, 1.0,
                                                  is_reference_male,
                                                  is_sample_female)
        out_dframe["ncopies"] = abs_dframe["absolute"].round().astype('int')
        abs_expect = abs_dframe["expect"]
    idx_losses = (out_dframe["ncopies"] < abs_expect)

    svlen = segments.end - segments.start
    svlen[idx_losses] *= -1
    out_dframe["svlen"] = svlen

    out_dframe["svtype"] = "DUP"
    out_dframe.loc[idx_losses, "svtype"] = "DEL"

    out_dframe["format"] = "GT:GQ:CN:CNQ"
    out_dframe.loc[idx_losses, "format"] = "GT:GQ"

    if "ci_left" in segments and "ci_right" in segments:
        has_ci = True

        left_margin = segments["ci_left"].values - segments.start.values
        right_margin = segments.end.values - segments["ci_right"].values
        out_dframe["ci_pos_left"] = np.r_[0, -right_margin[:-1]]
        out_dframe["ci_pos_right"] = left_margin
        out_dframe["ci_end_left"] = right_margin
        out_dframe["ci_end_right"] = np.r_[left_margin[1:], 0]
    else:
        has_ci = False

    for out_row, abs_exp in zip(out_dframe.itertuples(index=False), abs_expect):
        if (out_row.ncopies == abs_exp or

                not str(out_row.probes).isdigit()):
            continue

        if out_row.ncopies > abs_exp:
            genotype = "0/1:0:%d:%d" % (out_row.ncopies, out_row.probes)
        elif out_row.ncopies < abs_exp:

            if out_row.ncopies == 0:

                gt = "1/1"
            else:

                gt = "0/1"
            genotype = "%s:%d" % (gt, out_row.probes)

        fields = ["IMPRECISE",
                  "SVTYPE=%s" % out_row.svtype,
                  "END=%d" % out_row.end,
                  "SVLEN=%d" % out_row.svlen,
                  "FOLD_CHANGE=%f" % 2.0 ** out_row.log2,
                  "FOLD_CHANGE_LOG=%f" % out_row.log2,
                  "PROBES=%d" % out_row.probes
                  ]
        if has_ci:
            fields.extend([
                "CIPOS=(%d,%d)" % (out_row.ci_pos_left, out_row.ci_pos_right),
                "CIEND=(%d,%d)" % (out_row.ci_end_left, out_row.ci_end_right),
            ])
        info = ";".join(fields)

        yield (out_row.chromosome, out_row.start, '.', 'N',
               "<%s>" % out_row.svtype, '.', '.',
               info, out_row.format, genotype)


def export_gistic_markers(cnr_fnames):
    colnames = ["ID", "CHROM", "POS"]
    out_chunks = []

    for fname in cnr_fnames:
        cna = read_cna(fname).autosomes()
        marker_ids = cna.labels()
        tbl = pd.concat([
            pd.DataFrame({
                "ID": marker_ids,
                "CHROM": cna.chromosome,
                "POS": cna.start + 1,
            }, columns=colnames),
            pd.DataFrame({
                "ID": marker_ids,
                "CHROM": cna.chromosome,
                "POS": cna.end,
            }, columns=colnames),
        ], ignore_index=True)
        out_chunks.append(tbl)
    return pd.concat(out_chunks).drop_duplicates()


def export_theta(tumor_segs, normal_cn):
    out_columns = ["#ID", "chrm", "start", "end", "tumorCount", "normalCount"]
    if not tumor_segs:
        return pd.DataFrame(columns=out_columns)

    xy_names = []
    tumor_segs = tumor_segs.autosomes(also=xy_names)
    if normal_cn:
        normal_cn = normal_cn.autosomes(also=xy_names)

    table = tumor_segs.data.reindex(columns=["start", "end"])

    chr2idx = {c: i + 1
               for i, c in enumerate(tumor_segs.chromosome.drop_duplicates())}
    table["chrm"] = tumor_segs.chromosome.replace(chr2idx)

    table["#ID"] = ["start_%d_%d:end_%d_%d"
                    % (row.chrm, row.start, row.chrm, row.end)
                    for row in table.itertuples(index=False)]

    ref_means, nbins = ref_means_nbins(tumor_segs, normal_cn)
    table["tumorCount"] = theta_read_counts(tumor_segs.log2, nbins)
    table["normalCount"] = theta_read_counts(ref_means, nbins)
    return table[out_columns]


def ref_means_nbins(tumor_segs, normal_cn):
    if normal_cn:
        log2s_in_segs = [bins['log2']
                         for _seg, bins in normal_cn.by_ranges(tumor_segs)]

        ref_means = np.array([s.mean() for s in log2s_in_segs])
        if "probes" in tumor_segs:
            nbins = tumor_segs["probes"]
        else:
            nbins = np.array([len(s) for s in log2s_in_segs])
    else:

        ref_means = np.zeros(len(tumor_segs))
        if "weight" in tumor_segs and (tumor_segs["weight"] > 1.0).any():

            nbins = tumor_segs["weight"]

            nbins /= nbins.max() / nbins.mean()
        else:
            if "probes" in tumor_segs:
                nbins = tumor_segs["probes"]
            else:
                logging.warning("No probe counts in tumor segments file and no "
                                "normal reference given; guessing normal "
                                "read-counts-per-segment from segment sizes")
                sizes = tumor_segs.end - tumor_segs.start
                nbins = sizes / sizes.mean()
            if "weight" in tumor_segs:
                nbins *= tumor_segs["weight"] / tumor_segs["weight"].mean()
    return ref_means, nbins


def theta_read_counts(log2_ratio, nbins,

                      avg_depth=500, avg_bin_width=200, read_len=100):
    read_depth = (2 ** log2_ratio) * avg_depth
    read_count = nbins * avg_bin_width * read_depth / read_len
    return read_count.round().fillna(0).astype('int')


def export_theta_snps(varr):
    varr = varr.autosomes(also=(['chrX', 'chrY']
                                if varr.chromosome.iat[0].startswith('chr')
                                else ['X', 'Y']))

    varr = varr[(varr["ref"].str.len() == 1) & (varr["alt"].str.len() == 1)]

    varr.data.dropna(subset=["depth", "alt_count"], inplace=True)
    if "n_depth" in varr and "n_alt_count" in varr:
        varr.data.dropna(subset=["n_depth", "alt_count"], inplace=True)

    varr = varr[varr["depth"] >= varr["alt_count"]]

    for depth_key, alt_key in (("depth", "alt_count"),
                               ("n_depth", "n_alt_count")):
        table = varr.data.reindex(columns=('chromosome', 'start', depth_key, alt_key))
        table = (table.assign(ref_depth=table[depth_key] - table[alt_key])
                 .reindex(columns=('chromosome', 'start', 'ref_depth', alt_key))
                 .dropna())
        table['ref_depth'] = table['ref_depth'].astype('int')
        table[alt_key] = table[alt_key].astype('int')
        table.columns = ["#Chrm", "Pos", "Ref_Allele", "Mut_Allele"]
        yield table


EXPORT_FORMATS = {
    'cdt': fmt_cdt,

    'gistic': export_gistic_markers,
    'jtv': fmt_jtv,
    'nexus-basic': export_nexus_basic,
    'nexus-ogt': export_nexus_ogt,
    'seg': export_seg,
    'theta': export_theta,
    'vcf': export_vcf,
}
