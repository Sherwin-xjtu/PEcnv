import logging
import os

from matplotlib import pyplot
from pegeno import tabio, GenomicArray as GA

from . import (preprocess, offTarget, dynamicBin, prebin, finalcall, kernel, coverInfo,
               drawing, correct, multiprocess, refBaseline, segmenting,
               segmIndicators, target)
from .implementation import read_cna


def run_make_refBaseline(normal_bams, target_bed, offTarget_bed,
                         male_refBaseline, fasta, annotate, short_names,
                         target_avg_size, preprocess_bed, offTarget_avg_size,
                         offTarget_min_size, output_refBaseline, output_dir,
                         processes, by_count, method, do_cluster):
    if method in ("wgs", "amplicon"):
        if offTarget_bed:
            raise ValueError("%r protocol: offTarget should not be "
                             "given/specified." % method)
        if preprocess_bed and target_bed and preprocess_bed != target_bed:
            raise ValueError("%r protocol: targets and preprocess should not be "
                             "different." % method)

    bait_arr = None
    if method == "wgs":
        if not annotate:
            logging.warning("WGS protocol: recommend '--annotate' option "
                            "(e.g. refFlat.txt) to help locate genes "
                            "in output files.")
        preprocess_arr = None
        if not target_bed:

            if preprocess_bed:
                target_bed = preprocess_bed
            elif fasta:

                preprocess_arr = preprocess.do_preprocess(fasta)

                target_bed = os.path.splitext(os.path.basename(fasta)
                                              )[0] + ".bed"
                tabio.write(preprocess_arr, target_bed, "bed3")
            else:
                raise ValueError("WGS protocol: need to provide --targets, "
                                 "--preprocess, or --fasta options.")

        if not target_avg_size:
            if normal_bams:

                if fasta and not preprocess_arr:
                    preprocess_arr = preprocess.do_preprocess(fasta)
                if preprocess_arr:
                    autobin_args = ['wgs', None, preprocess_arr]
                else:

                    bait_arr = tabio.read_auto(target_bed)
                    autobin_args = ['amplicon', bait_arr]

                bam_fname = dynamicBin.midsize_file(normal_bams)
                (wgs_depth, target_avg_size), _ = dynamicBin.do_autobin(
                    bam_fname, *autobin_args, bp_per_bin=50000., fasta=fasta)

            else:

                target_avg_size = 5000

    tgt_name_base, _tgt_ext = os.path.splitext(os.path.basename(target_bed))
    if output_dir:
        tgt_name_base = os.path.join(output_dir, tgt_name_base)

    new_target_fname = tgt_name_base + '.target.bed'
    if bait_arr is None:
        bait_arr = tabio.read_auto(target_bed)
    target_arr = target.do_target(bait_arr, annotate, short_names, True,
                                  **({'avg_size': target_avg_size}
                                     if target_avg_size
                                     else {}))
    tabio.write(target_arr, new_target_fname, 'bed4')
    target_bed = new_target_fname

    if not offTarget_bed:

        offTarget_bed = tgt_name_base + '.offTarget.bed'
        if method == "hybrid":

            anti_kwargs = {}
            if preprocess_bed:
                anti_kwargs['preprocess'] = tabio.read_auto(preprocess_bed)
            if offTarget_avg_size:
                anti_kwargs['avg_bin_size'] = offTarget_avg_size
            if offTarget_min_size:
                anti_kwargs['min_bin_size'] = offTarget_min_size
            anti_arr = offTarget.do_offTarget(target_arr, **anti_kwargs)
        else:

            anti_arr = GA([])
        tabio.write(anti_arr, offTarget_bed, "bed4")

    if len(normal_bams) == 0:

        ref_arr = refBaseline.do_refBaseline_flat(target_bed, offTarget_bed, fasta,
                                                  male_refBaseline)
    else:

        with multiprocess.pick_pool(processes) as pool:
            tgt_futures = []
            anti_futures = []
            procs_per_cnn = max(1, processes // (2 * len(normal_bams)))
            for nbam in normal_bams:
                sample_id = kernel.fbase(nbam)
                sample_pfx = os.path.join(output_dir, sample_id)
                tgt_futures.append(
                    pool.submit(run_write_coverInfo,
                                target_bed, nbam,
                                sample_pfx + '.targetcoverInfo.tsv',
                                by_count, procs_per_cnn, fasta))
                anti_futures.append(
                    pool.submit(run_write_coverInfo,
                                offTarget_bed, nbam,
                                sample_pfx + '.offTargetcoverInfo.tsv',
                                by_count, procs_per_cnn, fasta))

        target_fnames = [tf.result() for tf in tgt_futures]
        offTarget_fnames = [af.result() for af in anti_futures]

        ref_arr = refBaseline.do_refBaseline(target_fnames, offTarget_fnames,
                                             fasta, male_refBaseline, None,
                                             do_gc=True,
                                             do_edge=(method == "hybrid"),
                                             do_rmask=True,
                                             do_cluster=do_cluster)
    if not output_refBaseline:
        output_refBaseline = os.path.join(output_dir, "refBaseline.tsv")
    kernel.ensure_path(output_refBaseline)
    tabio.write(ref_arr, output_refBaseline)
    return output_refBaseline, target_bed, offTarget_bed


def run_write_coverInfo(bed_fname, bam_fname, out_fname, by_count, processes, fasta):
    cnarr = coverInfo.do_coverInfo(bed_fname, bam_fname, by_count, 0, processes, fasta)
    tabio.write(cnarr, out_fname)
    return out_fname


def run_run_sample(bam_fname, target_bed, offTarget_bed, ref_fname,
                   output_dir, male_refBaseline, plot_scatter, plot_diagram,
                   rscript_path, by_count, skip_low, seq_method,
                   segment_method, processes, do_cluster, fasta=None):
    sample_id = kernel.fbase(bam_fname)
    sample_pfx = os.path.join(output_dir, sample_id)

    raw_tgt = coverInfo.do_coverInfo(target_bed, bam_fname, by_count, 0,
                                     processes, fasta)
    tabio.write(raw_tgt, sample_pfx + '.targetcoverInfo.tsv')

    raw_anti = coverInfo.do_coverInfo(offTarget_bed, bam_fname, by_count, 0,
                                      processes, fasta)
    tabio.write(raw_anti, sample_pfx + '.offTargetcoverInfo.tsv')

    cnarr = correct.do_correct(raw_tgt, raw_anti, read_cna(ref_fname),
                               do_gc=True, do_edge=(seq_method == "hybrid"), do_rmask=True,
                               do_cluster=do_cluster)
    tabio.write(cnarr, sample_pfx + '.adjusted.tsv')

    segments = segmenting.do_segmentation(cnarr, segment_method,
                                          rscript_path=rscript_path,
                                          skip_low=skip_low,
                                          processes=processes,
                                          **({'threshold': 1e-6}
                                               if seq_method == 'wgs'
                                               else {}))

    seg_metrics = segmIndicators.do_segmetrics(cnarr, segments,
                                           interval_stats=['ci'], alpha=0.5,
                                           smoothed=True)
    tabio.write(seg_metrics, sample_pfx + '.segm.tsv')

    seg_finalcall = finalcall.do_finalcall(seg_metrics, method="none", filters=['ci'])

    seg_alltest = segmIndicators.do_segmetrics(cnarr, seg_finalcall, location_stats=['p_ttest'])

    seg_alltest.center_all("median")
    seg_final = finalcall.do_finalcall(seg_alltest, method="threshold")
    tabio.write(seg_final, sample_pfx + '.finalcall.tsv')

    seg_bintest = prebin.do_bintest(cnarr, seg_finalcall, target_only=True)
    tabio.write(seg_bintest, sample_pfx + '.bintest.tsv')

    if plot_diagram:
        is_xx = cnarr.guess_xx(male_refBaseline)
        outfname = sample_pfx + '-diagram.pdf'
        drawing.create_diagram(cnarr.shift_xx(male_refBaseline, is_xx),
                               seg_final.shift_xx(male_refBaseline, is_xx),
                               0.5, 3, outfname)
