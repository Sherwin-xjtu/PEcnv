"""The 'run' command."""
import logging
import os

from matplotlib import pyplot
from pegeno import tabio, GenomicArray as GA

from . import (preprocess, offTarget, autobin, bintest, finalcall, kernel, coverInfo,
               diagram, correct, parallel, refBaseline, segmentation,
               segmetrics, target)
from .cmdutil import read_cna


def run_make_refBaseline(normal_bams, target_bed, offTarget_bed,
                         male_refBaseline, fasta, annotate, short_names,
                         target_avg_size, preprocess_bed, offTarget_avg_size,
                         offTarget_min_size, output_refBaseline, output_dir,
                         processes, by_count, method, do_cluster):
    """Build the CN refBaseline from normal samples, targets and offTarget."""
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
            # TODO check if target_bed has gene names
            logging.warning("WGS protocol: recommend '--annotate' option "
                            "(e.g. refFlat.txt) to help locate genes "
                            "in output files.")
        preprocess_arr = None
        if not target_bed:
            # TODO - drop weird contigs before writing, see offTarget.py
            if preprocess_bed:
                target_bed = preprocess_bed
            elif fasta:
                # Run 'preprocess' on the fly
                preprocess_arr = preprocess.do_preprocess(fasta)
                # Take filename base from FASTA, lacking any other clue
                target_bed = os.path.splitext(os.path.basename(fasta)
                                              )[0] + ".bed"
                tabio.write(preprocess_arr, target_bed, "bed3")
            else:
                raise ValueError("WGS protocol: need to provide --targets, "
                                 "--preprocess, or --fasta options.")

        # Tweak default parameters
        if not target_avg_size:
            if normal_bams:
                # Calculate bin size from .bai & preprocess
                if fasta and not preprocess_arr:
                    # Calculate wgs depth from all
                    # sequencing-preprocessible area (it doesn't take that long
                    # compared to WGS coverInfo); user-provided preprocess might be
                    # something else that excludes a significant number of
                    # mapped reads.
                    preprocess_arr = preprocess.do_preprocess(fasta)
                if preprocess_arr:
                    autobin_args = ['wgs', None, preprocess_arr]
                else:
                    # Don't assume the given targets/preprocess covers the whole
                    # genome; use autobin sampling to estimate bin size, as we
                    # do for amplicon
                    bait_arr = tabio.read_auto(target_bed)
                    autobin_args = ['amplicon', bait_arr]
                # Choose median-size normal bam or tumor bam
                bam_fname = autobin.midsize_file(normal_bams)
                (wgs_depth, target_avg_size), _ = autobin.do_autobin(
                    bam_fname, *autobin_args, bp_per_bin=50000., fasta=fasta)
                logging.info("WGS average depth %.2f --> using bin size %d",
                             wgs_depth, target_avg_size)
            else:
                # This bin size is OK down to 10x
                target_avg_size = 5000

    # To make temporary filenames for processed targets or offTarget
    tgt_name_base, _tgt_ext = os.path.splitext(os.path.basename(target_bed))
    if output_dir:
        tgt_name_base = os.path.join(output_dir, tgt_name_base)

    # Pre-process baits/targets
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
        # Devise a temporary offTarget filename
        offTarget_bed = tgt_name_base + '.offTarget.bed'
        if method == "hybrid":
            # Build offTarget BED from the given targets
            anti_kwargs = {}
            if preprocess_bed:
                anti_kwargs['preprocess'] = tabio.read_auto(preprocess_bed)
            if offTarget_avg_size:
                anti_kwargs['avg_bin_size'] = offTarget_avg_size
            if offTarget_min_size:
                anti_kwargs['min_bin_size'] = offTarget_min_size
            anti_arr = offTarget.do_offTarget(target_arr, **anti_kwargs)
        else:
            # No offTarget for wgs, amplicon
            anti_arr = GA([])
        tabio.write(anti_arr, offTarget_bed, "bed4")

    if len(normal_bams) == 0:
        logging.info("Building a flat refBaseline...")
        ref_arr = refBaseline.do_refBaseline_flat(target_bed, offTarget_bed, fasta,
                                                male_refBaseline)
    else:
        logging.info("Building a copy number refBaseline from normal samples...")
        # Run coverInfo on all normals
        with parallel.pick_pool(processes) as pool:
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
        # Build refBaseline from *.cnn
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
    """Run coverInfo on one sample, write to file."""
    cnarr = coverInfo.do_coverInfo(bed_fname, bam_fname, by_count, 0, processes, fasta)
    tabio.write(cnarr, out_fname)
    return out_fname


def run_run_sample(bam_fname, target_bed, offTarget_bed, ref_fname,
                     output_dir, male_refBaseline, plot_scatter, plot_diagram,
                     rscript_path, by_count, skip_low, seq_method,
                     segment_method, processes, do_cluster, fasta=None):
    """Run the pipeline on one BAM file."""
    # ENH - return probes, segments (cnarr, segarr)
    logging.info("Running the pecnv pipeline on %s ...", bam_fname)
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

    logging.info("Segmenting %s.adjusted.tsv ...", sample_pfx)
    segments = segmentation.do_segmentation(cnarr, segment_method,
                                            rscript_path=rscript_path,
                                            skip_low=skip_low,
                                            processes=processes,
                                            **({'threshold': 1e-6}
                                               if seq_method == 'wgs'
                                               else {}))
    logging.info("Post-processing %s.segm.tsv ...", sample_pfx)
    # TODO/ENH take centering shift & apply to .cnr for use in segmetrics
    seg_metrics = segmetrics.do_segmetrics(cnarr, segments,
                                           interval_stats=['ci'], alpha=0.5,
                                           smoothed=True)
    tabio.write(seg_metrics, sample_pfx + '.segm.tsv')

    # Remove likely false-positive breakpoints
    seg_finalcall = finalcall.do_finalcall(seg_metrics, method="none", filters=['ci'])
    # Calculate another segment-level test p-value
    seg_alltest = segmetrics.do_segmetrics(cnarr, seg_finalcall, location_stats=['p_ttest'])
    # Finally, assign absolute copy number values to each segment
    seg_alltest.center_all("median")
    seg_final = finalcall.do_finalcall(seg_alltest, method="threshold")
    tabio.write(seg_final, sample_pfx + '.finalcall.tsv')

    # Test for single-bin CNVs separately
    seg_bintest = bintest.do_bintest(cnarr, seg_finalcall, target_only=True)
    tabio.write(seg_bintest, sample_pfx + '.bintest.tsv')


    if plot_diagram:
        is_xx = cnarr.guess_xx(male_refBaseline)
        outfname = sample_pfx + '-diagram.pdf'
        diagram.create_diagram(cnarr.shift_xx(male_refBaseline, is_xx),
                               seg_final.shift_xx(male_refBaseline, is_xx),
                               0.5, 3, outfname)
        logging.info("Wrote %s", outfname)
