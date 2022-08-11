import argparse
import logging
import os
import sys

import warnings

warnings.filterwarnings('ignore', message="numpy.dtype size changed")
warnings.filterwarnings('ignore', message="numpy.ufunc size changed")

import pandas as pd
from pegeno import tabio, GenomicArray as _GA
from pegeno.rangelabel import to_label

from . import (preprocess, offTarget, dynamicBin, run, prebin, finalcall, kernel,
               coverInfo, outobj, correct, formats,
               indicators, multiprocess, refBaseline, functionsInfo, segmenting,
               segmIndicators, target)
from .implementation import (load_het_snps, read_cna, verify_sample_sex,
                             write_tsv, write_text, write_dataframe)

from ._version import __version__

__all__ = []


def public(fn):
    __all__.append(fn.__name__)
    return fn


AP = argparse.ArgumentParser(
    description="PEcnv, a command-line toolkit for copy number analysis.")
AP_subparsers = AP.add_subparsers(
    help="Sub-commands (use with -h for more info)")


def _cmd_run(args):
    logging.info("PEcnv %s", __version__)

    bad_args_msg = ""
    if args.refBaseline:
        bad_flags = [flag
                     for is_used, flag in (
                         (args.normal is not None, '-n/--normal'),
                         (args.fasta, '-f/--fasta'),
                         (args.targets, '-t/--targets'),
                         (args.offTargets, '-a/--offTargets'),
                         (args.preprocess, '-g/--preprocess'),
                         (args.annotate, '--annotate'),
                         (args.short_names, '--short-names'),
                         (args.target_avg_size, '--target-avg-size'),
                         (args.offTarget_avg_size, '--offTarget-avg-size'),
                         (args.offTarget_min_size, '--offTarget-min-size'),
                     ) if is_used]
        if bad_flags:
            bad_args_msg = ("If -r/--refBaseline is given, options to construct "
                            "a new refBaseline (%s) should not be used."
                            % ", ".join(bad_flags))
    elif args.normal is None:
        bad_args_msg = ("Option -n/--normal must be given to build a new "
                        "refBaseline if -r/--refBaseline is not used.")
    elif args.seq_method in ('hybrid', 'amplicon') and not args.targets:
        bad_args_msg = ("For the '%r' sequencing method, option -t/--targets "
                        "(at least) must be given to build a new refBaseline if "
                        "-r/--refBaseline is not used." % args.seq_method)
    if bad_args_msg:
        sys.exit(bad_args_msg + "\n(See: PEcnv.py run -h)")

    seen_sids = {}
    for fname in (args.bam_files or []) + (args.normal or []):
        sid = kernel.fbase(fname)
        if sid in seen_sids:
            sys.exit("Duplicate sample ID %r (from %s and %s)"
                     % (sid, fname, seen_sids[sid]))
        seen_sids[sid] = fname

    if args.processes < 1:
        import multiprocessing
        args.processes = multiprocessing.cpu_count()

    if not args.refBaseline:

        args.refBaseline, args.targets, args.offTargets = run.run_make_refBaseline(
            args.normal, args.targets, args.offTargets, args.male_refBaseline,
            args.fasta, args.annotate, args.short_names, args.target_avg_size,
            args.preprocess, args.offTarget_avg_size, args.offTarget_min_size,
            args.output_refBaseline, args.output_dir, args.processes,
            args.count_reads, args.seq_method, args.cluster)
    elif args.targets is None and args.offTargets is None:

        ref_arr = read_cna(args.refBaseline)
        targets, offTargets = refBaseline.refBaseline2regions(ref_arr)
        ref_pfx = os.path.join(args.output_dir, kernel.fbase(args.refBaseline))
        args.targets = ref_pfx + '.target-tmp.bed'
        args.offTargets = ref_pfx + '.offTarget-tmp.bed'
        tabio.write(targets, args.targets, 'bed4')
        tabio.write(offTargets, args.offTargets, 'bed4')

    if args.bam_files:
        if args.processes == 1:
            procs_per_bam = 1
            logging.info("Running %d samples in serial", len(args.bam_files))
        else:
            procs_per_bam = max(1, args.processes // len(args.bam_files))
            logging.info("Running %d samples in %d processes "
                         "(that's %d processes per bam)",
                         len(args.bam_files), args.processes, procs_per_bam)

        with multiprocess.pick_pool(args.processes) as pool:
            for bam in args.bam_files:
                pool.submit(run.run_run_sample,
                            bam, args.targets, args.offTargets, args.refBaseline,
                            args.output_dir, args.male_refBaseline, args.scatter,
                            args.diagram, args.rscript_path, args.count_reads,
                            args.drop_low_coverInfo, args.seq_method, args.segment_method, procs_per_bam,
                            args.cluster, args.fasta)
    else:
        logging.info("No tumor/test samples (but %d normal/control samples) "
                     "specified on the command line.",
                     len(args.normal))


P_run = AP_subparsers.add_parser('run', help=_cmd_run.__doc__)
P_run.add_argument('bam_files', nargs='*',
                   help="Mapped sequence reads (.bam)")
P_run.add_argument('-m', '--seq-method', '--method',
                   choices=('hybrid', 'amplicon', 'wgs'), default='hybrid',
                   help="""Sequencing assay type: hybridization capture ('hybrid'),
                targeted amplicon sequencing ('amplicon'), or whole genome
                sequencing ('wgs'). Determines whether and how to use offTarget
                bins. [Default: %(default)s]""")
P_run.add_argument('--segment-method',
                   choices=segmenting.SEGMENT_METHODS,
                   default='cbs',
                   help="""Method used in the 'segment' step. [Default: %(default)s]"""),
P_run.add_argument('-y', '--male-refBaseline', '--haploid-x-refBaseline',
                   action='store_true',
                   help="""Use or assume a male refBaseline (i.e. female samples will have +1
                log-CNR of chrX; otherwise male samples would have -1 chrX).""")
P_run.add_argument('-c', '--count-reads', action='store_true',
                   help="""Get read depths by counting read midpoints within each bin.
                (An alternative algorithm).""")
P_run.add_argument("--drop-low-coverInfo", action='store_true',
                   help="""Drop very-low-coverInfo bins before segmenting to avoid
                false-positive deletions in poor-quality tumor samples.""")
P_run.add_argument('-p', '--processes',
                   nargs='?', type=int, const=0, default=1,
                   help="""Number of subprocesses used to running each of the BAM files in
                multiprocess. Without an argument, use the maximum number of
                available CPUs. [Default: process each BAM in serial]""")
P_run.add_argument("--rscript-path", metavar="PATH", default="Rscript",
                   help="""Path to the Rscript excecutable to use for running R code.
                Use this option to specify a non-default R installation.
                [Default: %(default)s]""")

P_run_newref = P_run.add_argument_group(
    "To construct a new copy number refBaseline")
P_run_newref.add_argument('-n', '--normal', nargs='*', metavar="FILES",
                          help="""Normal samples (.bam) used to construct the pooled, paired, or
                flat refBaseline. If this option is used but no filenames are
                given, a "flat" refBaseline will be built. Otherwise, all
                filenames following this option will be used.""")
P_run_newref.add_argument('-f', '--fasta', metavar="FILENAME",
                          help="refBaseline genome, FASTA format (e.g. UCSC hg19.fa)")
P_run_newref.add_argument('-t', '--targets', metavar="FILENAME",
                          help="Target intervals (.bed or .list)")
P_run_newref.add_argument('-a', '--offTargets', metavar="FILENAME",
                          help="offTarget intervals (.bed or .list)")

P_run_newref.add_argument('--annotate', metavar="FILENAME",
                          help="""Use gene models from this file to assign names to the target
                regions. Format: UCSC refFlat.txt or ensFlat.txt file
                (preferred), or BED, interval list, GFF, or similar.""")
P_run_newref.add_argument('--short-names', action='store_true',
                          help="Reduce multi-preprocession bait labels to be short and consistent.")
P_run_newref.add_argument('--target-avg-size', type=int,
                          help="Average size of split target bins (results are approximate).")

P_run_newref.add_argument('-g', '--preprocess', metavar="FILENAME",
                          help="""Regions of preprocessible sequence on chromosomes (.bed), as
                output by the 'preprocess' command.""")
P_run_newref.add_argument('--offTarget-avg-size', type=int,
                          help="Average size of offTarget bins (results are approximate).")
P_run_newref.add_argument('--offTarget-min-size', type=int,
                          help="Minimum size of offTarget bins (smaller regions are dropped).")
P_run_newref.add_argument('--output-refBaseline', metavar="FILENAME",
                          help="""Output filename/path for the new refBaseline file being created.
                (If given, ignores the -o/--output-dir option and will write the
                file to the given path. Otherwise, \"refBaseline.tsv\" will be
                created in the current directory or specified output directory.)
                """)
P_run_newref.add_argument('--cluster',
                          action='store_true',
                          help="""Calculate and use cluster-specific summary stats in the
                refBaseline pool to normalize samples.""")

P_run_oldref = P_run.add_argument_group("To reuse an existing refBaseline")
P_run_oldref.add_argument('-r', '--refBaseline',
                          help="Copy number refBaseline file (.tsv).")

P_run_report = P_run.add_argument_group("Output options")
P_run_report.add_argument('-d', '--output-dir',
                          metavar="DIRECTORY", default='.',
                          help="Output directory.")
P_run_report.add_argument('--scatter', action='store_true',
                          help="Create a whole-genome copy ratio profile as a PDF scatter plot.")
P_run_report.add_argument('--diagram', action='store_true',
                          help="Create an ideogram of copy ratios on chromosomes as a PDF.")

P_run.set_defaults(func=_cmd_run)

do_target = public(target.do_target)


def _cmd_target(args):
    regions = tabio.read_auto(args.interval)
    regions = target.do_target(regions, args.annotate, args.short_names,
                               args.split, args.avg_size)
    tabio.write(regions, args.output, "bed4")


P_target = AP_subparsers.add_parser('target', help=_cmd_target.__doc__)
P_target.add_argument('interval',
                      help="""BED or interval file listing the targeted regions.""")
P_target.add_argument('--annotate',
                      help="""Use gene models from this file to assign names to the target
                regions. Format: UCSC refFlat.txt or ensFlat.txt file
                (preferred), or BED, interval list, GFF, or similar.""")
P_target.add_argument('--short-names', action='store_true',
                      help="Reduce multi-accession bait labels to be short and consistent.")
P_target.add_argument('--split', action='store_true',
                      help="Split large tiled intervals into smaller, consecutive targets.")

P_target.add_argument('-a', '--avg-size', type=int, default=200 / .75,
                      help="""Average size of split target bins (results are approximate).
                [Default: %(default)s]""")
P_target.add_argument('-o', '--output', metavar="FILENAME",
                      help="""Output file name.""")
P_target.set_defaults(func=_cmd_target)

do_preprocess = public(preprocess.do_preprocess)


def _cmd_preprocess(args):
    preprocess_arr = preprocess.do_preprocess(args.fa_fname, args.exclude,
                                              args.min_gap_size)
    tabio.write(preprocess_arr, args.output, "bed3")


P_preprocess = AP_subparsers.add_parser('preprocess', help=_cmd_preprocess.__doc__)
P_preprocess.add_argument("fa_fname",
                          help="Genome FASTA file name")
P_preprocess.add_argument("-s", "--min-gap-size", type=int, default=5000,
                          help="""Minimum gap size between preprocessible sequence
                regions.  Regions separated by less than this distance will
                be joined together. [Default: %(default)s]""")
P_preprocess.add_argument("-x", "--exclude", action="append", default=[],
                          help="""Additional regions to exclude, in BED format. Can be
                used multiple times.""")
P_preprocess.add_argument("-o", "--output", metavar="FILENAME",
                          type=argparse.FileType('w'), default=sys.stdout,
                          help="Output file name")
P_preprocess.set_defaults(func=_cmd_preprocess)

do_offTarget = public(offTarget.do_offTarget)


def _cmd_offTarget(args):
    targets = tabio.read_auto(args.targets)
    preprocess = tabio.read_auto(args.preprocess) if args.preprocess else None
    out_arr = offTarget.do_offTarget(targets, preprocess, args.avg_size,
                                     args.min_size)
    if not args.output:
        base, ext = args.interval.rsplit('.', 1)
        args.output = base + '.offTarget.' + ext
    tabio.write(out_arr, args.output, "bed4")


P_anti = AP_subparsers.add_parser('offTarget', help=_cmd_offTarget.__doc__)
P_anti.add_argument('targets',
                    help="""BED or interval file listing the targeted regions.""")
P_anti.add_argument('-g', '--preprocess', metavar="FILENAME",
                    help="""Regions of preprocessible sequence on chromosomes (.bed), as
                output by genome2preprocess.py.""")
P_anti.add_argument('-a', '--avg-size', type=int, default=150000,
                    help="""Average size of offTarget bins (results are approximate).
                [Default: %(default)s]""")
P_anti.add_argument('-m', '--min-size', type=int,
                    help="""Minimum size of offTarget bins (smaller regions are dropped).
                [Default: 1/16 avg size, calculated]""")
P_anti.add_argument('-o', '--output', metavar="FILENAME",
                    help="""Output file name.""")
P_anti.set_defaults(func=_cmd_offTarget)

do_autobin = public(dynamicBin.do_autobin)


def _cmd_dynamicBin(args):
    if args.method in ('hybrid', 'amplicon') and not args.targets:
        raise RuntimeError("Sequencing method %r requires targets (-t)",
                           args.method)
    if args.method == 'wgs':
        if not args.preprocess:
            raise RuntimeError("Sequencing method 'wgs' requires preprocessible "
                               "regions (-g)")
        if args.targets:
            logging.warning("Targets will be ignored: %s", args.targets)
    if args.method == 'amplicon' and args.preprocess:
        logging.warning("Sequencing-preprocessible regions will be ignored: %s",
                        args.preprocess)

    def read_regions(bed_fname):
        if bed_fname:
            regions = tabio.read_auto(bed_fname)
            if len(regions):
                return regions
            else:
                logging.warning("No regions to estimate depth from %s",
                                regions.meta.get('filename', ''))

    tgt_arr = read_regions(args.targets)
    preprocess_arr = read_regions(args.preprocess)
    bam_fname = dynamicBin.midsize_file(args.bams)
    fields = dynamicBin.do_autobin(bam_fname, args.method, tgt_arr, preprocess_arr,
                                args.bp_per_bin, args.target_min_size,
                                args.target_max_size, args.offTarget_min_size,
                                args.offTarget_max_size, args.fasta)
    (_tgt_depth, tgt_bin_size), (_anti_depth, anti_bin_size) = fields

    target_out_arr = target.do_target(preprocess_arr if args.method == 'wgs'
                                      else tgt_arr,
                                      args.annotate, args.short_names,
                                      do_split=True, avg_size=tgt_bin_size)
    tgt_name_base = tgt_arr.sample_id if tgt_arr else kernel.fbase(bam_fname)
    target_bed = tgt_name_base + '.target.bed'
    tabio.write(target_out_arr, target_bed, "bed4")
    if args.method == "hybrid" and anti_bin_size:

        anti_arr = offTarget.do_offTarget(target_out_arr,
                                          preprocess=preprocess_arr,
                                          avg_bin_size=anti_bin_size,
                                          min_bin_size=args.offTarget_min_size)
    else:

        anti_arr = _GA([])
    offTarget_bed = tgt_name_base + '.offTarget.bed'
    tabio.write(anti_arr, offTarget_bed, "bed4")

    labels = ("Target", "offTarget")
    width = max(map(len, labels)) + 1
    print(" " * width, "Depth", "Bin size", sep='\t')
    for label, (depth, binsize) in zip(labels, fields):
        if depth is not None:
            print((label + ":").ljust(width),
                  format(depth, ".3f"),
                  binsize,
                  sep='\t')


P_dynamicBin = AP_subparsers.add_parser('dynamicBin', help=_cmd_dynamicBin.__doc__)
P_dynamicBin.add_argument('bams', nargs='+',
                       help="""Sample BAM file(s) to test for target coverInfo""")
P_dynamicBin.add_argument('-f', '--fasta', metavar="FILENAME",
                       help="refBaseline genome, FASTA format (e.g. UCSC hg19.fa)")
P_dynamicBin.add_argument('-m', '--method',
                       choices=('hybrid', 'amplicon', 'wgs'), default='hybrid',
                       help="""Sequencing protocol: hybridization capture ('hybrid'), targeted
                amplicon sequencing ('amplicon'), or whole genome sequencing
                ('wgs'). Determines whether and how to use offTarget bins.
                [Default: %(default)s]""")
P_dynamicBin.add_argument('-g', '--preprocess', metavar="FILENAME",
                       help="""Sequencing-preprocessible genomic regions, or exons to use as
                possible targets (e.g. output of refFlat2bed.py)""")
P_dynamicBin.add_argument('-t', '--targets',
                       help="""Potentially targeted genomic regions, e.g. all possible exons
                for the refBaseline genome. Format: BED, interval list, etc.""")
P_dynamicBin.add_argument('-b', '--bp-per-bin',
                       type=float, default=100000.,
                       help="""Desired average number of sequencing read bases mapped to each
                bin. [Default: %(default)s]""")

P_dynamicBin.add_argument('--target-max-size', metavar="BASES",
                       type=int, default=20000,
                       help="Maximum size of target bins. [Default: %(default)s]")
P_dynamicBin.add_argument('--target-min-size', metavar="BASES",
                       type=int, default=20,
                       help="Minimum size of target bins. [Default: %(default)s]")
P_dynamicBin.add_argument('--offTarget-max-size', metavar="BASES",
                       type=int, default=500000,
                       help="Maximum size of offTarget bins. [Default: %(default)s]")
P_dynamicBin.add_argument('--offTarget-min-size', metavar="BASES",
                       type=int, default=500,
                       help="Minimum size of offTarget bins. [Default: %(default)s]")

P_dynamicBin.add_argument('--annotate', metavar="FILENAME",
                       help="""Use gene models from this file to assign names to the target
                regions. Format: UCSC refFlat.txt or ensFlat.txt file
                (preferred), or BED, interval list, GFF, or similar.""")
P_dynamicBin.add_argument('--short-names', action='store_true',
                       help="Reduce multi-preprocession bait labels to be short and consistent.")

P_dynamicBin.set_defaults(func=_cmd_dynamicBin)

do_coverInfo = public(coverInfo.do_coverInfo)


def _cmd_coverInfo(args):
    """Calculate coverInfo in the given regions from BAM read depths."""
    pset = coverInfo.do_coverInfo(args.interval, args.bam_file, args.count,
                                  args.min_mapq, args.processes, args.fasta)
    if not args.output:

        bambase = kernel.fbase(args.bam_file)
        bedbase = kernel.fbase(args.interval)
        tgtbase = ('offTargetcoverInfo'
                   if 'anti' in bedbase.lower()
                   else 'targetcoverInfo')
        args.output = '%s.%s.tsv' % (bambase, tgtbase)
        if os.path.exists(args.output):
            args.output = '%s.%s.tsv' % (bambase, bedbase)
    kernel.ensure_path(args.output)
    tabio.write(pset, args.output)


P_coverInfo = AP_subparsers.add_parser('coverInfo', help=_cmd_coverInfo.__doc__)
P_coverInfo.add_argument('bam_file', help="Mapped sequence reads (.bam)")
P_coverInfo.add_argument('interval', help="Intervals (.bed or .list)")
P_coverInfo.add_argument('-f', '--fasta', metavar="FILENAME",
                         help="refBaseline genome, FASTA format (e.g. UCSC hg19.fa)")
P_coverInfo.add_argument('-c', '--count', action='store_true',
                         help="""Get read depths by counting read midpoints within each bin.
                (An alternative algorithm).""")
P_coverInfo.add_argument('-q', '--min-mapq', type=int, default=0,
                         help="""Minimum mapping quality score (phred scale 0-60) to count a read
                for coverInfo depth.  [Default: %(default)s]""")
P_coverInfo.add_argument('-o', '--output', metavar="FILENAME",
                         help="""Output file name.""")
P_coverInfo.add_argument('-p', '--processes',
                         nargs='?', type=int, const=0, default=1,
                         help="""Number of subprocesses to calculate coverInfo in multiprocess.
                Without an argument, use the maximum number of available CPUs.
                [Default: use 1 process]""")
P_coverInfo.set_defaults(func=_cmd_coverInfo)

do_refBaseline = public(refBaseline.do_refBaseline)
do_refBaseline_flat = public(refBaseline.do_refBaseline_flat)


def _cmd_refBaseline(args):
    usage_err_msg = ("Give .tsv samples OR targets and (optionally) offTargets.")
    if args.targets:

        assert not args.refBaselines, usage_err_msg
        ref_probes = refBaseline.do_refBaseline_flat(args.targets, args.offTargets,
                                                     args.fasta,
                                                     args.male_refBaseline)
    elif args.refBaselines:

        assert not args.targets and not args.offTargets, usage_err_msg
        filenames = []
        for path in args.refBaselines:
            if os.path.isdir(path):
                filenames.extend(os.path.join(path, f) for f in os.listdir(path)
                                 if f.endswith('targetcoverInfo.tsv'))
            else:
                filenames.append(path)
        targets = [f for f in filenames if 'offTarget' not in f]
        offTargets = [f for f in filenames if 'offTarget' in f]
        logging.info("Number of target and offTarget files: %d, %d",
                     len(targets), len(offTargets))
        female_samples = ((args.sample_sex.lower() not in ['y', 'm', 'male'])
                          if args.sample_sex else None)
        ref_probes = refBaseline.do_refBaseline(targets, offTargets, args.fasta,
                                                args.male_refBaseline, female_samples,
                                                args.do_gc, args.do_edge,
                                                args.do_rmask, args.cluster,
                                                args.min_cluster_size)
    else:
        raise ValueError(usage_err_msg)

    ref_fname = args.output or "cnv_refBaseline.tsv"
    kernel.ensure_path(ref_fname)
    tabio.write(ref_probes, ref_fname)


P_refBaseline = AP_subparsers.add_parser('refBaseline', help=_cmd_refBaseline.__doc__)
P_refBaseline.add_argument('refBaselines', nargs='*',
                           help="""Normal-sample target or offTarget .tsv files, or the
                directory that contains them.""")
P_refBaseline.add_argument('-f', '--fasta',
                           help="refBaseline genome, FASTA format (e.g. UCSC hg19.fa)")
P_refBaseline.add_argument('-o', '--output', metavar="FILENAME",
                           help="Output file name.")
P_refBaseline.add_argument('-c', '--cluster',
                           action='store_true',
                           help="""Calculate and store summary stats for clustered subsets of the
                normal samples with similar coverInfo profiles.""")
P_refBaseline.add_argument('--min-cluster-size',
                           metavar="NUM",
                           type=int,
                           default=4,
                           help="""Minimum cluster size to keep in refBaseline profiles.""")
P_refBaseline.add_argument('-x', '--sample-sex', '-g', '--gender',
                           dest='sample_sex',
                           choices=('m', 'y', 'male', 'Male', 'f', 'x', 'female', 'Female'),
                           help="""Specify the chromosomal sex of all given samples as male or
                female. (Default: guess each sample from coverInfo of X and Y
                chromosomes).""")
P_refBaseline.add_argument('-y', '--male-refBaseline', '--haploid-x-refBaseline',
                           action='store_true',
                           help="""Create a male refBaseline: shift female samples' chrX
                log-coverInfo by -1, so the refBaseline chrX average is -1.
                Otherwise, shift male samples' chrX by +1, so the refBaseline chrX
                average is 0.""")

P_refBaseline_flat = P_refBaseline.add_argument_group(
    "To construct a generic, \"flat\" copy number refBaseline with neutral "
    "expected coverInfo")
P_refBaseline_flat.add_argument('-t', '--targets',
                                help="Target intervals (.bed or .list)")
P_refBaseline_flat.add_argument('-a', '--offTargets',
                                help="offTarget intervals (.bed or .list)")

P_refBaseline_bias = P_refBaseline.add_argument_group(
    "To disable specific automatic bias corrections")
P_refBaseline_bias.add_argument('--no-gc', dest='do_gc', action='store_false',
                                help="Skip GC correction.")
P_refBaseline_bias.add_argument('--no-edge', dest='do_edge', action='store_false',
                                help="Skip edge-effect correction.")
P_refBaseline_bias.add_argument('--no-rmask', dest='do_rmask', action='store_false',
                                help="Skip RepeatMasker correction.")
P_refBaseline.set_defaults(func=_cmd_refBaseline)

do_correct = public(correct.do_correct)


def _cmd_correct(args):
    tgt_raw = read_cna(args.target, sample_id=args.sample_id)
    anti_raw = read_cna(args.offTarget, sample_id=args.sample_id)
    if len(anti_raw) and tgt_raw.sample_id != anti_raw.sample_id:
        raise ValueError("Sample IDs do not match:"
                         "'%s' (target) vs. '%s' (offTarget)"
                         % (tgt_raw.sample_id, anti_raw.sample_id))
    target_table = correct.do_correct(tgt_raw, anti_raw, read_cna(args.refBaseline),
                                      args.do_gc, args.do_edge, args.do_rmask,
                                      args.cluster)
    tabio.write(target_table, args.output or tgt_raw.sample_id + '.adjusted.tsv')


P_correct = AP_subparsers.add_parser('correct', help=_cmd_correct.__doc__)
P_correct.add_argument('target',
                       help="Target coverInfo file (.targetcoverInfo.tsv).")
P_correct.add_argument('offTarget',
                       help="offTarget coverInfo file (.offTargetcoverInfo.tsv).")
P_correct.add_argument('refBaseline',
                       help="refBaseline coverInfo (.tsv).")
P_correct.add_argument('-c', '--cluster',
                       action='store_true',
                       help="""Compare and use cluster-specific values present in the
                refBaseline profile. (Requires that the refBaseline profile
                was built with the --cluster option.)""")
P_correct.add_argument('-i', '--sample-id',
                       help="Sample ID for target/offTarget files. Otherwise inferred from file names.")

P_correct.add_argument('--no-gc', dest='do_gc', action='store_false',
                       help="Skip GC correction.")
P_correct.add_argument('--no-edge', dest='do_edge', action='store_false',
                       help="Skip edge-effect correction.")
P_correct.add_argument('--no-rmask', dest='do_rmask', action='store_false',
                       help="Skip RepeatMasker correction.")
P_correct.add_argument('-o', '--output', metavar="FILENAME",
                       help="Output file name.")
P_correct.set_defaults(func=_cmd_correct)

do_segmentation = public(segmenting.do_segmentation)


def _cmd_segment(args):
    cnarr = read_cna(args.filename)
    variants = load_het_snps(args.vcf, args.sample_id, args.normal_id,
                             args.min_variant_depth, args.zygosity_freq)
    results = segmenting.do_segmentation(cnarr, args.method, args.threshold,
                                         variants=variants,
                                         skip_low=args.drop_low_coverInfo,
                                         skip_outliers=args.drop_outliers,
                                         save_dataframe=bool(args.dataframe),
                                         rscript_path=args.rscript_path,
                                         processes=args.processes,
                                         smooth_cbs=args.smooth_cbs)

    if args.dataframe:
        segments, dframe = results
        with open(args.dataframe, 'w') as handle:
            handle.write(dframe)
        logging.info("Wrote %s", args.dataframe)
    else:
        segments = results
    tabio.write(segments, args.output or segments.sample_id + '.segm.tsv')


P_segment = AP_subparsers.add_parser('segment', help=_cmd_segment.__doc__)
P_segment.add_argument('filename',
                       help="Bin-level log2 ratios (.cnr file), as produced by 'correct'.")
P_segment.add_argument('-o', '--output', metavar="FILENAME",
                       help="Output table file name (CNR-like table of segments, .segm.tsv).")
P_segment.add_argument('-d', '--dataframe',
                       help="""File name to save the raw R dataframe emitted by CBS or
                Fused Lasso. (Useful for debugging.)""")
P_segment.add_argument('-m', '--method',
                       choices=segmenting.SEGMENT_METHODS,
                       default='cbs',
                       help="""Segmentation method (see docs), or 'none' for chromosome
                arm-level averages as segments. [Default: %(default)s]""")
P_segment.add_argument('-t', '--threshold', type=float,
                       help="""Significance threshold (p-value or FDR, depending on method) to
                accept breakpoints during segmenting.
                For HMM methods, this is the adjusting window size.""")
P_segment.add_argument("--drop-low-coverInfo", action='store_true',
                       help="""Drop very-low-coverInfo bins before segmenting to avoid
                false-positive deletions in poor-quality tumor samples.""")
P_segment.add_argument("--drop-outliers", metavar="FACTOR",
                       type=float, default=10,
                       help="""Drop outlier bins more than this many multiples of the 95th
                quantile away from the average within a rolling window.
                Set to 0 for no outlier filtering.
                [Default: %(default)g]""")
P_segment.add_argument("--rscript-path", metavar="PATH", default="Rscript",
                       help="""Path to the Rscript excecutable to use for running R code.
                Use this option to specify a non-default R installation.
                [Default: %(default)s]""")
P_segment.add_argument('-p', '--processes',
                       nargs='?', type=int, const=0, default=1,
                       help="""Number of subprocesses to segment in multiprocess.
                Give 0 or a negative value to use the maximum number
                of available CPUs. [Default: use 1 process]""")
P_segment.add_argument('--smooth-cbs', action='store_true',
                       help="""Perform an additional adjusting before CBS segmenting, 
								which in some cases may increase the sensitivity. 
								Used only for CBS method.""")

P_segment_vcf = P_segment.add_argument_group(
    "To additionally segment SNP b-allele frequencies")
P_segment_vcf.add_argument('-v', '--vcf', metavar="FILENAME",
                           help="""VCF file name containing variants for segmenting by allele
                frequencies.""")
P_segment_vcf.add_argument('-i', '--sample-id',
                           help="""Specify the name of the sample in the VCF (-v/--vcf) to use for
                b-allele frequency extraction and as the default plot title.""")
P_segment_vcf.add_argument('-n', '--normal-id',
                           help="""Corresponding normal sample ID in the input VCF (-v/--vcf).
                This sample is used to select only germline SNVs to plot
                b-allele frequencies.""")
P_segment_vcf.add_argument('--min-variant-depth', type=int, default=20,
                           help="""Minimum read depth for a SNV to be displayed in the b-allele
                frequency plot. [Default: %(default)s]""")
P_segment_vcf.add_argument('-z', '--zygosity-freq',
                           metavar='ALT_FREQ', nargs='?', type=float, const=0.25,
                           help="""Ignore VCF's genotypes (GT field) and instead infer zygosity
                from allele frequencies.  [Default if used without a number:
                %(const)s]""")

P_segment.set_defaults(func=_cmd_segment)

do_finalcall = public(finalcall.do_finalcall)


def _cmd_finalcall(args):
    if args.purity and not 0.0 < args.purity <= 1.0:
        raise RuntimeError("Purity must be between 0 and 1.")

    cnarr = read_cna(args.filename)
    if args.center_at:
        logging.info("Shifting log2 ratios by %f", -args.center_at)
        cnarr['log2'] -= args.center_at
    elif args.center:
        cnarr.center_all(args.center, skip_low=args.drop_low_coverInfo,
                         verbose=True)

    varr = load_het_snps(args.vcf, args.sample_id, args.normal_id,
                         args.min_variant_depth, args.zygosity_freq)
    is_sample_female = (verify_sample_sex(cnarr, args.sample_sex,
                                          args.male_refBaseline)
                        if args.purity and args.purity < 1.0
                        else None)
    cnarr = finalcall.do_finalcall(cnarr, varr, args.method, args.ploidy, args.purity,
                                   args.male_refBaseline, is_sample_female, args.filters,
                                   args.thresholds)
    tabio.write(cnarr, args.output or cnarr.sample_id + '.finalcall.tsv')


def csvstring(text):
    return tuple(map(float, text.split(",")))


P_finalcall = AP_subparsers.add_parser('finalcall', help=_cmd_finalcall.__doc__)
P_finalcall.add_argument('filename',
                         help="Copy ratios (.cnr or .segm.tsv).")
P_finalcall.add_argument("--center", nargs='?', const='median',
                         choices=('mean', 'median', 'mode', 'biweight'),
                         help="""Re-center the log2 ratio values using this estimator of the
                center or average value. ('median' if no argument given.)""")
P_finalcall.add_argument("--center-at", type=float,
                         help="""Subtract a constant number from all log2 ratios. For "manual"
                re-centering, in case the --center option gives unsatisfactory
                results.)""")
P_finalcall.add_argument('--filter', action='append', default=[], dest='filters',
                         choices=('ampdel', 'cn', 'ci', 'sem',
                                  ),
                         help="""Merge segments flagged by the specified filter(s) with the
                adjacent segment(s).""")
P_finalcall.add_argument('-m', '--method',
                         choices=('threshold', 'clonal', 'none'), default='threshold',
                         help="""finalcalling method. [Default: %(default)s]""")
P_finalcall.add_argument('-t', '--thresholds',
                         type=csvstring, default="-1.1,-0.25,0.2,0.7",
                         help="""Hard thresholds for finalcalling each integer copy number, separated
                by commas. Use the '=' sign on the command line, e.g.: -t=-1,0,1
                [Default: %(default)s]""")
P_finalcall.add_argument("--ploidy", type=int, default=2,
                         help="Ploidy of the sample cells. [Default: %(default)d]")
P_finalcall.add_argument("--purity", type=float,
                         help="Estimated tumor cell fraction, a.k.a. purity or cellularity.")
P_finalcall.add_argument("--drop-low-coverInfo", action='store_true',
                         help="""Drop very-low-coverInfo bins before segmenting to avoid
                false-positive deletions in poor-quality tumor samples.""")
P_finalcall.add_argument('-x', '--sample-sex', '-g', '--gender', dest='sample_sex',
                         choices=('m', 'y', 'male', 'Male', 'f', 'x', 'female', 'Female'),
                         help="""Specify the sample's chromosomal sex as male or female.
                (Otherwise guessed from X and Y coverInfo).""")
P_finalcall.add_argument('-y', '--male-refBaseline', '--haploid-x-refBaseline',
                         action='store_true',
                         help="""Was a male refBaseline used?  If so, expect half ploidy on
                chrX and chrY; otherwise, only chrY has half ploidy.  In PEcnv,
                if a male refBaseline was used, the "neutral" copy number (ploidy)
                of chrX is 1; chrY is haploid for either refBaseline sex.""")
P_finalcall.add_argument('-o', '--output', metavar="FILENAME",
                         help="Output table file name (CNR-like table of segments, .segm.tsv).")

P_finalcall_vcf = P_finalcall.add_argument_group(
    "To additionally process SNP b-allele frequencies for allelic copy number")
P_finalcall_vcf.add_argument('-v', '--vcf', metavar="FILENAME",
                             help="""VCF file name containing variants for calculation of b-allele
                frequencies.""")
P_finalcall_vcf.add_argument('-i', '--sample-id',
                             help="""Name of the sample in the VCF (-v/--vcf) to use for b-allele
                frequency extraction.""")
P_finalcall_vcf.add_argument('-n', '--normal-id',
                             help="""Corresponding normal sample ID in the input VCF (-v/--vcf).
                This sample is used to select only germline SNVs to calculate
                b-allele frequencies.""")
P_finalcall_vcf.add_argument('--min-variant-depth', type=int, default=20,
                             help="""Minimum read depth for a SNV to be used in the b-allele
                frequency calculation. [Default: %(default)s]""")
P_finalcall_vcf.add_argument('-z', '--zygosity-freq',
                             metavar='ALT_FREQ', nargs='?', type=float, const=0.25,
                             help="""Ignore VCF's genotypes (GT field) and instead infer zygosity
                from allele frequencies.  [Default if used without a number:
                %(const)s]""")

P_finalcall.set_defaults(func=_cmd_finalcall)

do_breaks = public(functionsInfo.do_breaks)


def _cmd_breaks(args):
    cnarr = read_cna(args.filename)
    segarr = read_cna(args.segment)
    bpoints = do_breaks(cnarr, segarr, args.min_probes)
    logging.info("Found %d gene breakpoints", len(bpoints))
    write_dataframe(args.output, bpoints)


P_breaks = AP_subparsers.add_parser('breaks', help=_cmd_breaks.__doc__)
P_breaks.add_argument('filename',
                      help="""Processed sample coverInfo data file (*.cnr), the output
                of the 'correct' sub-command.""")
P_breaks.add_argument('segment',
                      help="Segmentation finalcalls (.segm.tsv), the output of the 'segment' command).")
P_breaks.add_argument('-m', '--min-probes', type=int, default=1,
                      help="""Minimum number of within-gene probes on both sides of a
                breakpoint to report it. [Default: %(default)d]""")
P_breaks.add_argument('-o', '--output', metavar="FILENAME",
                      help="Output table file name.")
P_breaks.set_defaults(func=_cmd_breaks)

do_genemetrics = public(functionsInfo.do_genemetrics)


def _cmd_geneindicators(args):
    cnarr = read_cna(args.filename)
    segarr = read_cna(args.segment) if args.segment else None
    is_sample_female = verify_sample_sex(cnarr, args.sample_sex,
                                         args.male_refBaseline)

    table = do_genemetrics(cnarr, segarr, args.threshold, args.min_probes,
                           args.drop_low_coverInfo, args.male_refBaseline,
                           is_sample_female)
    logging.info("Found %d gene-level gains and losses", len(table))
    write_dataframe(args.output, table)


P_geneindicators = AP_subparsers.add_parser('geneindicators',
                                         help=_cmd_geneindicators.__doc__)
P_geneindicators.add_argument('filename',
                           help="""Processed sample coverInfo data file (*.cnr), the output
                of the 'correct' sub-command.""")
P_geneindicators.add_argument('-s', '--segment',
                           help="Segmentation finalcalls (.segm.tsv), the output of the 'segment' command).")
P_geneindicators.add_argument('-t', '--threshold', type=float, default=0.2,
                           help="""Copy number change threshold to report a gene gain/loss.
                [Default: %(default)s]""")
P_geneindicators.add_argument('-m', '--min-probes', type=int, default=3,
                           help="""Minimum number of covered probes to report a gain/loss.
                [Default: %(default)d]""")
P_geneindicators.add_argument("--drop-low-coverInfo", action='store_true',
                           help="""Drop very-low-coverInfo bins before segmenting to avoid
                false-positive deletions in poor-quality tumor samples.""")
P_geneindicators.add_argument('-y', '--male-refBaseline', '--haploid-x-refBaseline',
                           action='store_true',
                           help="""Assume inputs were normalized to a male refBaseline
                (i.e. female samples will have +1 log-coverInfo of chrX;
                otherwise male samples would have -1 chrX).""")
P_geneindicators.add_argument('-x', '--sample-sex', '-g', '--gender',
                           dest='sample_sex',
                           choices=('m', 'y', 'male', 'Male', 'f', 'x', 'female', 'Female'),
                           help="""Specify the sample's chromosomal sex as male or female.
                (Otherwise guessed from X and Y coverInfo).""")
P_geneindicators.add_argument('-o', '--output', metavar="FILENAME",
                           help="Output table file name.")

P_geneindicators_stats = P_geneindicators.add_argument_group(
    "Statistics available")

P_geneindicators_stats.add_argument('--mean',
                                 action='append_const', dest='location_stats', const='mean',
                                 help="Mean log2-ratio (unweighted).")
P_geneindicators_stats.add_argument('--median',
                                 action='append_const', dest='location_stats', const='median',
                                 help="Median.")
P_geneindicators_stats.add_argument('--mode',
                                 action='append_const', dest='location_stats', const='mode',
                                 help="Mode (i.e. peak density of log2 ratios).")
P_geneindicators_stats.add_argument('--ttest',
                                 action='append_const', dest='location_stats', const='p_ttest',
                                 help="One-sample t-test of bin log2 ratios versus 0.0.")

P_geneindicators_stats.add_argument('--stdev',
                                 action='append_const', dest='spread_stats', const='stdev',
                                 help="Standard deviation.")
P_geneindicators_stats.add_argument('--sem',
                                 action='append_const', dest='spread_stats', const='sem',
                                 help="Standard error of the mean.")
P_geneindicators_stats.add_argument('--mad',
                                 action='append_const', dest='spread_stats', const='mad',
                                 help="Median absolute deviation (standardized).")
P_geneindicators_stats.add_argument('--mse',
                                 action='append_const', dest='spread_stats', const='mse',
                                 help="Mean squared error.")
P_geneindicators_stats.add_argument('--iqr',
                                 action='append_const', dest='spread_stats', const='iqr',
                                 help="Inter-quartile range.")
P_geneindicators_stats.add_argument('--bivar',
                                 action='append_const', dest='spread_stats', const='bivar',
                                 help="Tukey's biweight midvariance.")

P_geneindicators_stats.add_argument('--ci',
                                 action='append_const', dest='interval_stats', const='ci',
                                 help="Confidence interval (by bootstrap).")
P_geneindicators_stats.add_argument('--pi',
                                 action='append_const', dest='interval_stats', const='pi',
                                 help="Prediction interval.")
P_geneindicators_stats.add_argument('-a', '--alpha', type=float, default=.05,
                                 help="""Level to estimate confidence and prediction intervals;
                use with --ci and --pi. [Default: %(default)s]""")
P_geneindicators_stats.add_argument('-b', '--bootstrap', type=int, default=100,
                                 help="""Number of bootstrap iterations to estimate confidence interval;
                use with --ci. [Default: %(default)d]""")
P_geneindicators_stats.set_defaults(location_stats=[], spread_stats=[],
                                 interval_stats=[])
P_geneindicators.set_defaults(func=_cmd_geneindicators)

AP_subparsers._name_parser_map['gainloss'] = P_geneindicators
do_gainloss = public(do_genemetrics)


def _cmd_sex(args):
    cnarrs = map(read_cna, args.filenames)
    table = do_sex(cnarrs, args.male_refBaseline)
    write_dataframe(args.output, table, header=True)


@public
def do_sex(cnarrs, is_male_refBaseline):
    def strsign(num):
        if num > 0:
            return "+%.3g" % num
        return "%.3g" % num

    def guess_and_format(cna):
        is_xy, stats = cna.compare_sex_chromosomes(is_male_refBaseline)
        return (cna.meta["filename"] or cna.sample_id,
                "Male" if is_xy else "Female",
                strsign(stats['chrx_ratio']) if stats else "NA",
                strsign(stats['chry_ratio']) if stats else "NA")

    rows = (guess_and_format(cna) for cna in cnarrs)
    columns = ["sample", "sex", "X_logratio", "Y_logratio"]
    return pd.DataFrame.from_records(rows, columns=columns)


P_sex = AP_subparsers.add_parser('sex', help=_cmd_sex.__doc__)
P_sex.add_argument('filenames', nargs='+',
                   help="Copy number or copy ratio files (*.tsv, *.cnr).")
P_sex.add_argument('-y', '--male-refBaseline', '--haploid-x-refBaseline',
                   action='store_true',
                   help="""Assume inputs were normalized to a male refBaseline
                (i.e. female samples will have +1 log-coverInfo of chrX;
                otherwise male samples would have -1 chrX).""")
P_sex.add_argument('-o', '--output', metavar="FILENAME",
                   help="Output table file name.")
P_sex.set_defaults(func=_cmd_sex)

AP_subparsers._name_parser_map['gender'] = P_sex
do_gender = public(do_sex)

do_metrics = public(indicators.do_metrics)


def _cmd_indicators(args):
    if (len(args.cnarrays) > 1 and
            args.segments and len(args.segments) > 1 and
            len(args.cnarrays) != len(args.segments)):
        raise ValueError("Number of coverInfo/segment filenames given must be "
                         "equal, if more than 1 segment file is given.")

    cnarrs = map(read_cna, args.cnarrays)
    if args.segments:
        args.segments = map(read_cna, args.segments)
    table = indicators.do_metrics(cnarrs, args.segments, args.drop_low_coverInfo)
    write_dataframe(args.output, table)


P_indicators = AP_subparsers.add_parser('indicators', help=_cmd_indicators.__doc__)
P_indicators.add_argument('cnarrays', nargs='+',
                       help="""One or more bin-level coverInfo data files (*.tsv, *.cnr).""")
P_indicators.add_argument('-s', '--segments', nargs='+',
                       help="""One or more segmenting data files (*.segm.tsv, output of the
                'segment' command).  If more than one file is given, the number
                must match the coverInfo data files, in which case the input
                files will be paired together in the given order. Otherwise, the
                same segments will be used for all coverInfo files.""")
P_indicators.add_argument("--drop-low-coverInfo", action='store_true',
                       help="""Drop very-low-coverInfo bins before calculations to reduce
                negative "fat tail" of bin log2 values in poor-quality
                tumor samples.""")
P_indicators.add_argument('-o', '--output', metavar="FILENAME",
                       help="Output table file name.")
P_indicators.set_defaults(func=_cmd_indicators)

do_segmetrics = public(segmIndicators.do_segmetrics)


def _cmd_segmIndicators(args):
    if not 0.0 < args.alpha <= 1.0:
        raise RuntimeError("alpha must be between 0 and 1.")

    if not any((args.location_stats, args.spread_stats, args.interval_stats)):
        logging.info("No stats specified")
        return

    cnarr = read_cna(args.cnarray)
    if args.drop_low_coverInfo:
        cnarr = cnarr.drop_low_coverInfo()
    segarr = read_cna(args.segments)
    segarr = do_segmetrics(cnarr, segarr, args.location_stats,
                           args.spread_stats, args.interval_stats,
                           args.alpha, args.bootstrap, args.smooth_bootstrap)
    tabio.write(segarr, args.output or segarr.sample_id + ".segmIndicators.segm.tsv")


P_segmIndicators = AP_subparsers.add_parser('segmIndicators', help=_cmd_segmIndicators.__doc__)
P_segmIndicators.add_argument('cnarray',
                          help="""Bin-level copy ratio data file (*.tsv, *.cnr).""")
P_segmIndicators.add_argument('-s', '--segments', required=True,
                          help="Segmentation data file (*.segm.tsv, output of the 'segment' command).")
P_segmIndicators.add_argument("--drop-low-coverInfo", action='store_true',
                          help="""Drop very-low-coverInfo bins before calculations to avoid
                negative bias in poor-quality tumor samples.""")
P_segmIndicators.add_argument('-o', '--output', metavar="FILENAME",
                          help="Output table file name.")

P_segmIndicators_stats = P_segmIndicators.add_argument_group(
    "Statistics available")

P_segmIndicators_stats.add_argument('--mean',
                                action='append_const', dest='location_stats', const='mean',
                                help="Mean log2 ratio (unweighted).")
P_segmIndicators_stats.add_argument('--median',
                                action='append_const', dest='location_stats', const='median',
                                help="Median.")
P_segmIndicators_stats.add_argument('--mode',
                                action='append_const', dest='location_stats', const='mode',
                                help="Mode (i.e. peak density of bin log2 ratios).")
P_segmIndicators_stats.add_argument('--t-test',
                                action='append_const', dest='location_stats', const='p_ttest',
                                help="One-sample t-test of bin log2 ratios versus 0.0.")

P_segmIndicators_stats.add_argument('--stdev',
                                action='append_const', dest='spread_stats', const='stdev',
                                help="Standard deviation.")
P_segmIndicators_stats.add_argument('--sem',
                                action='append_const', dest='spread_stats', const='sem',
                                help="Standard error of the mean.")
P_segmIndicators_stats.add_argument('--mad',
                                action='append_const', dest='spread_stats', const='mad',
                                help="Median absolute deviation (standardized).")
P_segmIndicators_stats.add_argument('--mse',
                                action='append_const', dest='spread_stats', const='mse',
                                help="Mean squared error.")
P_segmIndicators_stats.add_argument('--iqr',
                                action='append_const', dest='spread_stats', const='iqr',
                                help="Inter-quartile range.")
P_segmIndicators_stats.add_argument('--bivar',
                                action='append_const', dest='spread_stats', const='bivar',
                                help="Tukey's biweight midvariance.")

P_segmIndicators_stats.add_argument('--ci',
                                action='append_const', dest='interval_stats', const='ci',
                                help="Confidence interval (by bootstrap).")
P_segmIndicators_stats.add_argument('--pi',
                                action='append_const', dest='interval_stats', const='pi',
                                help="Prediction interval.")
P_segmIndicators_stats.add_argument('-a', '--alpha',
                                type=float, default=.05,
                                help="""Level to estimate confidence and prediction intervals;
                use with --ci and --pi. [Default: %(default)s]""")
P_segmIndicators_stats.add_argument('-b', '--bootstrap',
                                type=int, default=100,
                                help="""Number of bootstrap iterations to estimate confidence interval;
                use with --ci. [Default: %(default)d]""")
P_segmIndicators_stats.add_argument('--smooth-bootstrap',
                                action='store_true',
                                help="""Apply Gaussian noise to bootstrap samples, a.k.a. smoothed
                bootstrap, to estimate confidence interval; use with --ci.
                """)

P_segmIndicators_stats.set_defaults(location_stats=[], spread_stats=[],
                                interval_stats=[])
P_segmIndicators.set_defaults(func=_cmd_segmIndicators)

do_bintest = public(prebin.do_bintest)


def _cmd_prebin(args):
    cnarr = read_cna(args.cnarray)
    segments = read_cna(args.segment) if args.segment else None
    sig = do_bintest(cnarr, segments, args.alpha, args.target)
    tabio.write(sig, args.output or sys.stdout)


P_prebin = AP_subparsers.add_parser('prebin', help=_cmd_prebin.__doc__)
P_prebin.add_argument('cnarray',
                       help="Bin-level log2 ratios (.cnr file), as produced by 'correct'.")
P_prebin.add_argument('-s', '--segment', metavar="FILENAME",
                       help="""Segmentation finalcalls (.segm.tsv), the output of the
                'segment' command).""")
P_prebin.add_argument("-a", "--alpha", type=float, default=0.005,
                       help="Significance threhold. [Default: %(default)s]")
P_prebin.add_argument("-t", "--target", action="store_true",
                       help="Test target bins only; ignore off-target bins.")
P_prebin.add_argument("-o", "--output",
                       help="Output filename.")
P_prebin.set_defaults(func=_cmd_prebin)


def _cmd_import_picard(args):
    for fname in args.targets:
        if not os.path.isfile(fname):
            raise ValueError("Not a file: %s" % fname)
        garr = formats.do_import_picard(fname)
        outfname = ("{}.{}targetcoverInfo.tsv"
                    .format(garr.sample_id,
                            'anti' if 'offTarget' in fname else ''))
        if args.output_dir:
            if not os.path.isdir(args.output_dir):
                os.mkdir(args.output_dir)
                logging.info("Created directory %s", args.output_dir)
            outfname = os.path.join(args.output_dir, outfname)
        tabio.write(garr, outfname)


P_import_picard = AP_subparsers.add_parser('import-picard',
                                           help=_cmd_import_picard.__doc__)
P_import_picard.add_argument('targets', nargs='+',
                             help="""Sample coverInfo .csv files (target and offTarget).""")
P_import_picard.add_argument('-d', '--output-dir',
                             metavar="DIRECTORY", default='.',
                             help="Output directory name.")
P_import_picard.set_defaults(func=_cmd_import_picard)


def _cmd_import_seg(args):
    from .cnv import CopyNumArray as _CNA
    if args.chromosomes:
        if args.chromosomes == 'human':
            chrom_names = {'23': 'X', '24': 'Y', '25': 'M'}
        else:
            chrom_names = dict(kv.split(':')
                               for kv in args.chromosomes.split(','))
    else:
        chrom_names = args.chromosomes
    for sid, segtable in tabio.seg.parse_seg(args.segfile, chrom_names,
                                             args.precorrect, args.from_log10):
        segarr = _CNA(segtable, {"sample_id": sid})
        tabio.write(segarr, os.path.join(args.output_dir, sid + '.segm.tsv'))


P_import_seg = AP_subparsers.add_parser('import-seg',
                                        help=_cmd_import_seg.__doc__)
P_import_seg.add_argument('segfile',
                          help="""Input file in SEG format. May contain multiple samples.""")
P_import_seg.add_argument('-c', '--chromosomes',
                          help="""Mapping of chromosome indexes to names. Syntax:
                "from1:to1,from2:to2". Or use "human" for the preset:
                "23:X,24:Y,25:M".""")
P_import_seg.add_argument('-p', '--precorrect',
                          help="""Precorrect to add to chromosome names (e.g 'chr' to rename '8' in
                the SEG file to 'chr8' in the output).""")
P_import_seg.add_argument('--from-log10', action='store_true',
                          help="Convert base-10 logarithm values in the input to base-2 logs.")
P_import_seg.add_argument('-d', '--output-dir',
                          metavar="DIRECTORY", default='.',
                          help="Output directory name.")
P_import_seg.set_defaults(func=_cmd_import_seg)

do_import_theta = public(formats.do_import_theta)


def _cmd_import_theta(args):
    tumor_segs = read_cna(args.tumor_segm.tsv)
    for i, new_segm in enumerate(do_import_theta(tumor_segs, args.theta_results,
                                                 args.ploidy)):
        tabio.write(new_segm,
                    os.path.join(args.output_dir,
                                 "%s-%d.segm.tsv" % (tumor_segs.sample_id, i + 1)))


P_import_theta = AP_subparsers.add_parser('import-theta',
                                          help=_cmd_import_theta.__doc__)
P_import_theta.add_argument("tumor_segm.tsv")
P_import_theta.add_argument("theta_results")
P_import_theta.add_argument("--ploidy", type=int, default=2,
                            help="Ploidy of normal cells. [Default: %(default)d]")
P_import_theta.add_argument('-d', '--output-dir',
                            metavar="DIRECTORY", default='.',
                            help="Output directory name.")
P_import_theta.set_defaults(func=_cmd_import_theta)

P_outobj = AP_subparsers.add_parser('outobj',
                                    help="""Convert PEcnv output files to another format.""")
P_outobj_subparsers = P_outobj.add_subparsers(
    help="outobj formats (use with -h for more info).")


def _cmd_export_bed(args):
    bed_tables = []
    for segfname in args.segments:
        segments = read_cna(segfname)

        is_sample_female = verify_sample_sex(segments, args.sample_sex,
                                             args.male_refBaseline)
        if args.sample_id:
            label = args.sample_id
        elif args.label_genes:
            label = None
        else:
            label = segments.sample_id
        tbl = outobj.export_bed(segments, args.ploidy,
                                args.male_refBaseline, is_sample_female,
                                label, args.show)
        bed_tables.append(tbl)
    table = pd.concat(bed_tables)
    write_dataframe(args.output, table, header=False)


P_export_bed = P_outobj_subparsers.add_parser('bed',
                                              help=_cmd_export_bed.__doc__)
P_export_bed.add_argument('segments', nargs='+',
                          help="""Segmented copy ratio data files (*.segm.tsv), the output of the
                'segment' or 'finalcall' sub-commands.""")
P_export_bed.add_argument("-i", "--sample-id", metavar="LABEL",
                          help="""Identifier to write in the 4th column of the BED file.
                [Default: use the sample ID, taken from the file name]""")
P_export_bed.add_argument('--label-genes', action='store_true',
                          help="""Show gene names in the 4th column of the BED file.
        (This is a bad idea if >1 input files are given.)""")
P_export_bed.add_argument("--ploidy", type=int, default=2,
                          help="Ploidy of the sample cells. [Default: %(default)d]")
P_export_bed.add_argument('-x', '--sample-sex', '-g', '--gender',
                          dest='sample_sex',
                          choices=('m', 'y', 'male', 'Male', 'f', 'x', 'female', 'Female'),
                          help="""Specify the sample's chromosomal sex as male or female.
                (Otherwise guessed from X and Y coverInfo).""")
P_export_bed.add_argument("--show",
                          choices=('ploidy', 'variant', 'all'), default="ploidy",
                          help="""Which segmented regions to show:
                'all' = all segment regions;
                'variant' = CNA regions with non-neutral copy number;
                'ploidy' = CNA regions with non-default ploidy.
                [Default: %(default)s]""")
P_export_bed.add_argument('-y', '--male-refBaseline', '--haploid-x-refBaseline',
                          action='store_true',
                          help="""Was a male refBaseline used?  If so, expect half ploidy on
                chrX and chrY; otherwise, only chrY has half ploidy.  In PEcnv,
                if a male refBaseline was used, the "neutral" copy number (ploidy)
                of chrX is 1; chrY is haploid for either refBaseline sex.""")
P_export_bed.add_argument('-o', '--output', metavar="FILENAME",
                          help="Output file name.")
P_export_bed.set_defaults(func=_cmd_export_bed)


def _cmd_export_seg(args):
    table = outobj.export_seg(args.filenames, chrom_ids=args.enumerate_chroms)
    write_dataframe(args.output, table)


P_export_seg = P_outobj_subparsers.add_parser('seg',
                                              help=_cmd_export_seg.__doc__)
P_export_seg.add_argument('filenames', nargs='+',
                          help="""Segmented copy ratio data file(s) (*.segm.tsv), the output of the
                'segment' sub-command.""")
P_export_seg.add_argument('--enumerate-chroms', action='store_true',
                          help="""Replace chromosome names with sequential integer IDs.""")
P_export_seg.add_argument('-o', '--output', metavar="FILENAME",
                          help="Output file name.")
P_export_seg.set_defaults(func=_cmd_export_seg)


def _cmd_export_vcf(args):
    segarr = read_cna(args.segments)
    cnarr = read_cna(args.cnr) if args.cnr else None
    is_sample_female = verify_sample_sex(segarr, args.sample_sex,
                                         args.male_refBaseline)
    header, body = outobj.export_vcf(segarr, args.ploidy, args.male_refBaseline,
                                     is_sample_female, args.sample_id, cnarr)
    write_text(args.output, header, body)


P_export_vcf = P_outobj_subparsers.add_parser('vcf',
                                              help=_cmd_export_vcf.__doc__)
P_export_vcf.add_argument('segments',
                          help="""Segmented copy ratio data file (*.segm.tsv), the output of the
                'segment' or 'finalcall' sub-commands.""")

P_export_vcf.add_argument("--cnr",
                          help="""Bin-level copy ratios (*.cnr). Used to indicate fuzzy boundaries
                for segments in the output VCF via the CIPOS and CIEND tags.""")
P_export_vcf.add_argument("-i", "--sample-id", metavar="LABEL",
                          help="""Sample name to write in the genotype field of the output VCF file.
                [Default: use the sample ID, taken from the file name]""")
P_export_vcf.add_argument("--ploidy", type=int, default=2,
                          help="Ploidy of the sample cells. [Default: %(default)d]")
P_export_vcf.add_argument('-x', '--sample-sex', '-g', '--gender',
                          dest='sample_sex',
                          choices=('m', 'y', 'male', 'Male', 'f', 'x', 'female', 'Female'),
                          help="""Specify the sample's chromosomal sex as male or female.
                (Otherwise guessed from X and Y coverInfo).""")
P_export_vcf.add_argument('-y', '--male-refBaseline', '--haploid-x-refBaseline',
                          action='store_true',
                          help="""Was a male refBaseline used?  If so, expect half ploidy on
                chrX and chrY; otherwise, only chrY has half ploidy.  In PEcnv,
                if a male refBaseline was used, the "neutral" copy number (ploidy)
                of chrX is 1; chrY is haploid for either refBaseline sex.""")
P_export_vcf.add_argument('-o', '--output', metavar="FILENAME",
                          help="Output file name.")
P_export_vcf.set_defaults(func=_cmd_export_vcf)


def _cmd_export_theta(args):
    tumor_cn = read_cna(args.tumor_segment)
    normal_cn = read_cna(args.refBaseline) if args.refBaseline else None
    table = outobj.export_theta(tumor_cn, normal_cn)
    if not args.output:
        args.output = tumor_cn.sample_id + ".interval_count"
    table.to_csv(args.output, sep='\t', index=False)
    logging.info("Wrote %s", args.output)
    if args.vcf:
        variants = load_het_snps(args.vcf,
                                 args.sample_id,
                                 args.normal_id, args.min_variant_depth,
                                 args.zygosity_freq)
        if not len(variants):
            raise ValueError("VCF contains no usable SNV records")
        try:
            tumor_snps, normal_snps = outobj.export_theta_snps(variants)
        except ValueError:
            raise ValueError("VCF does not contain any tumor/normal paired "
                             "samples")
        for title, table in [("tumor", tumor_snps), ("normal", normal_snps)]:
            out_fname = "{}.{}.snp_formatted.txt".format(tumor_cn.sample_id, title)
            table.to_csv(out_fname, sep='\t', index=False)
            logging.info("Wrote %s", out_fname)


P_export_theta = P_outobj_subparsers.add_parser('theta',
                                                help=_cmd_export_theta.__doc__)
P_export_theta.add_argument('tumor_segment',
                            help="""Tumor-sample segmenting file from PEcnv (.segm.tsv).""")
P_export_theta.add_argument('-r', '--refBaseline',
                            help="""refBaseline copy number profile (.tsv), or normal-sample bin-level
                log2 copy ratios (.cnr). Use if the tumor_segment input file
                does not contain a "weight" column.""")
P_export_theta.add_argument('-o', '--output', metavar="FILENAME",
                            help="Output file name.")

P_extheta_vcf = P_export_theta.add_argument_group(
    "To also output tables of SNP b-allele frequencies for THetA2")
P_extheta_vcf.add_argument('-v', '--vcf',
                           help="""VCF file containing SNVs observed in both the tumor and normal
                samples. Tumor sample ID should match the `tumor_segment`
                filename or be specified with -i/--sample-id.""")
P_extheta_vcf.add_argument('-i', '--sample-id',
                           help="""Specify the name of the tumor sample in the VCF (given with
                -v/--vcf). [Default: taken the tumor_segment file name]""")
P_extheta_vcf.add_argument('-n', '--normal-id',
                           help="Corresponding normal sample ID in the input VCF.")
P_extheta_vcf.add_argument('-m', '--min-variant-depth', type=int, default=20,
                           help="""Minimum read depth for a SNP in the VCF to be counted.
                [Default: %(default)s]""")
P_extheta_vcf.add_argument('-z', '--zygosity-freq',
                           metavar='ALT_FREQ', nargs='?', type=float, const=0.25,
                           help="""Ignore VCF's genotypes (GT field) and instead infer zygosity
                from allele frequencies.  [Default if used without a number:
                %(const)s]""")

P_export_theta.set_defaults(func=_cmd_export_theta)


def _cmd_outobj_nb(args):
    cnarr = read_cna(args.filename)
    table = outobj.export_nexus_basic(cnarr)
    write_dataframe(args.output, table)


P_outobj_nb = P_outobj_subparsers.add_parser('nexus-basic',
                                             help=_cmd_outobj_nb.__doc__)
P_outobj_nb.add_argument('filename',
                         help="""Log2 copy ratio data file (*.cnr), the output of the 'correct'
                sub-command.""")
P_outobj_nb.add_argument('-o', '--output', metavar="FILENAME",
                         help="Output file name.")
P_outobj_nb.set_defaults(func=_cmd_outobj_nb)


def _cmd_outobj_nbo(args):
    cnarr = read_cna(args.filename)
    varr = load_het_snps(args.vcf, args.sample_id, args.normal_id,
                         args.min_variant_depth, args.zygosity_freq)
    table = outobj.export_nexus_ogt(cnarr, varr, args.min_weight)
    write_dataframe(args.output, table)


P_outobj_nbo = P_outobj_subparsers.add_parser('nexus-ogt',
                                              help=_cmd_outobj_nbo.__doc__)
P_outobj_nbo.add_argument('filename',
                          help="""Log2 copy ratio data file (*.cnr), the output of the 'correct'
                sub-command.""")
P_outobj_nbo.add_argument('vcf',
                          help="""VCF of SNVs for the same sample, to calculate b-allele
                frequencies.""")
P_outobj_nbo.add_argument('-i', '--sample-id',
                          help="""Specify the name of the sample in the VCF to use to extract
                b-allele frequencies.""")
P_outobj_nbo.add_argument('-n', '--normal-id',
                          help="Corresponding normal sample ID in the input VCF.")
P_outobj_nbo.add_argument('-m', '--min-variant-depth', type=int, default=20,
                          help="""Minimum read depth for a SNV to be included in the b-allele
                frequency calculation. [Default: %(default)s]""")
P_outobj_nbo.add_argument('-z', '--zygosity-freq',
                          metavar='ALT_FREQ', nargs='?', type=float, const=0.25,
                          help="""Ignore VCF's genotypes (GT field) and instead infer zygosity
                from allele frequencies.  [Default if used without a number:
                %(const)s]""")
P_outobj_nbo.add_argument('-w', '--min-weight', type=float, default=0.0,
                          help="""Minimum weight (between 0 and 1) for a bin to be included in
                the output. [Default: %(default)s]""")
P_outobj_nbo.add_argument('-o', '--output', metavar="FILENAME",
                          help="Output file name.")
P_outobj_nbo.set_defaults(func=_cmd_outobj_nbo)


def _cmd_outobj_cdt(args):
    sample_ids = list(map(kernel.fbase, args.filenames))
    table = outobj.merge_samples(args.filenames)
    formatter = outobj.EXPORT_FORMATS['cdt']
    outheader, outrows = formatter(sample_ids, table)
    write_tsv(args.output, outrows, colnames=outheader)


P_outobj_cdt = P_outobj_subparsers.add_parser('cdt',
                                              help=_cmd_outobj_cdt.__doc__)
P_outobj_cdt.add_argument('filenames', nargs='+',
                          help="""Log2 copy ratio data file(s) (*.cnr), the output of the
                'correct' sub-command.""")
P_outobj_cdt.add_argument('-o', '--output', metavar="FILENAME",
                          help="Output file name.")
P_outobj_cdt.set_defaults(func=_cmd_outobj_cdt)


def _cmd_outobj_jtv(args):
    sample_ids = list(map(kernel.fbase, args.filenames))
    table = outobj.merge_samples(args.filenames)
    formatter = outobj.EXPORT_FORMATS['jtv']
    outheader, outrows = formatter(sample_ids, table)
    write_tsv(args.output, outrows, colnames=outheader)


P_outobj_jtv = P_outobj_subparsers.add_parser('jtv',
                                              help=_cmd_outobj_jtv.__doc__)
P_outobj_jtv.add_argument('filenames', nargs='+',
                          help="""Log2 copy ratio data file(s) (*.cnr), the output of the
                'correct' sub-command.""")
P_outobj_jtv.add_argument('-o', '--output', metavar="FILENAME",
                          help="Output file name.")
P_outobj_jtv.set_defaults(func=_cmd_outobj_jtv)


def print_version(_args):
    print(__version__)


P_version = AP_subparsers.add_parser('version', help=print_version.__doc__)
P_version.set_defaults(func=print_version)


def parse_args(args=None):
    return AP.parse_args(args=args)
