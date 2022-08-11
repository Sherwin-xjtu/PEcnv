import logging

import numpy as np
from pegeno import tabio

from . import hyperparameters


def do_import_picard(fname, too_many_no_coverage=100):
    garr = tabio.read(fname, "picardhs")
    garr["gene"] = garr["gene"].apply(unpipe_name)

    coverages = garr["ratio"].copy()
    no_cvg_idx = (coverages == 0)
    if no_cvg_idx.sum() > too_many_no_coverage:
        logging.warning("WARNING: Sample %s has >%d bins with no coverage",
                        garr.sample_id, too_many_no_coverage)
    coverages[no_cvg_idx] = 2 ** hyperparameters.NULL_LOG2_coverInfo
    garr["log2"] = np.log2(coverages)
    return garr


def unpipe_name(name):
    if '|' not in name:
        return name
    gene_names = set(name.split('|'))
    if len(gene_names) == 1:
        return gene_names.pop()
    cleaned_names = gene_names.difference(hyperparameters.IGNORE_GENE_NAMES)
    if cleaned_names:
        gene_names = cleaned_names
    new_name = sorted(gene_names, key=len, reverse=True)[0]
    if len(gene_names) > 1:
        logging.warning("WARNING: Ambiguous gene name %r; using %r",
                        name, new_name)
    return new_name


def do_import_theta(segarr, theta_results_fname, ploidy=2):
    theta = parse_theta_results(theta_results_fname)

    segarr = segarr.autosomes()
    for copies in theta['C']:
        if len(copies) != len(segarr):
            copies = copies[:len(segarr)]

        mask_drop = np.array([c is None for c in copies], dtype='bool')
        segarr = segarr[~mask_drop].copy()
        ok_copies = np.asfarray([c for c in copies if c is not None])

        segarr["cn"] = ok_copies.astype('int')
        ok_copies[ok_copies == 0] = 0.5
        segarr["log2"] = np.log2(ok_copies / ploidy)
        segarr.sort_columns()
        yield segarr


def parse_theta_results(fname):
    with open(fname) as handle:
        header = next(handle).rstrip().split('\t')
        body = next(handle).rstrip().split('\t')
        assert len(body) == len(header) == 4

        nll = float(body[0])

        mu = body[1].split(',')
        mu_normal = float(mu[0])
        mu_tumors = list(map(float, mu[1:]))

        copies = body[2].split(':')
        if len(mu_tumors) == 1:

            copies = [[int(c) if c.isdigit() else None
                       for c in copies]]
        else:

            copies = [[int(c) if c.isdigit() else None
                       for c in subcop]
                      for subcop in zip(*[c.split(',') for c in copies])]

        probs = body[3].split(',')
        if len(mu_tumors) == 1:

            probs = [float(p) if not p.isalpha() else None
                     for p in probs]
        else:
            probs = [[float(p) if not p.isalpha() else None
                      for p in subprob]
                     for subprob in zip(*[p.split(',') for p in probs])]
    return {"NLL": nll,
            "mu_normal": mu_normal,
            "mu_tumors": mu_tumors,
            "C": copies,
            "p*": probs}
