import collections
import logging
import warnings

warnings.simplefilter('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import scipy.special
import pomegranate as pom

from ..cnv import CopyNumArray as CNA
from ..measures import biweight_midvariance
from ..filteringSegm import squash_by_groups


def segment_hmm(cnarr, method, window=None, variants=None, processes=1):
    orig_log2 = cnarr['log2'].values.copy()
    cnarr['log2'] = cnarr.smooth_log2()

    logging.info("Building model from observations")
    model = hmm_get_model(cnarr, method, processes)

    logging.info("Predicting states from model")
    observations = as_observation_matrix(cnarr)
    states = np.concatenate([np.array(model.predict(obs, algorithm='map'))
                             for obs in observations])

    logging.info("Done, now finalizing")
    logging.debug("Model states: %s", model.states)
    logging.debug("Predicted states: %s", states[:100])
    logging.debug(str(collections.Counter(states)))
    logging.debug("Observations: %s", observations[0][:100])
    logging.debug("Edges: %s", model.edges)

    cnarr['log2'] = orig_log2
    cnarr['probes'] = 1
    segarr = squash_by_groups(cnarr,
                              pd.Series(states, index=cnarr.data.index),
                              by_arm=True)
    if not (segarr.start < segarr.end).all():
        bad_segs = segarr[segarr.start >= segarr.end]
        logging.warning("Bad segments:\n%s", bad_segs.data)
    return segarr


def hmm_get_model(cnarr, method, processes):
    assert method in ('hmm-tumor', 'hmm-germline', 'hmm')
    observations = as_observation_matrix(cnarr.autosomes())

    stdev = biweight_midvariance(np.concatenate(observations), initial=0)
    if method == 'hmm-germline':
        state_names = ["loss", "neutral", "gain"]
        distributions = [
            pom.NormalDistribution(-1.0, stdev, frozen=True),
            pom.NormalDistribution(0.0, stdev, frozen=True),
            pom.NormalDistribution(0.585, stdev, frozen=True),
        ]
    elif method == 'hmm-tumor':
        state_names = ["del", "loss", "neutral", "gain", "amp"]
        distributions = [
            pom.NormalDistribution(-2.0, stdev, frozen=False),
            pom.NormalDistribution(-0.5, stdev, frozen=False),
            pom.NormalDistribution(0.0, stdev, frozen=True),
            pom.NormalDistribution(0.3, stdev, frozen=False),
            pom.NormalDistribution(1.0, stdev, frozen=False),
        ]
    else:
        state_names = ["loss", "neutral", "gain"]
        distributions = [
            pom.NormalDistribution(-1.0, stdev, frozen=False),
            pom.NormalDistribution(0.0, stdev, frozen=False),
            pom.NormalDistribution(0.585, stdev, frozen=False),
        ]

    n_states = len(distributions)
    binom_coefs = scipy.special.binom(n_states - 1, range(n_states))
    start_probabilities = binom_coefs / binom_coefs.sum()

    transition_matrix = (np.identity(n_states) * 100
                         + np.ones((n_states, n_states)) / n_states)

    model = pom.HiddenMarkovModel.from_matrix(transition_matrix, distributions,
                                              start_probabilities, state_names=state_names, name=method)

    model.fit(sequences=observations,
              weights=[len(obs) for obs in observations],
              distribution_inertia=.8,
              edge_inertia=0.1,
              pseudocount=5,
              use_pseudocount=True,
              max_iterations=100000,
              n_jobs=processes,
              verbose=False)
    return model


def as_observation_matrix(cnarr, variants=None):
    observations = [arm.log2.values
                    for _c, arm in cnarr.by_arm()]
    return observations


def variants_in_segment(varr, segment, min_variants=50):
    if len(varr) > min_variants:
        observations = varr.mirrored_baf(above_half=True)
        state_names = ["neutral", "alt"]
        distributions = [
            pom.NormalDistribution(0.5, .1, frozen=True),
            pom.NormalDistribution(0.67, .1, frozen=True),
        ]
        n_states = len(distributions)
        start_probabilities = [.95, .05]

        transition_matrix = (np.identity(n_states) * 100
                             + np.ones((n_states, n_states)) / n_states)
        model = pom.HiddenMarkovModel.from_matrix(transition_matrix, distributions,
                                                  start_probabilities, state_names=state_names, name="loh")

        model.fit(sequences=[observations],
                  edge_inertia=0.1,
                  lr_decay=.75,
                  pseudocount=5,
                  use_pseudocount=True,
                  max_iterations=100000,

                  verbose=False)
        states = np.array(model.predict(observations, algorithm='map'))

        logging.info("Done, now finalizing")
        logging.debug("Model states: %s", model.states)
        logging.debug("Predicted states: %s", states[:100])
        logging.debug(str(collections.Counter(states)))

        logging.debug("Edges: %s", model.edges)

        fake_cnarr = CNA(varr.add_columns(weight=1, log2=0, gene='.').data)
        results = squash_by_groups(fake_cnarr,
                                   varr.as_series(states),
                                   by_arm=False)
        assert (results.start < results.end).all()

    else:
        results = None

    if results is not None and len(results) > 1:
        logging.info("Segment %s:%d-%d on allele freqs for %d additional breakpoints",
                     segment.chromosome, segment.start, segment.end,
                     len(results) - 1)

        mid_breakpoints = (results.start.values[1:] + results.end.values[:-1]) // 2
        starts = np.concatenate([[segment.start], mid_breakpoints])
        ends = np.concatenate([mid_breakpoints, [segment.end]])
        dframe = pd.DataFrame({
            'chromosome': segment.chromosome,
            'start': starts,
            'end': ends,

            'gene': segment.gene,
            'log2': segment.log2,
            'probes': results['probes'],

        })
        bad_segs_idx = (dframe.start >= dframe.end)
        if bad_segs_idx.any():
            raise RuntimeError("Improper post-processing of segment {} -- "
                               "{} bins start >= end:\n{}\n"
                               .format(segment, bad_segs_idx.sum(),
                                       dframe[bad_segs_idx]))

    else:
        dframe = pd.DataFrame({
            'chromosome': segment.chromosome,
            'start': segment.start,
            'end': segment.end,
            'gene': segment.gene,
            'log2': segment.log2,
            'probes': segment.probes,

        }, index=[0])

    return dframe
