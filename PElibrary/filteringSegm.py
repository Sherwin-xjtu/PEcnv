import functools
import logging

import numpy as np
import pandas as pd

from .measures import weighted_median


def require_column(*colnames):
    if len(colnames) == 1:
        msg = "'{}' filter requires column '{}'"
    else:
        msg = "'{}' filter requires columns " + \
              ", ".join(["'{}'"] * len(colnames))

    def wrap(func):
        @functools.wraps(func)
        def wrapped_f(segarr):
            filtname = func.__name__
            if any(c not in segarr for c in colnames):
                raise ValueError(msg.format(filtname, *colnames))
            result = func(segarr)
            logging.info("Filtered by '%s' from %d to %d rows",
                         filtname, len(segarr), len(result))
            return result

        return wrapped_f

    return wrap


def squash_by_groups(cnarr, levels, by_arm=False):
    change_levels = enumerate_changes(levels)
    assert (change_levels.index == levels.index).all()
    assert cnarr.data.index.is_unique
    assert levels.index.is_unique
    assert change_levels.index.is_unique
    if by_arm:

        arm_levels = []
        for i, (_chrom, cnarm) in enumerate(cnarr.by_arm()):
            arm_levels.append(np.repeat(i, len(cnarm)))
        change_levels += np.concatenate(arm_levels)
    else:

        chrom_names = cnarr['chromosome'].unique()
        chrom_col = (cnarr['chromosome']
                     .replace(chrom_names, np.arange(len(chrom_names))))
        change_levels += chrom_col
    data = cnarr.data.assign(_group=change_levels)
    groupkey = ['_group']
    if 'cn1' in cnarr:
        data['_g1'] = enumerate_changes(cnarr['cn1'])
        data['_g2'] = enumerate_changes(cnarr['cn2'])
        groupkey.extend(['_g1', '_g2'])
    data = (data.groupby(groupkey, as_index=False, group_keys=False, sort=False)
            .apply(squash_region)
            .reset_index(drop=True))
    return cnarr.as_dataframe(data)


def enumerate_changes(levels):
    return levels.diff().fillna(0).abs().cumsum().astype(int)


def squash_region(cnarr):
    assert 'weight' in cnarr
    out = {'chromosome': [cnarr['chromosome'].iat[0]],
           'start': cnarr['start'].iat[0],
           'end': cnarr['end'].iat[-1],
           }
    region_weight = cnarr['weight'].sum()
    if region_weight > 0:
        out['log2'] = np.average(cnarr['log2'], weights=cnarr['weight'])
    else:
        out['log2'] = np.mean(cnarr['log2'])
    out['gene'] = ','.join(cnarr['gene'].drop_duplicates())
    out['probes'] = cnarr['probes'].sum() if 'probes' in cnarr else len(cnarr)
    out['weight'] = region_weight
    if 'depth' in cnarr:
        if region_weight > 0:
            out['depth'] = np.average(cnarr['depth'], weights=cnarr['weight'])
        else:
            out['depth'] = np.mean(cnarr['depth'])
    if 'baf' in cnarr:
        if region_weight > 0:
            out['baf'] = np.average(cnarr['baf'], weights=cnarr['weight'])
        else:
            out['baf'] = np.mean(cnarr['baf'])
    if 'cn' in cnarr:
        if region_weight > 0:
            out['cn'] = weighted_median(cnarr['cn'], cnarr['weight'])
        else:
            out['cn'] = np.median(cnarr['cn'])
        if 'cn1' in cnarr:
            if region_weight > 0:
                out['cn1'] = weighted_median(cnarr['cn1'], cnarr['weight'])
            else:
                out['cn1'] = np.median(cnarr['cn1'])
            out['cn2'] = out['cn'] - out['cn1']
    if 'p_bintest' in cnarr:
        out['p_bintest'] = cnarr['p_bintest'].max()
    return pd.DataFrame(out)


@require_column('cn')
def ampdel(segarr):
    levels = np.zeros(len(segarr))
    levels[segarr['cn'] == 0] = -1
    levels[segarr['cn'] >= 5] = 1

    cnarr = squash_by_groups(segarr, pd.Series(levels))
    return cnarr[(cnarr['cn'] == 0) | (cnarr['cn'] >= 5)]


@require_column('depth')
def bic(segarr):
    return NotImplemented


@require_column('ci_lo', 'ci_hi')
def ci(segarr):
    levels = np.zeros(len(segarr))

    levels[segarr['ci_lo'].values > 0] = 1
    levels[segarr['ci_hi'].values < 0] = -1
    return squash_by_groups(segarr, pd.Series(levels))


@require_column('cn')
def cn(segarr):
    return squash_by_groups(segarr, segarr['cn'])


@require_column('sem')
def sem(segarr, zscore=1.96):
    margin = segarr['sem'] * zscore
    levels = np.zeros(len(segarr))
    levels[segarr['log2'] - margin > 0] = 1
    levels[segarr['log2'] + margin < 0] = -1
    return squash_by_groups(segarr, pd.Series(levels))
