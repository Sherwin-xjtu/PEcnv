
import logging
import re

import pandas as pd


def read_gff(infile, tag=r'(Name|gene_id|gene_name|gene)', keep_type=None):

    colnames = ['chromosome', 'source', 'type', 'start', 'end',
                'score', 'strand', 'phase', 'attribute']
    coltypes = ['str', 'str', 'str', 'int', 'int',
                'str', 'str', 'str', 'str']
    dframe = pd.read_csv(infile, sep='\t', comment='#', header=None,
                         na_filter=False, names=colnames,
                         dtype=dict(zip(colnames, coltypes)))
    dframe = (dframe
              .assign(start=dframe.start - 1,
                      score=dframe.score.replace('.', 'nan').astype('float'))
              .sort_values(['chromosome', 'start', 'end'])
              .reset_index(drop=True))
    if keep_type:
        ok_type = (dframe['type'] == keep_type)
        logging.info("Keeping %d '%s' / %d total records",
                     ok_type.sum(), keep_type, len(dframe))
        dframe = dframe[ok_type]
    if len(dframe):
        rx = re.compile(tag + r'[= ]"?(?P<gene>\S+?)"?(;|$)')
        matches = dframe['attribute'].str.extract(rx, expand=True)['gene']
        if len(matches):
            dframe['gene'] = matches
    if 'gene' in dframe.columns:
        dframe['gene'] = dframe['gene'].fillna('-').astype('str')
    else:
        dframe['gene'] = ['-'] * len(dframe)
    return dframe
