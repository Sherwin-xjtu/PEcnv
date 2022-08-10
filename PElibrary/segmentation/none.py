
import pandas as pd

from ..segmetrics import segment_mean

def segment_none(cnarr):
    """Return a single segment for the given region."""
    colnames = ['chromosome', 'start', 'end', 'log2', 'gene', 'probes']
    rows = [(cnarr.chromosome.iat[0],
             cnarr.start.iat[0],
             cnarr.end.iat[-1],
             segment_mean(cnarr),
             '-',
             len(cnarr))]
    table = pd.DataFrame.from_records(rows, columns=colnames)
    segarr = cnarr.as_dataframe(table)
    segarr.sort_columns()
    return segarr
