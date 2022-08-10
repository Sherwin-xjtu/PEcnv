
import pandas as pd
from Bio.File import as_handle


def read_dict(infile):
    colnames = ["chromosome", "start", "end", # "file", "md5"
               ]
    with as_handle(infile, 'rU') as handle:
        rows = _parse_lines(handle)
        return pd.DataFrame.from_records(rows, columns=colnames)


def _parse_lines(lines):
    for line in lines:
        if line.startswith("@SQ"):
            _sq, sn, ln, _ur, _m5 = line.split("\t")
            if sn.startswith("SN:") and ln.startswith("LN:"):
                chrom = sn[3:]
                length = int(ln[3:])
                yield (chrom, 0, length)
            else:
                raise ValueError("Bad line: %r" % line)
        elif line.startswith("@HD"):
            pass
        else:
            # NB: not sure if there's any other valid row type
            # Assume it's some garbage at the end of the file & bail
            # (or an interval list with SAM header, but we've specified dict and
            #  not interval, so still return what we've parsed up to this point)
            break
