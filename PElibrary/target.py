import logging

from pegeno import tabio

from . import offTarget


def do_target(bait_arr, annotate=None, do_short_names=False, do_split=False,
              avg_size=200 / .75):
    tgt_arr = bait_arr.copy()

    tgt_arr = tgt_arr[tgt_arr.start != tgt_arr.end]
    if do_split:
        tgt_arr = tgt_arr.subdivide(avg_size, 0)
    if annotate:
        annotation = tabio.read_auto(annotate)
        offTarget.compare_chrom_names(tgt_arr, annotation)
        tgt_arr['gene'] = annotation.into_ranges(tgt_arr, 'gene', '-')
    if do_short_names:
        tgt_arr['gene'] = list(shorten_labels(tgt_arr['gene']))
    return tgt_arr


def shorten_labels(gene_labels):
    longest_name_len = 0
    curr_names = set()
    curr_gene_count = 0

    for label in gene_labels:
        next_names = set(label.rstrip().split(','))
        assert len(next_names)
        overlap = curr_names.intersection(next_names)
        if overlap:

            curr_names = filter_names(overlap)
            curr_gene_count += 1
        else:

            for _i in range(curr_gene_count):
                out_name = shortest_name(curr_names)
                yield out_name
                longest_name_len = max(longest_name_len, len(out_name))

            curr_gene_count = 1
            curr_names = next_names

    for _i in range(curr_gene_count):
        out_name = shortest_name(curr_names)
        yield out_name
        longest_name_len = max(longest_name_len, len(out_name))


def filter_names(names, exclude=('mRNA',)):
    if len(names) > 1:
        ok_names = set(n for n in names
                       if not any(n.startswith(ex) for ex in exclude))
        if ok_names:
            return ok_names

    return names


def shortest_name(names):
    name = min(filter_names(names), key=len)
    if len(name) > 2 and '|' in name[1:-1]:
        name = name.split('|')[-1]
    return name
