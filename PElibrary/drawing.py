import collections
import math
import warnings

from Bio.Graphics import BasicChromosome as BC
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from . import hyperparameters, functionsInfo, plots

from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

TELOMERE_LENGTH = 6e6
CHROM_FATNESS = 0.3
PAGE_SIZE = (11.0 * inch, 8.5 * inch)


def create_diagram(cnarr, segarr, threshold, min_probes, outfname, title=None):
    if cnarr and segarr:
        do_both = True
        cnarr_is_seg = False
    else:
        if cnarr:
            cnarr_is_seg = False
        elif segarr:
            cnarr = segarr
            cnarr_is_seg = True
        else:
            raise ValueError("Must provide argument cnarr or segarr, or both. ")
        do_both = False
    gene_labels = _get_gene_labels(cnarr, segarr, cnarr_is_seg, threshold,
                                   min_probes)

    seen_genes = set()

    features = collections.defaultdict(list)
    strand = 1 if do_both else None
    chrom_sizes = plots.chromosome_sizes(cnarr)
    if not cnarr_is_seg:
        cnarr = cnarr.squash_genes()
    for row in cnarr:
        if row.start - 1 >= 0 and row.end <= chrom_sizes[row.chromosome]:
            if row.gene in gene_labels and row.gene not in seen_genes:
                seen_genes.add(row.gene)
                feat_name = row.gene
                if "," in feat_name:
                    feat_name = feat_name.replace(",", ", ")
            else:
                feat_name = None
            features[row.chromosome].append(
                (row.start - 1, row.end, strand, feat_name,
                 colors.Color(*plots.cvg2rgb(row.log2, not cnarr_is_seg))))
    if do_both:

        for chrom, segrows in segarr.by_chromosome():
            for srow in segrows:
                if srow.start - 1 >= 0 and srow.end <= chrom_sizes[chrom]:
                    features[chrom].append(
                        (srow.start - 1, srow.end, -1, None,
                         colors.Color(*plots.cvg2rgb(srow.log2, False))))

    if not outfname:
        outfname = cnarr.sample_id + '-diagram.pdf'
    drawing = build_chrom_diagram(features, chrom_sizes, cnarr.sample_id, title)
    cvs = canvas.Canvas(outfname, pagesize=PAGE_SIZE)
    renderPDF.draw(drawing, cvs, 0, 0)
    cvs.showPage()
    cvs.save()
    return outfname


def _get_gene_labels(cnarr, segarr, cnarr_is_seg, threshold, min_probes):
    if cnarr_is_seg:

        sel = cnarr.data[(cnarr.data.log2.abs() >= threshold) &
                         ~cnarr.data.gene.isin(hyperparameters.IGNORE_GENE_NAMES)]
        rows = sel.itertuples(index=False)
        probes_attr = 'probes'
    elif segarr:

        rows = functionsInfo.gene_metrics_by_segment(cnarr, segarr, threshold)
        probes_attr = 'segment_probes'
    else:

        rows = functionsInfo.gene_metrics_by_gene(cnarr, threshold)
        probes_attr = 'n_bins'
    return [row.gene for row in rows if getattr(row, probes_attr) >= min_probes]


def build_chrom_diagram(features, chr_sizes, sample_id, title=None):
    max_chr_len = max(chr_sizes.values())

    chr_diagram = BC.Organism()
    chr_diagram.page_size = PAGE_SIZE
    chr_diagram.title_size = 18

    for chrom, length in list(chr_sizes.items()):
        chrom_features = features.get(chrom)
        if not chrom_features:
            continue
        body = BC.AnnotatedChromosomeSegment(length, chrom_features)
        body.label_size = 4
        body.scale = length
        body.chr_percent = CHROM_FATNESS

        tel_start = BC.TelomereSegment()
        tel_start.scale = TELOMERE_LENGTH
        tel_start.chr_percent = CHROM_FATNESS
        tel_end = BC.TelomereSegment(inverted=True)
        tel_end.scale = TELOMERE_LENGTH
        tel_end.chr_percent = CHROM_FATNESS

        cur_chromosome = BC.Chromosome(chrom)
        cur_chromosome.title_size = 14

        cur_chromosome.scale_num = max_chr_len + 2 * TELOMERE_LENGTH
        cur_chromosome.add(tel_start)
        cur_chromosome.add(body)
        cur_chromosome.add(tel_end)
        chr_diagram.add(cur_chromosome)

    if not title:
        title = "Sample " + sample_id
    return bc_organism_draw(chr_diagram, title)


def bc_organism_draw(org, title, wrap=12):
    margin_top = 1.25 * inch
    margin_bottom = 0.1 * inch
    margin_side = 0.5 * inch

    width, height = org.page_size
    cur_drawing = BC.Drawing(width, height)

    title_string = BC.String(width / 2, height - margin_top + .5 * inch, title)
    title_string.fontName = 'Helvetica-Bold'
    title_string.fontSize = org.title_size
    title_string.textAnchor = "middle"
    cur_drawing.add(title_string)

    if len(org._sub_components) > 0:
        nrows = math.ceil(len(org._sub_components) / wrap)
        x_pos_change = (width - 2 * margin_side) / wrap
        y_pos_change = (height - margin_top - margin_bottom) / nrows

    cur_x_pos = margin_side
    cur_row = 0
    for i, sub_component in enumerate(org._sub_components):
        if i % wrap == 0 and i != 0:
            cur_row += 1
            cur_x_pos = margin_side

        sub_component.start_x_position = cur_x_pos + 0.05 * x_pos_change
        sub_component.end_x_position = cur_x_pos + 0.95 * x_pos_change
        sub_component.start_y_position = (height - margin_top
                                          - y_pos_change * cur_row)
        sub_component.end_y_position = (margin_bottom
                                        + y_pos_change * (nrows - cur_row - 1))

        sub_component.draw(cur_drawing)

        cur_x_pos += x_pos_change

    cur_drawing.add(BC.Rect(width / 2 - .8 * inch, .5 * inch, 1.6 * inch, .4 * inch,
                            fillColor=colors.white))

    cur_drawing.add(BC.Rect(width / 2 - .7 * inch, .6 * inch, .2 * inch, .2 * inch,
                            fillColor=colors.Color(.8, .2, .2)))
    cur_drawing.add(BC.String(width / 2 - .42 * inch, .65 * inch, "Gain",
                              fontName='Helvetica', fontSize=12))

    cur_drawing.add(BC.Rect(width / 2 + .07 * inch, .6 * inch, .2 * inch, .2 * inch,
                            fillColor=colors.Color(.2, .2, .8)))
    cur_drawing.add(BC.String(width / 2 + .35 * inch, .65 * inch, "Loss",
                              fontName='Helvetica', fontSize=12))

    return cur_drawing


def bc_chromosome_draw_label(self, cur_drawing, label_name):
    x_position = 0.5 * (self.start_x_position + self.end_x_position)

    y_position = self.start_y_position + 0.1 * inch
    label_string = BC.String(x_position, y_position, label_name)
    label_string.fontName = 'Times-BoldItalic'
    label_string.fontSize = self.title_size
    label_string.textAnchor = 'middle'
    cur_drawing.add(label_string)


BC.Chromosome._draw_label = bc_chromosome_draw_label
