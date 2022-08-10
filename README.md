# PEcnv

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/Sherwin-xjtu/PEcnv/edit/master/README.md)

PEcnv: Accurate and Efficient Detection of Copy-number Variations of Various Lengths.

A novel tool to detecting various sizes of CNVs based on WGS, WES, and panel sequencing! 

## Table of Contents

- [Features](#features)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Example Readmes](#example-readmes)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Features

1. PEcnv is a novel approach to detect copy number variations that vary in size and copy number.
2. PEcnv enables accurate detection of copy number variants with short-to-medium length, which are hard to identify with existing approaches but have been emphasized recently in cancer research and treatment development.
3. PEcnv is among the first approaches to incorporate the genomic bases’ features around a target base to correct the bias and noise on the read depth. This solves the lack of training data for clinical panel sequencing data. 
4. PEcnv applies to panel sequencing data and also works for whole genome sequencing and whole-exome sequencing data. 

## Background

Copy number variation (CNV) is a class of key biomarkers in many complex traits and diseases. Detecting CNV from sequencing data is a substantial bioinformatics problem and a standard requirement in clinical practice. Although many proposed CNV detection approaches exist, the core statistical model at their foundation is weakened by two critical computational issues: 1) identifying the optimal setting on the sliding window and 2) correcting for bias and noise. We designed a statistical process model to overcome these limitations by calculating regional read depths via an exponentially weighted moving average strategy. A one-run detection of CNVs of various lengths is then achieved by a dynamic sliding window, whose size is self-adopted according to the weighted averages. We also designed a novel bias/noise reduction model, accompanied by the moving average, which can handle complicated patterns and extend training data. This model, called PEcnv, accurately detects CNVs ranging from kb-scale to chromosome-arm level. The model performance was validated with simulation samples and real samples. Comparative analysis showed that PEcnv outperforms current popular approaches. Notably, PEcnv provided considerable advantages in detecting small CNVs (1 kb–1 Mb) in panel sequencing data. Thus, PEcnv fills the gap left by existing methods focusing on large CNVs. PEcnv may have broad applications in clinical testing where panel sequencing is the dominant strategy.

## Install
Uncompress the installation zip:

    $ cd /my/install/dir/
    $ unzip /path/to/PEcnv.zip
    
This project uses [node](http://nodejs.org) and [npm](https://npmjs.com). Go check them out if you don't have them locally installed.

```sh
$ npm install --global standard-readme-spec
```

## Usage

This is only a documentation package. You can print out [spec.md](spec.md) to your console:

```sh
$ standard-readme-spec
# Prints out the standard-readme spec
```

## Example Readmes

To see how the specification has been applied, see the [example-readmes](example-readmes/).

## Related Efforts

- [Art of Readme](https://github.com/noffle/art-of-readme) - 💌 Learn the art of writing quality READMEs.
- [open-source-template](https://github.com/davidbgk/open-source-template/) - A README template to encourage open-source contributions.

## Maintainers

[@Sherwin](https://github.com/Sherwin-xjtu).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/RichardLitt/standard-readme/issues/new) or submit PRs.

Standard Readme follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.


## License

[MIT](LICENSE) © Sherwin


