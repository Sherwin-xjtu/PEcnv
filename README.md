# PEcnv

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/Sherwin-xjtu/PEcnv/edit/master/README.md)

PEcnv: Accurate and Efficient Detection of Copy-number Variations of Various Lengths.

A novel tool to detecting various sizes of CNVs based on WGS, WES, and panel sequencing! 

## Table of Contents

- [Features](#features)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Features



## Background



## Install
Uncompress the installation zip:

    $ cd /my/install/dir/
    $ unzip /path/to/PEcnv.zip
    

## Usage


```sh
$ python PEcnv.py run tumor.bam --normal normal.bam  --targets CNVTarget.bed --fasta reference.fa --annotate refFlat.txt --preprocess dukeExcludeRegions.bed --output-refBaseline refBaseline.tsv --output-dir our_dir
```


## Maintainers

[@Sherwin](https://github.com/Sherwin-xjtu).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/Sherwin-xjtu/PEcnv/issues/new) or submit PRs.

## License

[MIT](LICENSE) © Sherwin


