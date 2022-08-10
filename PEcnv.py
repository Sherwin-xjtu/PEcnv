#!/usr/bin/env python3
"""Command-line interface for PEcnv, the Copy Number Variation toolkit."""

import logging
from PElibrary import commands

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = commands.parse_args()
    args.func(args)
