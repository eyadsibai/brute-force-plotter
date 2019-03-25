#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Brute Force Plotter
-----------------
Command Line Interface

"""

from __future__ import unicode_literals

import logging
import json

import pandas as pd

import click
from plotter import create_plots


logger = logging.getLogger(__name__)


@click.command()
@click.argument("input-file")
@click.argument("dtypes")
@click.argument("output-path")
def main(input_file, dtypes, output_path):
    """Create Plots From data in input"""

    data = pd.read_csv(input_file)

    data_types = json.load(open(dtypes, "r"))
    create_plots(data, data_types, output_path)


if __name__ == "__main__":
    main()
