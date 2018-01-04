# seq-experiment

Copyright &#169; 2018, Richard Campen. All rights reserved.

See LICENSE.txt for full license conditions.

---

### Overview

seq-experiment is a package designed to facilitate analysis of sequencing data. It does this through the 
`SeqExp` object which contains feature abundance data (sequences, OTUs, etc) along with related 
classifications, metadata, and raw sequences. The `SeqExp` object is built on top of the pandas library,
with each of these data types stored in their own pandas DataFrames as attributes on the `SeqExp` object.
The `SeqExp` object then provides methods to easily manipulate the data within each DataFrame simultaneously
and intelligently. For example, grouping sequence abundances based on classifications or metadata. The 
`SeqExp` object also provides some basic plotting of sequence abundance data, incorporating the 
classifications and metadata. In this way the `SeqExp` object provides similar functionality to the 
`phyloseq` object in the R package `phyloseq`, within a python environment.

---

### Installation

To install the most up to date version you should download/clone this repository and create a binary 
distribution using `python setup.py bdist_wheel` that will create a .whl file in the dist folder. 
You can then install seq-experiment with pip from the .whl file using `pip install <wheel_file_name>`. 
The advantage of this method over just running `python setup.py install` is that you can easily 
remove or update the package via pip.


---

### Requirements

This package requires the following packages to run:

* Numpy
* Pandas
* Matplotlib

An easy way to meet these dependency requirements is to install the 
[Anaconda](https://www.continuum.io/downloads) Python distribution.
 