# seq-experiment

## Overview

seq-experiment is a library for the analysis of high-throughput sequencing data. It is designed to be used
following the initial sequence processing step using your pipeline of choice (Mothur, Qiime, etc.) It is
designed to provide some of the functionality found in the R package *phyloseq* in a Python environment. This
package has overlaps, and sometimes pulls from the existing Python packages [PyCogent](http://pycogent.org/) 
and [scikit-bio](http://scikit-bio.org/) in a lighter-weight format, and without the limitations on
OS support that these two packages have (i.e. Windows support).

The seq-experiment based analysis revolves around a `SeqExp` object that is functionally analogous to the
phyloseq `phyloseq` object that stores abundance data, classificaiton data, and sample metadata. In 
seq-experiment these three different data tables are stored in pandas DataFrame like objects that can be
manipulated using the standard pandas API. Additonally, many of the basic functions that are performed on
the combined data tables are implemented as methods on the encapsulating SeqExp object using an API similar
to the pandas API.

## Installation

Currently the way to install seq-exp is to clone/download this repository and copy it into your working
directory.

## Requirements

This package requires the following packages:

* Scipy
* Numpy
* Pandas
* Matplotlib

An easy way to meet these dependency requirements is to install the 
[Anaconda](https://www.continuum.io/downloads) Python distribution.
 