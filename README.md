# ASAPy

ASAPy is an analyzing tool for algorithm selection (AS) written in Python 3.
Given a matrix of performance data and a matrix of instance features,
ASAPy automatically generates a thorough web-based report with many diagrams to provide insights in the data.
Hence, ASAPy addresses two kinds of users:

* Users of algorithm selection to get insights in their own data, to verify the correctness of the data and to understand how algorithm selection handles the data
* Developers of algorithm selection tools to get better insights on what kind of data they operate

## License

This program is free software: you can redistribute it and/or modify it under the terms of the MIT license (please see the LICENSE file).
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
You should have received a copy of the MIT license along with this program (see LICENSE file). If not, see https://opensource.org/licenses/MIT.

## Installation

### Requirements

ASAPy runs with '''Python 3.5'''.
Many of its dependencies can be fulfilled by using [Anaconda 3.4](https://www.continuum.io/).  

To install all requirements, please run:

`pip install -r requirements.txt`

If you do not use Anaconda or a virtual environment, you may want to add the option `--user` to install it in your user space.

## USAGE

### Input Formats 

ASAPy reads two input formats: CSV and [ASlib](www.aslib.net).
The CSV format is easier for new users but has some limitations to express all kind of input data.
The ASlib format has a higher expressiveness -- please see [www.aslib.net](www.aslib.net) for all details on this input format.

For the CSV format, simply two files are required.
One file with the performance data of each algorithm on each instance (each row an instance, and each column an algorithm).
And another file with the instance features for each instance (each row an instance and each column an feature).
All other meta-data (such as runtime cutoff) has to be specified by command line options (see `python scripts/asapy --help`).

### CSV Interface

The performance matrix has to be passed by the option `--performance_csv [file name]` and the feature matrix by `--feature_csv [file name]`.
Furthermore, you can specify the objective (running time or solution quality; `--objective`) , the running time cutoff (`runtime_cutoff`) and to maximize or minimize the objective (`--maximize`).

For example, a call could look like

`python scripts/asapy --performance_csv [file name] --feature_csv [file name]`

### ASlib Interface

To use the full expressiveness of algorithm selection data, you can use the ASlib format by providing a folder with all files of an ASLib szenario.

`python scripts/asapy --scenario [folder]`

### Further Options

By default ASAPy writes into the current directory. But you can also redirect the output with the option `--output`. Please note that ASAPy generates a lot of plots. Depending on the number of algorithms in your data, ASAPy can generate more than 100MB.

If you want to disable certain plots (for example, the scatter plots needs a lot of time to be generated), 
you can use a json-formatted configuration file. To get a template for the configuration file, please use the option `--print_config`.

## Reference

[AIJ Journal Article on ASlib](https://arxiv.org/abs/1506.02465)

```
@ARTICLE{bischl-aij16a,
  author    = {B. Bischl and P. Kerschke and L. Kotthoff and M. Lindauer and Y. Malitsky and A. Frech\'ette  and H. Hoos and F. Hutter and K. Leyton-Brown and K. Tierney and J. Vanschoren},
  title     = {ASlib: A Benchmark Library for Algorithm Selection},
  journal   = {Artificial Intelligence Journal (AIJ)},
  volume = {237},
  year      = {2016},
  pages = {41-58}
}
```

## Contact

Marius Lindauer: lindauer@cs.uni-freiburg.de


  




 
