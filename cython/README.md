Welcome to the the Cython-BART project.

# Cython-BART

This directory contains the Cython implementation of PyMC-BART.

## Getting started

To get started, create and activate a new conda environment with the dependencies in `environment.yml`

```bash
conda env create -n cbart -f environment.yml
conda activate cbart
```

As this Cython implementation is not yet an installable package, the Cython code must be compiled before running any tests. To compile the Cython code, run the following command

```bash
python setup.py build_ext --inplace
```

This will create a `build` directory with the compiled Cython code, and will also place the shared object `.so` files in the `pymc_bart` directory.

You can now run the tests in the `tests` directory

```bash
python tests/test_tree.py
```

## Project structure

```bash
~/cython
├── README.md
├── pymc_bart               // PyMC-BART Cython implementation   
│   ├── __init__.py
│   ├── _tree.pxd
│   ├── _tree.pyx
│   ├── _typedefs.pxd
│   ├── bart.py
│   ├── leaf_ops.pyx
│   ├── pgbart.py
│   ├── split_rules.py
│   ├── tree.py
│   └── utils.py
├── setup.py                // Setup file for compiling Cython-BART
└── tests                   // Unit and performance of Cython code
    ├── test_bart.py
    └── test_tree.py
```