from setuptools import setup

import numpy as np

from Cython.Build import cythonize

setup(
    name="pymc_bart",
    packages=["pymc_bart"],
    ext_modules=cythonize(
        [
            "pymc_bart/leaf_ops.pyx", 
            "pymc_bart/_tree.pyx"
        ], 
        compiler_directives={"language_level": "3"},
        build_dir="build",
        annotate=True,
        ),
    include_dirs=[np.get_include()]
    )