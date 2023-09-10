import setuptools
from Cython.Build import cythonize
import numpy as np


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hampel",
    version="0.0.5",
    author="MTrofficus",
    author_email="miguel.otero.pedrido.1993@gmail.com",
    description="Python implementation of the Hampel Filter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichaelisTrofficus/hampel_filter",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=cythonize("src/c_hampel.pyx"),
    install_requires=[
        "numpy",
        "cython"
    ],
    extras_requires={
        "dev": [
            "pytest>=6.0.2"
        ]
    },
    python_requires='>=3.8',
    include_dirs=[np.get_include()],
)
