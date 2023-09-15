from os import path
from setuptools import find_packages, setup, Extension

from setuptools import dist

dist.Distribution().fetch_build_eggs(["numpy>=1.17.3"])

try:
    import numpy as np
except ImportError:
    exit("Please install numpy>=1.17.3 first.")

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

__version__ = "1.0.1"

here = path.abspath(path.dirname(__file__))

# Get long description from README.md
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    install_requires = [line.strip() for line in f.read().split("\n")]

cmdclass = {}

ext = ".pyx" if USE_CYTHON else ".c"

extensions = [
    Extension(
        "hampel.extension.hampel",
        ["src/hampel/extension/hampel" + ext],
        include_dirs=[np.get_include()],
    )
]

if USE_CYTHON:
    extensions = cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
        }
    )
    cmdclass.update({"build_ext": build_ext})

setup(
    name="hampel",
    author="MTrofficus",
    author_email="miguel.otero.pedrido.1993@gmail.com",
    description="Python implementation of the Hampel Filter",
    version=__version__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichaelisTrofficus/hampel_filter",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src", exclude=["tests*"]),
    package_dir={"": "src"},
    python_requires='>=3.9',
    include_package_data=True,
    ext_modules=extensions,
    cmdclass=cmdclass,
    install_requires=install_requires,
)
