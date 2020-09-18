import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hampel",
    version="0.0.2",
    author="MTrofficus",
    author_email="miguel.otero.pedrido.1993@gmail.com",
    description="Python implementation of the Hampel Filter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    py_modules=["hampel"],
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.19.2",
        "pandas>=1.1.2"
    ],
    extras_requires={
        "dev": [
            "pytest>=6.0.2"
        ]
    },
    python_requires='>=3.7',
)
