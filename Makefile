# Makefile for building Cython extensions and running tests

# Variables
CYTHON_MODULE = src/hampel/c_hampel
TESTS_DIR = tests

.PHONY: all build test clean

# Default target: Build Cython extensions and run tests
all: build test

# Build Cython extensions
build:
	python setup.py build_ext --inplace

# Run tests (build extensions if not built)
test: build
	pytest $(TESTS_DIR)

# Clean build artifacts
clean:
	rm -rf build $(CYTHON_MODULE).c $(CYTHON_MODULE)*.so