#!/bin/bash


# install package locally
pip install -e .
# or:
# python setup.py develop


# install Twine & Wheel packages
pip install --upgrade Twine Wheel


# register package on PyPI
python setup.py sdist bdist_wheel --universal
python setup.py register
twine upload dist/*
# or:
# python setup.py sdist bdist_wheel upload
