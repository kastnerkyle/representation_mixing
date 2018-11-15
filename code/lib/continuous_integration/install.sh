#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip
if [[ -d $HOME/download ]]; then
    echo 'download'
    ls $HOME/download
fi

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Use the miniconda installer for faster download / install of conda
# itself
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo
if [[ ! -f miniconda.sh ]]
    then
    wget https://repo.continuum.io/miniconda/Miniconda2-4.3.11-Linux-x86_64.sh \
        -O miniconda.sh
    fi
chmod +x miniconda.sh && ./miniconda.sh -b
cd ..
echo $(ls /home/travis/m*)
export PATH=/home/travis/miniconda2/bin:$PATH
conda update --yes conda
popd

conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
    numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION tensorflow=$TF_VERSION
source activate testenv

if [[ "$INSTALL_MKL" == "true" ]]; then
    # Make sure that MKL is used
    conda install --yes mkl
else
    # Make sure that MKL is not used
    conda remove --yes --features mkl || echo "MKL not installed"
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

# Build scikit-learn in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import tensorflow as tf; print('tensorflow %s' % tf.__version__)"
python setup.py build_ext --inplace
