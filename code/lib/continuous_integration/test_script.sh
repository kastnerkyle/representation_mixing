#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# still doesn't fix anything...
export TF_CPP_MIN_LOG_LEVEL=3
set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import tensorflow as tf; print('tensorflow %s' % tf.__version__)"

# Do not use "make test" or "make test-coverage" as they enable verbose mode
# which renders travis output too slow to display in a browser.
if [[ "$COVERAGE" == "true" ]]; then
    nosetests -s --with-coverage tfbldr
else
    nosetests -s tfbldr
fi
