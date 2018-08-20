# Sigpy is for signomial optimization
It implements the SAGE hierarchy. 

To install...

0. Download this repository. If needed, change your directory so that you are in the same directory as sigpy's ``setup.py`` file. Activate the Python virtual environment of your choice. (Please do not install this directly to your system's Python distribution!)
1. Run ``python setup.py install``, so as to install sigpy to your current Python environment.
2. Run ``python -c "import sigpy; print(sigpy.__version__)"`` to verify that sigpy installed correctly.
3. [Optional] Run ``python tests/test_signomials.py`` to verify behavior of Signomial objects, and ``python tests/test_sage.py`` to verify behavior of CVXPY implementations of SAGE relaxations.

Examples are given in the second set of slides under "Newton Polytopes and Relative Entropy Optimization" [at my website](http://rileyjmurray.com/research).
