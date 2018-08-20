# Sigpy is for signomial optimization
It implements the SAGE hierarchy.

To install...
1. Activate the virtual environment of your choice (something conda-based, if you're like me).
2. Make sure you're in the directory with sigpy's ``setup.py`` file.
3. Run ``python setup.py install``, so as to install sigpy to your current Python environment.
4. Run ``python -c "import sigpy; print(sigpy.__version__)"`` to verify that sigpy installed correctly.
5. [Optional] Run ``python tests/test_signomials.py`` to verify behavior of Signomial objects, and ``python tests/test_sage.py`` to verify behavior of CVXPY implementations of SAGE relaxations.
