# Sigpy is for signomial optimization
It implements the SAGE hierarchy. 

To install...

0. Download this repository, and my ``coniclifts`` repository.
1. Activate your virtual environment of choice. (Please do not install this diretly to your system's Python distribution!)
2. Change your directory so that you are in the same directory as coniclifts's ``setup.py`` file.
3. Run ``python setup.py install`` to install ``coniclifts`` to your current Python environment.
4. Run ``python -c "import sigpy; print(coniclifts.__version__)"``, and verify that it prints >= 0.1.
5. Change your directory so that you are in the same directory as sigpy's ``setup.py`` file.
6. Run ``python setup.py install`` to install ``sigpy`` to your current Python environment.
7. Run ``python -c "import sigpy; print(sigpy.__version__)"`` to verify that sigpy can import correctly.
8. Run ``python tests/test_signomials.py`` to verify behavior of Signomial objects, and ``python tests/test_sage.py`` to verify behavior of the ``coniclifts`` implementations of SAGE relaxations.

Examples are given in the second set of slides under "Newton Polytopes and Relative Entropy Optimization" [at my website](http://rileyjmurray.com/research).
