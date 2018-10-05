# Sigpy is for signomial and polynomial optimization

For Signomials, it implements the SAGE hierarchy as described by Chandrasekaran and Shah: https://arxiv.org/abs/1409.7640.

For Polynomials, it uses the idea of a "signomial representative" to produce global lower bounds as described in my recent paper: https://arxiv.org/abs/1810.01614. The result is a theoretically and practically efficient method for computing lower bounds on polynomials, and is especially well-suited to sparse polynomials or polynomials of high degree. The bounds returned are always at least as strong as those computed by "SONC" as in https://arxiv.org/abs/1808.08431, and can be stronger. Tests and documentation for this sub-package of sigpy are on the way!

## To install

0. Download this repository. If needed, change your directory so that you are in the same directory as sigpy's ``setup.py`` file. Activate the Python virtual environment of your choice. (Please do not install this directly to your system's Python distribution!)
1. Run ``python setup.py install``, so as to install sigpy to your current Python environment.
2. Run ``python -c "import sigpy; print(sigpy.__version__)"`` to verify that sigpy installed correctly.
3. [Optional] Run ``python tests/test_signomials.py`` to verify behavior of Signomial objects, and ``python tests/test_sage.py`` to verify behavior of CVXPY implementations of SAGE relaxations.

## Examples

A couple examples are given in the second set of slides under "Newton Polytopes and Relative Entropy Optimization" [at my website](http://rileyjmurray.com/research).

Soon we will have a dedicated collection of examples hosted on this repository.
