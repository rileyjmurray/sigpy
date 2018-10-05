from sigpy.polys.polynomials import Polynomial
from sigpy.sage import sage_primal, sage_dual, sage_feasibility


# Unconstrained polynomial optimization is easy.

def sage_poly_dual(p, level=0):
    sr, cons = p.sig_rep
    # If p.c contains no cvxpy Expressions, then
    # cons should be a list of length zero.
    return sage_dual(sr, level, additional_cons=cons)


def sage_poly_primal(p, level=0):
    sr, cons = p.sig_rep
    # If p.c contains no cvxpy Expressions, then
    # cons should be a list of length zero.
    return sage_primal(sr, level, additional_cons=cons)


def sage_poly_feasibility(p):
    sr, cons = p.sig_rep
    # If p.c contains no cvxpy Expressions, then
    # cons should be a list of length zero.
    return sage_feasibility(sr, additional_cons=cons)


# For constrained polynomial optimization, you need to
# the the Signomial represenative of the Lagrangian
# (when appropriately formed). Forming the Lagrangian is
# a bit of a pain, so we aren't implementing that yet.
