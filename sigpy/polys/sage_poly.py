from sigpy.polys.polynomials import Polynomial
from sigpy.sage import sage_primal, sage_dual, sage_feasibility, hierarchy_e_k, relative_c_sage
import cvxpy
import numpy as np
from itertools import combinations_with_replacement


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


# The function below is COMPLETELY UNTESTED.


def constrained_sage_poly_primal(f, gs, p=0, q=1):
    """
    Compute the primal f_{SAGE}^{(p, q)} bound for

        inf f(x) : g(x) >= 0 for g \in gs.

    :param f: a Polynomial.
    :param gs: a list of Polynomials.
    :param p: a nonnegative integer.
    :param q: a positive integer.
    :return: a CVXPY Problem that defines the primal formulation for f_{SAGE}^{(p, q)}.
    """
    lagrangian, dualized_polynomials = make_poly_lagrangian(f, gs, p=p, q=q)
    constrs = []
    for s_h, _ in dualized_polynomials:
        s_h_sr, s_h_sr_cons = s_h.sig_rep
        constrs += s_h_sr_cons
        constrs += relative_c_sage(s_h_sr)
    lagrangian_sr, lagrangian_sr_cons = lagrangian.sig_rep
    constrs += lagrangian_sr_cons
    constrs += relative_c_sage(lagrangian_sr)
    for v in lagrangian.constant_term().variables():
        if v.name() == 'gamma':
            gamma = v
            break
    # noinspection PyUnboundLocalVariable
    obj = cvxpy.Maximize(gamma)
    prob = cvxpy.Problem(obj, constrs)
    return prob


def make_poly_lagrangian(f, gs, p, q, add_constant_poly=True):
    """
    Given a problem \inf{ f(x) : g(x) >= 0 for g in gs}, construct the q-fold constraints H,
    and the lagrangian
        L = f - \gamma - \sum_{h \in H} s_h * h
    where \gamma and the coefficients on Polynomials s_h are CVXPY Variables.

    :param f: a Polynomial (or a constant numeric type).
    :param gs: a nonempty list of Polynomials.
    :param p: a nonnegative integer. Defines the exponent set of the Polynomials s_h.
    :param q: a positive integer. Defines "H" as all products of q elements from gs.
    :param add_constant_poly: a boolean. If True, makes sure that "gs" contains a
    Polynomial that is identically equal to 1.

    :return: a Polynomial object with coefficients as affine expressions of CVXPY Variables.
    The coefficients will either be optimized directly (in the case of constrained_sage_primal),
    or simply used to determine appropriate dual variables (in the case of constrained_sage_dual).

    Also return a list of pairs of Polynomial objects. If the pair (p1, p2) is in this list,
    then p1 is a generalized Lagrange multiplier (which we should constrain to be nonnegative,
    somehow), and p2 represents a constraint p2(x) >= 0 in the polynomial program after taking
    products of the gs.
    """
    if not all([isinstance(g, Polynomial) for g in gs]):
        raise RuntimeError('Constraints must be Polynomial objects.')
    if add_constant_poly:
        gs.append(Polynomial({(0,) * gs[0].n: 1}))  # add the constant signomial
    if not isinstance(f, Polynomial):
        f = Polynomial({(0,) * gs[0].n: f})
    gs = set(gs)  # remove duplicates
    hs = set([np.prod(comb) for comb in combinations_with_replacement(gs, q)])
    gamma = cvxpy.Variable(name='gamma')
    lagrangian = f - gamma
    alpha_E_p = hierarchy_e_k([f] + list(gs), k=p)
    dualized_polynomials = []
    for h in hs:
        temp_shc = cvxpy.Variable(name='shc_' + str(h), shape=(alpha_E_p.shape[0],))
        temp_sh = Polynomial(alpha_E_p, temp_shc)
        lagrangian -= temp_sh * h
        dualized_polynomials.append((temp_sh, h))
    return lagrangian, dualized_polynomials
