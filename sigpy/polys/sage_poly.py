from sigpy.polys.polynomials import Polynomial
from sigpy.sage import sage_primal, sage_dual, sage_feasibility, hierarchy_e_k, relative_c_sage, relative_c_sage_star, \
    relative_coeff_vector
import cvxpy
import numpy as np
from itertools import combinations_with_replacement


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
    lagrangian, dualized_polynomials = make_poly_lagrangian(f, gs, p=p, q=q, add_constant_poly=(q != 1))
    # The Lagrangian must be a SAGE polynomial.
    lagrangian_sr, lagrangian_sr_cons = lagrangian.sig_rep
    constrs = lagrangian_sr_cons
    constrs += relative_c_sage(lagrangian_sr)
    # The generalized dual variables must be SAGE polynomials.
    for s_h, _ in dualized_polynomials:
        s_h_sr, s_h_sr_cons = s_h.sig_rep
        constrs += s_h_sr_cons
        constrs += relative_c_sage(s_h_sr)
    # The Lagrangian's coefficient vector contains references to a variable called "gamma".
    # Find this variable, and define the objective appropriately.
    obj = None
    for v in lagrangian.constant_term().variables():
        if v.name() == 'gamma':
            obj = cvxpy.Maximize(v)
            break
    # Construct the CVXPY Problem.
    prob = cvxpy.Problem(obj, constrs)
    return prob


def constrained_sage_poly_dual(f, gs, p=0, q=1):
    """
    Compute the dual f_{SAGE}^{(p, q)} bound for

        inf f(x) : g(x) >= 0 for g \in gs.

    :param f: a Polynomial.
    :param gs: a list of Polynomials.
    :param p: a nonnegative integer,
    :param q: a positive integer.

    :return: a CVXPY Problem that defines the dual formulation for f_{SAGE}^{(p, q)}.
    """
    lagrangian, dualized_polynomials = make_poly_lagrangian(f, gs, p=p, q=q, add_constant_poly=(q != 1))
    # In primal form, the Lagrangian is constrained to be a SAGE polynomial.
    # Introduce a dual variable "v" for this constraint.
    v = cvxpy.Variable(shape=(lagrangian.m, 1))
    constraints = relative_c_poly_sage_star(lagrangian, v)
    a = relative_coeff_vector(Polynomial({(0,) * f.n: 1}), lagrangian.alpha)
    a = a.reshape(a.size, 1)
    constraints.append(a.T * v == 1)
    # The generalized Lagrange multipliers "s_h" are presumed to be SAGE polynomials.
    # For each such multiplier, introduce an appropriate dual variable "v_h", along
    # with constraints over that dual variable.
    for s_h, h in dualized_polynomials:
        v_h = cvxpy.Variable(name='v_h_' + str(s_h), shape=(s_h.m, 1))
        constraints += relative_c_poly_sage_star(s_h, v_h)
        c_h = hierarchy_c_h_array(s_h, h, lagrangian)
        constraints.append(c_h * v == v_h)
    # Define the dual objective function.
    obj_vec = relative_coeff_vector(f, lagrangian.alpha)
    obj = cvxpy.Minimize(obj_vec * v)
    # Return the CVXPY Problem.
    prob = cvxpy.Problem(obj, constraints)
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
        gs.append(Polynomial({(0,) * gs[0].n: 1}))
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


def relative_c_poly_sage_star(p, y):
    """
    :param p: a Polynomial
    :param y: a cvxpy Variable with y.shape == (p.m, 1).

    :return: CVXPY Constraints over y (and additional auxilliary variables, as
    necessary) so that y defines a dual variable to the constraint that
    "p is a SAGE polynomial."
    """
    sr, sr_cons = p.sig_rep
    constrs = []
    evens = [i for i, row in enumerate(sr.alpha) if np.all(row % 2 == 0)]
    if len(evens) < sr.m:
        is_even = np.zeros(shape=(sr.m,), dtype=bool)
        is_even[evens] = True
        lambda_1_expr = []
        lambda_2_expr = []
        mu_expr = []
        for i in range(sr.m):
            if i in evens:
                lambda_1_expr.append(0)
                lambda_2_expr.append(0)
                mu_expr.append(cvxpy.Variable(shape=()))
            else:
                lambda_1_expr.append(cvxpy.Variable(shape=(), nonneg=True))
                lambda_2_expr.append(cvxpy.Variable(shape=(), nonneg=True))
                mu_expr.append(0)
        lambda_1_expr = cvxpy.vstack(lambda_1_expr)
        lambda_2_expr = cvxpy.vstack(lambda_2_expr)
        mu_expr = cvxpy.vstack(mu_expr)
        v = cvxpy.Variable(shape=(sr.m, 1))
        constrs = [v == lambda_1_expr + lambda_2_expr + mu_expr,
                   y == lambda_2_expr - lambda_1_expr + mu_expr]
        constrs += relative_c_sage_star(sr, v)
    else:
        constrs += relative_c_sage_star(sr, y)
    return constrs


def hierarchy_c_h_array(s_h, h, lagrangian):
    """
    Assume (s_h * h).alpha is a subset of lagrangian.alpha.

    :param s_h: a SAGE multiplier Polynomial for the constrained hierarchy
    :param h: the constraint Polynomial
    :param lagrangian: the Polynomial f - \gamma - \sum_{h \in H} s_h * h.

    :return: a matrix c_h so that if "v" is a dual variable to the constraint
    "lagrangian is a SAGE polynomial", then the constraint "s_h is SAGE"
    dualizes to "c_h * v \in C_{SAGE}^{POLY}(s_h)^{\star}".
    """
    c_h = np.zeros((s_h.alpha.shape[0], lagrangian.alpha.shape[0]))
    for i, row in enumerate(s_h.alpha):
        temp_poly = Polynomial({tuple(row): 1}) * h
        c_h[i, :] = relative_coeff_vector(temp_poly, lagrangian.alpha)
    return c_h
