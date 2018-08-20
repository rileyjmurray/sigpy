"""
   Copyright 2018 Riley John Murray

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import cvxpy
import numpy as np
from itertools import combinations_with_replacement
from signomials import Signomial, relative_coeff_vector


__NUMERIC_TYPES__ = (int, float, np.int_, np.float_)


def sage_dual(s, level=0):
    """
    :param s: a Signomial object.
    :param level: a nonnegative integer
    :return: a CVXPY Problem object representing the dual formulation for s_{SAGE}^{(level)}

    In the discussion that follows, let s satisfy s.alpha[0,:] == np.zeros((1,n)).

    When level == 0, the returned CVXPY problem has the following explicit form:
            min  (s.c).T * v
            s.t.    v[0] == 1
                    v[i] * ln(v[i] / v[j]) <= (s.alpha[i,:] - s.alpha[j,:]) * mu[i] for i \in N0, j \in Nc0, j != i.
                    mu[i] \in R^{s.n} for i \in N0
                    v \in R^{s.m}_{+}
            where N = { i : s.c[i] < 0}, N0 = union(N, {0}), Nc = { i : s.c[i] >= 0}, and Nc0 = union(Nc, {0}).

    When level > 0, the form of the optimization problem is harder to state explicitly. At a high level, the resultant
    CVXPY problem is the same as above, with the following modifications:
        (1) we introduce a multiplier signomial
                t_mul = Signomial(s.alpha, np.ones(s.m)) ** level,
        (2) as well as a constant signomial
                t_cst = Signomial(s.alpha, [1, 0, ..., 0]).
        (3) Then "s" is replaced by
                s_mod == s * t_mul,
        (4) and "v[0] == 1" is replaced by
                a * v == 1,
            where vector "a" is an appropriate permutation of (t_mul * t_cst).c, and finally
        (5) the index sets N0 and Nc0 are replaced by
                N_I = union(N, I) and Nc_I = union(Nc, I)
            for
                I = { i | a[i] != 0 }.
    """
    # Signomial definitions (for the objective).
    s_mod = Signomial(s.alpha_c)
    t_mul = Signomial(s.alpha, np.ones(s.m)) ** level
    lagrangian = (s_mod - cvxpy.Variable(name='gamma')) * t_mul
    s_mod = s_mod * t_mul
    # C_SAGE^STAR (v must belong to the set defined by these constraints).
    v = cvxpy.Variable(shape=(lagrangian.m, 1), name='v')
    constraints = relative_c_sage_star(lagrangian, v)
    # Equality constraint (for the Lagrangian to be bounded).
    a = relative_coeff_vector(t_mul, lagrangian.alpha)
    a = a.reshape(a.size, 1)
    constraints.append(a.T * v == 1)
    # Objective definition and problem creation.
    obj_vec = relative_coeff_vector(s_mod, lagrangian.alpha)
    obj = cvxpy.Minimize(obj_vec * v)
    prob = cvxpy.Problem(obj, constraints)
    # Add fields that we can access later.
    prob.s_mod = s_mod
    prob.s = s
    prob.level = level
    return prob


def relative_c_sage_star(s, v):
    """
    Given the Signomial s and a CVXPY variable "v", return a list of CVXPY Constraint objects such
    that v is a conic dual variable to the constraint "s.c \in C_{SAGE}(s.alpha)".

    :param s: a Signomial object
    :param v: a CVXPY Variable with v.size == s.m.
    :return a list of CVXPY Constraint objects.

    Remark 1: The CVXPY function kl_div operates in a way that differs from the relative entropy function as described
    in the literature on SAGE relaxations. Refer to the CVXPY documentation if our usage seems odd.

    Remark 2: This implementation is vectorized to minimize the length of the list "constraints". By doing this we
    significantly speed up the process of CVXPY converting our problem to its internal standard form.
    """
    alpha, c = s.alpha_c_arrays()
    if s.m <= 2:
        return [v >= 0]
    non_constants = [i for i, c_i in enumerate(c) if not isinstance(c_i, __NUMERIC_TYPES__)]
    N_I = [i for i, c_i in enumerate(c) if (i in non_constants) or c_i < 0]
    Nc_I = [i for i, c_i in enumerate(c) if (i in non_constants) or c_i > 0]
    # variable definitions
    mu = cvxpy.Variable(shape=(len(N_I), s.n), name=('mu_' + str(v.id)))
    # constraints
    constraints = []
    for i, ii in enumerate(N_I):
        # i = the index used for "mu", ii = index used for alpha and v
        j_neq_ii = [j for j in Nc_I if j != ii]
        expr1 = v[ii] * np.ones((len(j_neq_ii), 1))
        expr2 = cvxpy.kl_div(expr1, v[j_neq_ii]) + expr1 - v[j_neq_ii]
        expr3 = (alpha[ii, :] - alpha[j_neq_ii, :]) * mu[i, :].T
        constraints.append(expr2 <= cvxpy.reshape(expr3, (len(j_neq_ii), 1)))
    constraints.append(v[list(set(N_I + Nc_I))] >= 0)
    return constraints


def sage_primal(s, level=0, special_multiplier=None):
    """
    :param s: a Signomial object.
    :param level: a nonnegative integer
    :param special_multiplier: an optional parameter, applicable when level > 0. Must be a nonzero
    SAGE function.
    :return: a CVXPY Problem object representing the primal formulation for s_{SAGE}^{(level)}

    Unlike the sage_dual, this formulation can be stated in full generality without too much trouble.
    We define a multiplier signomial "t" as either the standard multiplier (Signomial(s.alpha, np.ones(s.n))),
    or a user-provided multiplier. We then return a CVXPY Problem representing

        max  gamma
        s.t.    s_mod.c \in C_{SAGE}(s_mod.alpha)
        where   s_mod := (t ** level) * (s - gamma).

    Our implementation of Signomial objects allows CVXPY variables in the coefficient vector c. As a result, the
    mapping "gamma \to s_mod.c" is an affine function that takes in a CVXPY Variable and returns a CVXPY Expression.
    This makes it very simple to represent "s_mod.c \in C_{SAGE}(s_mod.alpha)" via CVXPY Constraints. The work defining
    the necessary CVXPY variables and constructing the CVXPY constraints is handled by the function "c_sage."
    """
    if special_multiplier is None:
        t = Signomial(s.alpha, np.ones(s.m))
    else:
        # noinspection PyTypeChecker
        if np.all(special_multiplier.c == 0):
            raise RuntimeError('The multiplier must be a nonzero signomial.')
        # test if SAGE
        prob = sage_feasibility(special_multiplier)
        if prob.solve() < 0:
            raise RuntimeError('The multiplier must be a SAGE function.')
        t = special_multiplier
    gamma = cvxpy.Variable(name='gamma')
    s_mod = (s - gamma) * (t ** level)
    s_mod.remove_terms_with_zero_as_coefficient()
    constraints = relative_c_sage(s_mod)
    obj = cvxpy.Maximize(gamma)
    prob = cvxpy.Problem(obj, constraints)
    # Add fields that we can access later.
    prob.s_mod = s_mod
    prob.s = s
    prob.level = level
    return prob


def relative_c_sage(s):
    """
    Given a signomial "s", return a list of CVXPY Constraint objects over CVXPY Variables c_vars and nu_vars
    such that s is SAGE iff c_vars and nu_vars satisfy every constraint in this list.

    :param s: a Signomial object (likely with the property that s.c is a CVXPY Expression).
    :return: constraints - a list of CVXPY Constraint objects.
    """
    if s.m <= 2:
        return [cvxpy.vstack(s.c.tolist()) >= 0]
    alpha, c = s.alpha_c_arrays()
    non_constants = [i for i, c_i in enumerate(c) if not isinstance(c_i, __NUMERIC_TYPES__)]
    N_I = [i for i, c_i in enumerate(c) if (i in non_constants) or (c_i < 0)]
    c_vars = dict()
    nu_vars = dict()
    constraints = []
    for i in N_I:
        c_i, nu_i, constrs_i = relative_c_age(s, i)
        c_vars[i] = c_i
        nu_vars[i] = nu_i
        constraints += constrs_i
    # Now the constraints that the c_vars sum to c.
    vec_expr = sum(c_vars.values())
    c = cvxpy.vstack(c.tolist())
    constraints.append(vec_expr == c)
    return constraints


def relative_c_age(s, i):
    constraints = list()
    idx_set = np.arange(s.m) != i
    # variable definitions
    c_var = cvxpy.Variable(shape=(s.m, 1), name='c^{(' + str(i) + '})_' + str(s))
    nu_var = cvxpy.Variable(shape=(s.m, 1), name='nu^{(' + str(i) + '})_' + str(s))
    # variable non-negativity constraints
    constraints.append(c_var[idx_set] >= 0)
    constraints.append(nu_var[idx_set] >= 0)
    # main constraints
    constraints.append(s.alpha.T * nu_var == np.zeros(shape=(s.n, 1)))  # convex cover constraint 1
    constraints.append(cvxpy.sum(nu_var) == 0)  # convex cover constraint 2
    kl_expr1 = cvxpy.kl_div(nu_var[idx_set], np.exp(1) * c_var[idx_set])
    kl_expr2 = nu_var[idx_set] - np.exp(1) * c_var[idx_set]
    rel_ent = kl_expr1 + kl_expr2
    constraints.append(cvxpy.sum(rel_ent) - c_var[i] <= 0)  # relative entropy constraint
    return c_var, nu_var, constraints


def sage_feasibility(s):
    """
    :param s: a Signomial object
    :return: a CVXPY Problem which is feasible iff s.c \in C_{SAGE}(s.alpha)
    """
    constraints = relative_c_sage(s)
    # noinspection PyTypeChecker
    obj = cvxpy.Maximize(0)
    prob = cvxpy.Problem(obj, constraints)
    return prob


def sage_multiplier_search(s, level=1):
    """
    Suppose we have a nonnegative signomial s, where s_mod := s * (Signomial(s.alpha, np.ones(s.m))) ** level
    is not SAGE. Do we have an alternative do proving that s is nonnegative other than moving up the SAGE
    hierarchy? Indeed we do. We can define a multiplier

        mult = Signomial(alpha_hat, c_tilde)

    where the rows of alpha_hat are all level-wise sums of rows from s.alpha, and c_tilde is a CVXPY Variable
    defining a nonzero SAGE function. Then we can check if s_mod := s * mult is SAGE for any choice of c_tilde.

    :param s: a Signomial object
    :param level: a nonnegative integer
    :return: a CVXPY Problem that is feasible iff s * mult is SAGE for some SAGE multiplier signomial "mult".
    """
    s.remove_terms_with_zero_as_coefficient()
    constraints = []
    mult_alpha = hierarchy_e_k([s], k=level)
    c_tilde = cvxpy.Variable(mult_alpha.shape[0], name='c_tilde')
    mult = Signomial(mult_alpha, c_tilde)
    constraints += relative_c_sage(mult)
    constraints.append(cvxpy.sum(c_tilde) >= 1)
    sig_under_test = mult * s
    sage_membership_constraints = relative_c_sage(sig_under_test)
    constraints += sage_membership_constraints
    # noinspection PyTypeChecker
    obj = cvxpy.Maximize(0)
    prob = cvxpy.Problem(obj, constraints)
    return prob


def constrained_sage_primal(f, gs, p=0, q=1):
    """
    Compute the primal f_{SAGE}^{(p, q)} bound for

        inf f(x) : g(x) >= 0 for g \in gs.

    :param f: a Signomial.
    :param gs: a list of Signomials.
    :param p: a nonnegative integer.
    :param q: a positive integer.
    :return: a CVXPY Problem that defines the primal formulation for f_{SAGE}^{(p, q)}.
    """
    lagrangian, dualized_signomials = make_lagrangian(f, gs, p=p, q=q)
    constrs = sum([relative_c_sage(s_h) for s_h, _ in dualized_signomials], [])
    constrs += relative_c_sage(lagrangian)
    for v in lagrangian.constant_term().variables():
        if v.name() == 'gamma':
            gamma = v
            break
    # noinspection PyUnboundLocalVariable
    obj = cvxpy.Maximize(gamma)
    prob = cvxpy.Problem(obj, constrs)
    return prob


def constrained_sage_dual(f, gs, p=0, q=1):
    """
    Compute the dual f_{SAGE}^{(p, q)} bound for

        inf f(x) : g(x) >= 0 for g \in gs.

    :param f: a Signomial.
    :param gs: a list of Signomials.
    :param p: a nonnegative integer,
    :param q: a positive integer.
    :return: a CVXPY Problem that defines the dual formulation for f_{SAGE}^{(p, q)}.
    """
    lagrangian, dualized_signomials = make_lagrangian(f, gs, p=p, q=q)
    v = cvxpy.Variable(shape=(lagrangian.m, 1))
    constraints = relative_c_sage_star(lagrangian, v)
    for s_h, h in dualized_signomials:
        v_h = cvxpy.Variable(name='v_h_' + str(s_h), shape=(s_h.m, 1))
        constraints += relative_c_sage_star(s_h, v_h)
        c_h = hierarchy_c_h_array(s_h, h, lagrangian)
        constraints.append(c_h * v == v_h)
    # Equality constraint (for the Lagrangian to be bounded).
    a = relative_coeff_vector(Signomial({(0,) * f.n: 1}), lagrangian.alpha)
    a = a.reshape(a.size, 1)
    constraints.append(a.T * v == 1)
    obj_vec = relative_coeff_vector(f, lagrangian.alpha)
    obj = cvxpy.Minimize(obj_vec * v)
    prob = cvxpy.Problem(obj, constraints)
    return prob


def make_lagrangian(f, gs, p, q, add_constant_sig=True):
    """
    Given a problem \inf{ f(x) : g(x) >= 0 for g in gs}, construct the q-fold constraints H,
    and the lagrangian
        L = f - \gamma - \sum_{h \in H} s_h * h
    where \gamma and the coefficients on Signomials s_h are CVXPY Variables.

    :param f: a Signomial (or a constant numeric type).
    :param gs: a nonempty list of Signomials.
    :param p: a nonnegative integer. Defines the exponent set of the Signomials s_h.
    :param q: a positive integer. Defines "H" as all products of q elements from gs.
    :param add_constant_sig: a boolean. If True, makes sure that "gs" contains a
    Signomial that is identically equal to 1.

    :return: a Signomial object with coefficients as affine expressions of CVXPY Variables.
    The coefficients will either be optimized directly (in the case of constrained_sage_primal),
    or simply used to determine appropriate dual variables (in the case of constrained_sage_dual).
    """
    if not all([isinstance(g, Signomial) for g in gs]):
        raise RuntimeError('Constraints must be Signomial objects.')
    if add_constant_sig:
        gs.append(Signomial({(0,) * gs[0].n: 1}))  # add the constant signomial
    if not isinstance(f, Signomial):
        f = Signomial({(0,) * gs[0].n: f})
    gs = set(gs)  # remove duplicates
    hs = set([np.prod(comb) for comb in combinations_with_replacement(gs, q)])
    gamma = cvxpy.Variable(name='gamma')
    lagrangian = f - gamma
    alpha_E_p = hierarchy_e_k([f] + list(gs), k=p)
    dualized_signomials = []
    for h in hs:
        temp_shc = cvxpy.Variable(name='shc_' + str(h), shape=(alpha_E_p.shape[0],))
        temp_sh = Signomial(alpha_E_p, temp_shc)
        lagrangian -= temp_sh * h
        dualized_signomials.append((temp_sh, h))
    return lagrangian, dualized_signomials


def get_vars_by_name(prob):
    """
    A helper function for retrieving the values of variables in a given CVXPY Problem.

    :param prob: a CVXPY Problem.
    :return: a dictionary from the variable's name (in the CVXPY Problem) to it's current value.
    """
    variables = prob.variables()
    named_vars = dict()
    for var in variables:
        named_vars[var.name()] = var.value
    return named_vars


def hierarchy_c_h_array(s_h, h, lagrangian):
    """
    Assume (s_h * h).alpha is a subset of lagrangian.alpha.

    :param s_h: a SAGE multiplier Signomial for the constrained hierarchy
    :param h: the constraint Signomial
    :param lagrangian: the Signomial f - \gamma - \sum_{h \in H} s_h * h.

    :return: a matrix c_h so that if "v" is a dual variable to the constraint
    "lagrangian is SAGE", then the constraint "s_h is SAGE" is dualizes to
    "c_h * v \in C_{SAGE}^{\star}(s_h)".
    """
    c_h = np.zeros((s_h.alpha.shape[0], lagrangian.alpha.shape[0]))
    for i, row in enumerate(s_h.alpha):
        temp_sig = Signomial({tuple(row): 1}) * h
        c_h[i, :] = relative_coeff_vector(temp_sig, lagrangian.alpha)
    return c_h


def hierarchy_e_k(sig_list, k):
    """
    :param sig_list: a list of Signomial objects over a common domain R^n
    :param k: a nonnegative integer
    :return: If "alpha" denotes the union of exponent vectors over Signomials in
    sig_list, then this function returns "E_k(alpha)" from the original paper
    on the SAGE hierarchy.
    """
    alpha_tups = sum([list(s.alpha_c.keys()) for s in sig_list], [])
    alpha_tups = set(alpha_tups)
    s = Signomial(dict([(a, 1.0) for a in alpha_tups]))
    s = s ** k
    return s.alpha


