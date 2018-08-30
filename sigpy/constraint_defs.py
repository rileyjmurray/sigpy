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
from sigpy.signomials import Signomial


__NUMERIC_TYPES__ = (int, float, np.int_, np.float_)


def dual_variable_to_sage_signomial(s, v):
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


def signomial_is_sage(s):
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
        c_i, nu_i, constrs_i = age_cone(s, i)
        c_vars[i] = c_i
        nu_vars[i] = nu_i
        constraints += constrs_i
    # Now the constraints that the c_vars sum to c.
    vec_expr = sum(c_vars.values())
    c = cvxpy.vstack(c.tolist())
    constraints.append(vec_expr == c)
    return constraints


def age_cone(s, i):
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


def polynomial_is_sage(p):
    """

    :param p: a Polynomial object
    :return: a list of constraints over
    """
    sig_rep = signomial_representative(p)
    need_constrs = [i for i in range(p.m) if not isinstance(sig_rep.c[i], __NUMERIC_TYPES__)]
    constrs = [sig_rep.c[i] <= cvxpy.abs(p.c[i]) for i in need_constrs] + signomial_is_sage(sig_rep)
    return constrs


def signomial_representative(p):
    sig_rep = Signomial(p.alpha_c)
    even_locs = p.even_monomial_locations()
    for i in range(p.m):
        if i not in even_locs:
            if isinstance(p.c[i], __NUMERIC_TYPES__):
                sig_rep.c[i] = -abs(p.c[i])
            else:
                sig_rep.c[i] = cvxpy.Variable()
    return sig_rep


