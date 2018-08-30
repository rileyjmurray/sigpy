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
import numpy as np
from sage import get_vars_by_name
from signomials import row_correspondence, Signomial


def sol_recovery_from_dual(alpha, c, v):
    alpha = (alpha[c >= 0, :])
    v = v[c >= 0]
    b = np.reshape(np.log(v), newshape=(alpha.shape[0], 1))
    x = np.linalg.lstsq(alpha, b, rcond=None)
    return x


def collapse_lifted_dual_variable_to_r_m(prob):
    big_v = get_vars_by_name(prob)['v']
    v = np.zeros(prob.s.ell)
    t_mul = multiplier_signomial(prob.s) ** prob.level
    for i, row in enumerate(prob.s.alpha):
        ti = multiplier_signomial(prob.s, single_term=True, tgt_row=row)
        temp_sig = ti * t_mul
        corr = row_correspondence(prob.s_mod, temp_sig)
        v[i] = temp_sig.c[corr] * big_v
    return v


def multiplier_signomial(s, single_term=False, tgt_row=None):
    """
    This is a helper function for the SAGE hierarchy when "level" > 0. It returns a Signomial with the same
    exponent configuration as "s", but allows us to choose among coefficient vectors which are particularly
    useful in managing the hierarchy's bookkeeping requirements.

    :param s: a Signomial object
    :param single_term: indicates whether the coefficient vector of the returned signomial has exactly one
    nonzero entry.
    :param tgt_row: applicable only when single_term is True. In this case, the resultant signomial's coefficient vector
     will have a 1 on the term exp(tgt_row * x), and a zero on all other terms.
    :return: a Signomial object with (1) the same exponent configuration as s, and (2) coefficient vector either all
    1's, or one 1 and all other entries set to zero.
    """
    alpha_t = dict()
    if tgt_row is None:
        tgt_row = np.zeros([1, s.n])
    for row in s.alpha_c.keys():
        alpha_t[row] = 1
        if single_term and np.any(row != tgt_row):
            alpha_t[row] = 0
    return Signomial(alpha_t)
