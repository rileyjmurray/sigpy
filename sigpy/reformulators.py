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

from sigpy.signomials import Signomial


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


def sig_for_copositivity_test(a_array):
    # a_array is an n-by-n numpy array (NOT A "MATRIX")
    n = a_array.shape[0]
    alpha_c = dict()
    for i in range(n):
        for j in range(i+1):
            if a_array[i, j] != 0:
                vec = np.zeros(n)
                vec[i] += 1
                vec[j] += 1
                alpha_c[tuple(vec.tolist())] = a_array[i, j]
                if i != j:
                    alpha_c[tuple(vec.tolist())] += a_array[j, i]
            else:
                continue
    return Signomial(alpha_c)


def sig_for_psd_test(a_array):
    # a_array is an n-by-n numpy array (NOT A "MATRIX")
    n = a_array.shape[0]
    s = 0
    xs = [hyperbolic_sine_transform(n, i) for i in range(n)]
    for i in range(n):
        for j in range(i+1):
            if a_array[i, j] != 0:
                if i == j:
                    s += a_array[i, j] * (xs[i] * xs[j])
                else:
                    s += (a_array[i, j] + a_array[j, i]) * (xs[i] * xs[j])
            else:
                continue
    return s


def hyperbolic_sine_transform(n, i, scale=1):
    vec = np.zeros(n)
    vec[i] = 1
    s = Signomial({tuple(vec.tolist()): scale, tuple((-vec).tolist()): -scale})
    return s
