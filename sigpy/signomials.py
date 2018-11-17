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
from collections import defaultdict


__NUMERIC_TYPES__ = (int, float, np.int_, np.float_)


def relative_coeff_vector(s, reference_alpha):
    c = np.zeros(reference_alpha.shape[0])
    corr = row_correspondence(s.alpha, reference_alpha)
    c[corr] = s.c
    return c


def row_correspondence(alpha1, alpha2):
    """
    This is a helper function for the SAGE hierarchy when "level" > 0. It applies only
    when the rows of alpha1 are a subset of the rows of alpha2, and returns a list
    "alpha1_to_alpha2" such that

        alpha1 == alpha2[alpha1_to_alpha2, :].

    This is useful because it allows us to speak of the "i-th" exponent in a meaningful
    way when dealing with Signomials, without having to adopt a canonical ordering for
    exponent vectors.

    :param alpha1: numpy n-d array.
    :param alpha2: a numpy n-d array.
    :return: a list "alpha1_to_alpha2" such that alpha1 == alpha2[alpha1_to_alpha2, :].
    """
    alpha1_to_alpha2 = []
    for row in alpha1:
        # noinspection PyTypeChecker
        loc = np.where(np.all(alpha2 == row, axis=1))[0][0]
        alpha1_to_alpha2.append(loc)
    return alpha1_to_alpha2


def standard_monomials(n):
    """
    Returns a numpy array "x" of length n, with "x[i]" as a signomial with
    one term corresponding to the (i+1)-th standard basis vector in R^n.

    This is useful for constructing signomials with syntax such as
        f = (x[0] ** 1.5) * (x[2] ** -0.6) - x[1] * x[2]
    """
    x = np.empty(shape=(n,), dtype=object)
    for i in range(n):
        ei = np.zeros(shape=(1, n))
        ei[0, i] = 1
        x[i] = Signomial(ei, np.array([1]))
    return x


class Signomial(object):

    def __init__(self, alpha_maybe_c, c=None):
        """
        PURPOSE:

        Provide a symbolic representation of a function
            s(x) = \sum_{i=1}^m c[i] * exp(alpha[i] * x),
        where alpha[i] are row vectors of length n, and c[i] are scalars.

        CONSTRUCTION:

        There are two ways to call the Signomial constructor. The first way is to specify a dictionary from tuples to
        scalars. The tuples are interpreted as linear functionals alpha[i], and the scalars are the corresponding
        coefficients. The second way is to specify two arguments. In this case the first argument is a NumPy array
        where the rows represent linear functionals, and the second argument is a vector of corresponding coefficients.

        CAPABILITIES:

        Signomial objects are closed under addition, subtraction, and multiplication (but not division). These
        arithmetic operations are enabled through Python's default operator overloading conventions. That is,
            s1 + s2, s1 - s2, s1 * s2
        all do what you think they should. Arithmetic is defined between Signomial and non-Signomial objects by
        treating the non-Signomial object as a scalar; in such a setting the non-Signomial object is assumed
        to implement the operations "+", "-", and "*". Common use-cases include forming Signomials such as:
            s1 + v, s1 * v
        when "v" is a CVXPY Variable, CVXPY Expression, or numeric type.
        Signomial objects are callable. If x is a numpy array of length n, then s(x) computes the Signomial object
        as though it were any other elementary Python function.

        WORDS OF CAUTION:

        Signomials contain redundant information. In particular, s.alpha_c is the dictionary which is taken to *define*
        the signomial as a mathematical object. However, it is useful to have rapid access to the matrix of linear
        functionals "alpha", or the coefficient vector "c" as numpy arrays. The current implementation of this
        class is such that if a user modifies the variables s.c or s.alpha directly, there may be an inconsistency
        between these fields and the dictionary s.alpha_c. THEREFORE THOSE FIELDS SHOULD NOT BE MODIFIED WITHOUT TAKING
        GREAT CARE TO ENSURE CONSISTENCY WITH THE SIGNOMIAL'S DICTIONARY REPRESENTATION.

        PARAMETERS:

        :param alpha_maybe_c: either (1) a dictionary from tuples-of-numbers to scalars, or (2) a numpy array object
        with the same number of rows as c has entries (in the event that the second argument "c" is provided).
        :param c: optional. specified iff alpha_maybe_c is a numpy array.
        """
        # noinspection PyArgumentList
        if c is None:
            self.alpha_c = defaultdict(lambda: 0, alpha_maybe_c)
        else:
            alpha = alpha_maybe_c.tolist()
            if len(alpha) != c.size:
                raise RuntimeError('alpha and c specify different numbers of terms')
            self.alpha_c = defaultdict(lambda: 0)
            for j in range(c.size):
                self.alpha_c[tuple(alpha[j])] += c[j]
        self.n = len(list(self.alpha_c.items())[0][0])
        self.alpha_c[(0,) * self.n] += 0  # ensures that there's a constant term.
        self.m = len(self.alpha_c)
        self._update_alpha_c_arrays()

    def constant_term(self):
        return self.alpha_c[(0,) * self.n]

    def query_coeff(self, a):
        """
        :param a: a numpy array of shape (self.n,).
        :return:
        """
        atup = tuple(a)
        if atup in self.alpha_c:
            return self.alpha_c[atup]
        else:
            return 0

    def constant_location(self):
        return np.where((self.alpha == 0).all(axis=1))[0][0]

    def alpha_c_arrays(self):
        return self.alpha, self.c

    def _update_alpha_c_arrays(self):
        """
        Call this function whenever the dictionary representation of this Signomial object has been updated.
        """
        alpha = []
        c = []
        for k, v in self.alpha_c.items():
            alpha.append(k)
            c.append(v)
        self.alpha = np.array(alpha)
        self.c = np.array(c)

    def __add__(self, other):
        if not isinstance(other, Signomial):
            tup = (0,) * self.n
            d = {tup: other}
            other = Signomial(d)
        d = defaultdict(lambda: 0, self.alpha_c)
        for k, v in other.alpha_c.items():
            d[k] += v
        return Signomial(d)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Signomial):
            tup = (0,) * self.n
            d = {tup: other}
            other = Signomial(d)
        d = defaultdict(lambda: 0)
        alpha1, c1 = self.alpha_c_arrays()
        alpha2, c2 = other.alpha_c_arrays()
        for i1, v1 in enumerate(alpha1):
            for i2, v2 in enumerate(alpha2):
                d[tuple(v1 + v2)] += c1[i1] * c2[i2]
        return Signomial(d)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        # noinspection PyTypeChecker
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        # noinspection PyTypeChecker
        return other + (-1) * self

    def __pow__(self, power, modulo=None):
        if self.c.dtype not in __NUMERIC_TYPES__:
            raise RuntimeError('Cannot exponentiate signomials with symbolic coefficients.')
        if isinstance(power, int) and power >= 0:
            s = Signomial(self.alpha_c)
            for t in range(power-1):
                s = s * self
            if power == 0:
                # noinspection PyTypeChecker
                return Signomial({(0,) * s.n: 1})
            elif power == 1:
                return s
            else:
                return Signomial(s.alpha_c)
        else:
            d = dict((k, v) for (k, v) in self.alpha_c.items() if v != 0)
            if len(d) != 1:
                raise ValueError('Only signomials with exactly one term can be raised to power %s.')
            v = list(d.values())[0]
            if v < 0 and not isinstance(power, int):
                raise ValueError('Cannot compute non-integer power %s of coefficient %s', power, v)
            alpha_tup = tuple(power * ai for ai in list(d.keys())[0])
            c = float(v) ** power
            s = Signomial(alpha_maybe_c={alpha_tup: c})
            return s

    def __neg__(self):
        # noinspection PyTypeChecker
        return self.__mul__(-1)

    def __call__(self, x):
        """
        Evaluates the mathematical function specified by the current Signomial object.

        :param x: either a scalar (if self.n == 1), or a numpy n-d array with len(x.shape) <= 2
        and x.shape[0] == self.n.
        :return:  If x is a scalar or an n-d array of shape (self.n,), then "val" is a numeric
        type equal to the signomial evaluated at x. If instead x is of shape (self.n, k) for
        some positive integer k, then "val" is a numpy n-d array of shape (k,), with val[i]
        equal to the current signomial evaluated on the i^th column of x.

        This function's behavior is undefined when x is not a scalar and has len(x.shape) > 2.
        """
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1):
            x = np.array([np.asscalar(np.array(x))])  # coerce into a 1d array of shape (1,).
        if not x.shape[0] == self.n:
            raise ValueError('The point must be in R^' + str(self.n) +
                             ', but the provided point is in R^' + str(x.shape[0]))
        exponents = np.dot(self.alpha.astype(np.float128), x.astype(np.float128))
        linear_vars = np.exp(exponents).astype(np.float128)
        val = np.dot(self.c, linear_vars)
        return val

    def remove_terms_with_zero_as_coefficient(self):
        d = dict()
        for (k, v) in self.alpha_c.items():
            if (not isinstance(v, __NUMERIC_TYPES__)) or v != 0:
                d[k] = v
        self.alpha_c = defaultdict(lambda: 0, d)
        tup = (0,) * self.n
        self.alpha_c[tup] += 0
        self.m = len(self.alpha_c)
        self._update_alpha_c_arrays()
        return

    def num_nontrivial_neg_terms(self):
        zero_location = self.constant_location()
        negs = np.where(self.c < 0)
        if len(negs[0]) > 0 and negs[0][0] == zero_location:
            return len(negs[0]) - 1
        else:
            return len(negs[0])
