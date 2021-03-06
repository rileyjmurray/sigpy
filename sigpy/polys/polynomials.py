from collections import defaultdict
from sigpy.signomials import Signomial
import numpy as np
import cvxpy

__NUMERIC_TYPES__ = (int, float, np.int_, np.float_)


def standard_monomials(n):
    """
    Returns a numpy array "x" of length n, with "x[i]" as a Polynomial with
    one term corresponding to the (i+1)-th standard basis vector in R^n.

    This is useful for constructing polynomials with syntax such as ...

        x = standard_monomials(3)

        f = (x[0] ** 2) * x[2] - 4 * x[1] * x[2] * x[0]
    """
    x = np.empty(shape=(n,), dtype=object)
    for i in range(n):
        ei = np.zeros(shape=(1, n))
        ei[0, i] = 1
        x[i] = Polynomial(ei, np.array([1]))
    return x


class Polynomial(Signomial):

    def __init__(self, alpha_maybe_c, c=None):
        Signomial.__init__(self, alpha_maybe_c, c)
        if not np.all(self.alpha % 1 == 0):
            raise RuntimeError('Exponents must belong the the integer lattice.')
        if not np.all(self.alpha >= 0):
            raise RuntimeError('Exponents must be nonnegative.')
        self._sig_rep = None
        self._sig_rep_constrs = []

    def __mul__(self, other):
        if not isinstance(other, Polynomial):
            if isinstance(other, Signomial):
                raise RuntimeError('Cannot multiply signomials and polynomials.')
            # else, we assume that "other" is a scalar type
            other = Polynomial.promote_scalar_to_polynomial(other, self.n)
        self_var_coeffs = (self.c.dtype not in __NUMERIC_TYPES__)
        other_var_coeffs = (other.c.dtype not in __NUMERIC_TYPES__)
        if self_var_coeffs and other_var_coeffs:
            raise RuntimeError('Cannot multiply two polynomials that posesses non-numeric coefficients.')
        temp = Signomial.__mul__(self, other)
        temp = Polynomial(temp.alpha, temp.c)
        return temp

    def __add__(self, other):
        if isinstance(other, Signomial) and not isinstance(other, Polynomial):
            raise RuntimeError('Cannot add signomials to polynomials.')
        temp = Signomial.__add__(self, other)
        temp = Polynomial(temp.alpha, temp.c)
        return temp

    def __sub__(self, other):
        if isinstance(other, Signomial) and not isinstance(other, Polynomial):
            raise RuntimeError('Cannot subtract a signomial from a polynomial (or vice versa).')
        temp = Signomial.__sub__(self, other)
        temp = Polynomial(temp.alpha, temp.c)
        return temp

    def __rmul__(self, other):
        # multiplication is commutative
        return Polynomial.__mul__(self, other)

    def __radd__(self, other):
        # addition is commutative
        return Polynomial.__add__(self, other)

    def __rsub__(self, other):
        # subtract self, from other
        # rely on correctness of __add__ and __mul__
        return other + (-1) * self

    def __neg__(self):
        # rely on correctness of __mul__
        return (-1) * self

    def __pow__(self, power, modulo=None):
        if self.c.dtype not in __NUMERIC_TYPES__:
            raise RuntimeError('Cannot exponentiate polynomials with symbolic coefficients.')
        temp = Signomial(self.alpha, self.c)
        temp = temp ** power
        temp = Polynomial(temp.alpha, temp.c)
        return temp

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('')

    @property
    def sig_rep(self):
        if self._sig_rep is None:
            self.compute_sig_rep()
        return self._sig_rep, self._sig_rep_constrs

    def compute_sig_rep(self):
        self._sig_rep = None
        self._sig_rep_constrs = []
        d = defaultdict(lambda: 0)
        for i, row in enumerate(self.alpha):
            if np.any(row % 2 != 0):
                row = tuple(row)
                if isinstance(self.c[i], __NUMERIC_TYPES__):
                    d[row] = -abs(self.c[i])
                else:
                    d[row] = cvxpy.Variable(shape=(), name=('sig_rep_coeff[' + str(i) + ']'))
                    self._sig_rep_constrs.append(d[row] <= self.c[i])
                    self._sig_rep_constrs.append(d[row] <= -self.c[i])
            else:
                d[tuple(row)] = self.c[i]
        self._sig_rep = Signomial(d)
        pass

    @staticmethod
    def promote_scalar_to_polynomial(scalar, n):
        alpha = np.array([[0] * n])
        c = np.array([scalar])
        return Polynomial(alpha, c)

