import numpy as np
import unittest
from sigpy.signomials import Signomial
from sigpy.signomials import standard_monomials


class TestSignomials(unittest.TestCase):

    @staticmethod
    def are_equal(sig1, sig2):
        diff = sig1 - sig2
        diff.remove_terms_with_zero_as_coefficient()
        return diff.m == 1 and diff.c[0] == 0

    def test_construction(self):
        # data for tests
        alpha = np.array([[0], [1], [2]])
        c = np.array([1, -1, -2])
        alpha_c = {(0,): 1, (1,): -1, (2,): -2}
        # Construction with two numpy arrays as arguments
        s = Signomial(alpha, c)
        assert s.n == 1 and s.m == 3 and s.alpha_c == alpha_c
        # Construction with a vector-to-coefficient dictionary
        s = Signomial(alpha_c)
        recovered_alpha_c = dict()
        for i in range(s.m):
            recovered_alpha_c[tuple(s.alpha[i, :])] = s.c[i]
        assert s.n == 1 and s.m == 3 and alpha_c == recovered_alpha_c

    def test_standard_monomials(self):
        x = standard_monomials(2)
        y_actual = x[0] + 3 * x[1] ** 2
        y_expect = Signomial({(1,0): 1, (0,2): 3})
        assert TestSignomials.are_equal(y_actual, y_expect)
        x = standard_monomials(4)
        y_actual = np.sum(x)
        y_expect = Signomial(np.eye(4), np.ones(shape=(4,)))
        assert TestSignomials.are_equal(y_actual, y_expect)

    # noinspection PyUnresolvedReferences
    def test_scalar_multiplication(self):
        # data for tests
        alpha0 = np.array([[0], [1], [2]])
        c0 = np.array([1, 2, 3])
        s0 = Signomial(alpha0, c0)
        # Tests
        s = 2 * s0
        # noinspection PyTypeChecker
        assert set(s.c) == set(2 * s0.c)
        s = s0 * 2
        # noinspection PyTypeChecker
        assert set(s.c) == set(2 * s0.c)
        s = 1 * s0
        assert s.alpha_c == s0.alpha_c
        s = 0 * s0
        s.remove_terms_with_zero_as_coefficient()
        assert s.m == 1 and set(s.c) == {0}

    def test_addition_and_subtraction(self):
        # data for tests
        s0 = Signomial({(0,): 1, (1,): 2, (2,): 3})
        t0 = Signomial({(-1,): 5})
        # tests
        s = s0 - s0
        s.remove_terms_with_zero_as_coefficient()
        assert s.m == 1 and set(s.c) == {0}
        s = -s0 + s0
        s.remove_terms_with_zero_as_coefficient()
        assert s.m == 1 and set(s.c) == {0}
        s = s0 + t0
        assert s.alpha_c == {(-1,): 5, (0,): 1, (1,): 2, (2,): 3}

    def test_signomial_multiplication(self):
        # data for tests
        s0 = Signomial({(0,): 1, (1,): 2, (2,): 3})
        t0 = Signomial({(-1,): 1})
        q0 = Signomial({(5,): 0})
        # tests
        s = s0 * t0
        s.remove_terms_with_zero_as_coefficient()
        assert s.alpha_c == {(-1,): 1, (0,): 2, (1,): 3}
        s = t0 * s0
        s.remove_terms_with_zero_as_coefficient()
        assert s.alpha_c == {(-1,): 1, (0,): 2, (1,): 3}
        s = s0 * q0
        s.remove_terms_with_zero_as_coefficient()
        assert s.alpha_c == {(0,): 0}

    def test_exponentiation(self):
        # raise to a negative power
        s = Signomial({(0.25,): -1})
        t_actual = s ** -3
        t_expect = Signomial({(-0.75,): -1})
        assert TestSignomials.are_equal(t_actual, t_expect)
        # raise to a fractional power
        s = Signomial({(2,): 9})
        t_actual = s ** 0.5
        t_expect = Signomial({(1,): 3})
        assert TestSignomials.are_equal(t_actual, t_expect)
        # raise to a nonnegative integer power
        s = Signomial({(0,): 1, (1,): 2})
        t_actual = s ** 2
        t_expect = Signomial({(0,): 1, (1,): 4, (2,): 4})
        assert TestSignomials.are_equal(t_actual, t_expect)

    def test_signomial_evaluation(self):
        s = Signomial({(1,): 1})
        assert s(0) == 1 and abs(s(1) - np.exp(1)) < 1e-10
        zero = np.array([0])
        one = np.array([1])
        assert s(zero) == 1 and abs(s(one) - np.exp(1)) < 1e-10
        zero_one = np.array([[0, 1]])
        assert np.allclose(s(zero_one), np.exp(zero_one), rtol=0, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
