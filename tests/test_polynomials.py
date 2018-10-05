import numpy as np
import unittest
from sigpy.polys.polynomials import Polynomial


class TestPolynomials(unittest.TestCase):

    #
    #    Test arithmetic and operator overloading.
    #

    # noinspection PyUnresolvedReferences
    def test_scalar_multiplication(self):
        # data for tests
        alpha0 = np.array([[0], [1], [2]])
        c0 = np.array([1, 2, 3])
        s0 = Polynomial(alpha0, c0)
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
        s0 = Polynomial(np.array([[0], [1], [2]]),
                        np.array([1, 2, 3]))
        t0 = Polynomial(np.array([[4]]),
                        np.array([5]))
        # tests
        s = s0 - s0
        s.remove_terms_with_zero_as_coefficient()
        assert s.m == 1 and set(s.c) == {0}
        s = -s0 + s0
        s.remove_terms_with_zero_as_coefficient()
        assert s.m == 1 and set(s.c) == {0}
        s = s0 + t0
        assert s.alpha_c == {(0,): 1, (1,): 2, (2,): 3, (4,): 5}

    def test_polynomial_multiplication(self):
        # data for tests
        s0 = Polynomial(np.array([[0], [1], [2]]),
                        np.array([1, 2, 3]))
        t0 = Polynomial(np.array([[1]]),
                        np.array([1]))
        q0 = Polynomial(np.array([[5]]),
                        np.array([0]))
        # tests
        s = s0 * t0
        s.remove_terms_with_zero_as_coefficient()
        assert s.alpha_c == {(1,): 1, (2,): 2, (3,): 3, (0,): 0}
        s = t0 * s0
        s.remove_terms_with_zero_as_coefficient()
        assert s.alpha_c == {(1,): 1, (2,): 2, (3,): 3, (0,): 0}
        s = s0 * q0
        s.remove_terms_with_zero_as_coefficient()
        assert s.alpha_c == {(0,): 0}

    #
    # Test construction of Signomial representatives (in the constant case).
    #

if __name__ == '__main__':
    unittest.main()
