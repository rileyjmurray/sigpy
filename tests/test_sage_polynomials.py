import unittest
import cvxpy
from cvxpy.expressions.expression import Expression
import numpy as np
from sigpy.polys.polynomials import Polynomial
from sigpy.polys import sage_poly as sage


def primal_dual_unconstrained(p, level, verbose=False):
    res1 = sage.sage_poly_primal(p, level=level).solve(solver='ECOS', max_iters=10000, verbose=verbose)
    res2 = sage.sage_poly_dual(p, level=level).solve(solver='ECOS', max_iters=10000, verbose=verbose)
    return [res1, res2]


class TestSagePolynomials(unittest.TestCase):

    #
    #   Test non-constant signomal representatives
    #

    def test_sigrep_1(self):
        p = Polynomial({(0, 0): -1, (1, 2): 1, (2, 2): 10})
        gamma = cvxpy.Variable(shape=(), name='gamma')
        p -= gamma
        sr, sr_cons = p.sig_rep
        # Even though there is a Variable in p.c, no auxiliary
        # variables should have been introduced by defining this
        # signomial representative.
        assert len(sr_cons) == 0
        count_nonconstants = 0
        for i, ci in enumerate(sr.c):
            if isinstance(ci, Expression):
                assert len(ci.variables()) == 1
                count_nonconstants += 1
                assert ci.variables()[0].name() == 'gamma'
                gamma.value = 42
                assert ci.value == -43
            elif sr.alpha[i, 0] == 1 and sr.alpha[i, 1] == 2:
                assert ci == -1
            elif sr.alpha[i, 0] == 2 and sr.alpha[i, 1] == 2:
                assert ci == 10
            else:
                assert False
        assert count_nonconstants == 1

    def test_sigrep_2(self):
        c33 = cvxpy.Variable(shape=(), name='c33')
        p = Polynomial({(0, 0): 0, (1, 1): -1, (3, 3): c33})
        sr, sr_cons = p.sig_rep
        assert len(sr_cons) == 2
        var_names = set(v.name() for v in sr_cons[0].variables())
        var_names.union(set(v.name() for v in sr_cons[1].variables()))
        for v in var_names:
            assert v == 'c33' or v[:-3] == 'sig_rep_coeff'
        assert sr.alpha_c[(1, 1)] == -1

    #
    # Test unconstrained relaxations
    #

    def test_unconstrained_1(self):
        alpha = np.array([[0, 0], [1, 1], [2, 2], [0, 2], [2, 0]])
        c = np.array([1, -3, 1, 4, 4])
        p = Polynomial(alpha, c)
        res0 = primal_dual_unconstrained(p, level=0)
        assert abs(res0[0] - res0[1]) <= 1e-6
        c = np.array([1, 3, 1, 4, 4])
        # ^ change the sign in a way that won't affect the signomial
        # representative
        p = Polynomial(alpha, c)
        res1 = primal_dual_unconstrained(p, level=0)
        assert abs(res1[0] - res1[1]) <= 1e-6
        assert abs(res0[0] - res0[1]) <= 1e-6

    def test_unconstrained_2(self):
        p = Polynomial({(0, 0): 1,
                        (2, 6): 3,
                        (6, 2): 2,
                        (2, 2): 6,
                        (1, 2): -1,
                        (2, 1): 2,
                        (3, 3): -3})
        res0 = primal_dual_unconstrained(p, level=0)
        assert abs(res0[0] - res0[1]) <= 1e-6
        res1 = primal_dual_unconstrained(p, level=1)
        assert abs(res1[0] - res1[1]) <= 1e-5
        assert res1[0] - res0[0] > 1e-1



if __name__ == '__main__':
    unittest.main()
