import unittest
import cvxpy
from cvxpy.expressions.expression import Expression
import numpy as np
from sigpy.polys.polynomials import Polynomial, standard_monomials
from sigpy.polys import sage_poly as sage


def primal_dual_unconstrained(p, level, verbose=False, solver=None):
    if solver is None:
        solver = 'ECOS'
    res1 = sage.sage_poly_primal(p, level=level).solve(solver=solver, verbose=verbose)
    res2 = sage.sage_poly_dual(p, level=level).solve(solver=solver, verbose=verbose)
    return [res1, res2]


def primal_dual_constrained(f, gs, p, q, verbose=False, solver=None):
    if solver is None:
        res1 = sage.constrained_sage_poly_primal(f, gs, p, q).solve(solver='ECOS', max_iters=1000, verbose=verbose)
        res2 = sage.constrained_sage_poly_dual(f, gs, p, q).solve(solver='ECOS', max_iters=1000, verbose=verbose)
    else:
        res1 = sage.constrained_sage_poly_primal(f, gs, p, q).solve(solver=solver, verbose=verbose)
        res2 = sage.constrained_sage_poly_dual(f, gs, p, q).solve(solver=solver, verbose=verbose)
    return [res1, res2]


class TestSagePolynomials(unittest.TestCase):

    #
    #   Test non-constant signomial representatives
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
    #   Test unconstrained relaxations
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
        # an example from
        # https://arxiv.org/abs/1808.08431
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
        assert res1[0] - res0[0] > 1e-2

    def test_unconstrained_3(self):
        # an example from
        # http://homepages.laas.fr/henrion/papers/gloptipoly3.pdf
        p = Polynomial({(0, 0): 0,
                        (2, 0): 4,
                        (1, 1): 1,
                        (0, 2): -4,
                        (4, 0): -2.1,
                        (0, 4): 4,
                        (6, 0): 1.0 / 3.0})
        # We'll see that level=0 has a decent bound, and level=1 is nearly
        # optimal. ECOS is unable to solver level=2 due to conditioning problems,
        # but MOSEK solves level=2 without any trouble; the solution when level=2
        # is globally optimal.
        res0 = primal_dual_unconstrained(p, level=0)
        assert abs(res0[0] - res0[1]) <= 1e-6
        res1 = primal_dual_unconstrained(p, level=1)
        assert abs(res1[0] - res1[1]) <= 1e-6

    #
    #   Test constrained relaxations
    #

    def test_constrained_1(self):
        # an example from page 16 of
        # http://homepages.laas.fr/henrion/papers/gloptipoly3.pdf
        # --- which is itself borrowed from somewhere else.
        f = Polynomial({(1, 0, 0): -2,
                        (0, 1, 0): 1,
                        (0, 0, 1): -1})
        # Constraints over more than one variable
        g1 = Polynomial({(0, 0, 0): 24,
                         (1, 0, 0): -20,
                         (0, 1, 0): 9,
                         (0, 0, 1): -13,
                         (2, 0, 0): 4,
                         (1, 1, 0): -4,
                         (1, 0, 1): 4,
                         (0, 2, 0): 2,
                         (0, 1, 1): -2,
                         (0, 0, 2): 2})
        g2 = Polynomial({(1, 0, 0): -1,
                         (0, 1, 0): -1,
                         (0, 0, 1): -1,
                         (0, 0, 0): 4})
        g3 = Polynomial({(0, 1, 0): -3,
                         (0, 0, 1): -1,
                         (0, 0, 0): 6})
        # Bound constraints on x_1
        g4 = Polynomial({(1, 0, 0): 1})
        g5 = Polynomial({(1, 0, 0): -1,
                         (0, 0, 0): 2})
        # Bound constraints on x_2
        g6 = Polynomial({(0, 1, 0): 1})
        # Bound constraints on x_3
        g7 = Polynomial({(0, 0, 1): 1})
        g8 = Polynomial({(0, 0, 1): -1,
                         (0, 0, 0): 3})
        # Assemble!
        gs = [g1, g2, g3, g4, g5, g6, g7, g8]
        res = primal_dual_constrained(f, gs, 0, 1)
        assert abs(res[0] - (-6)) <= 1e-6
        assert abs(res[1] - (-6)) <= 1e-6
        # ^ incidentally, this is the same as gloptipoly3 !
        res1 = primal_dual_constrained(f, gs, 1, 1)
        assert abs(res1[0] - (-5.7)) <= 0.02
        assert abs(res1[0] - (-5.7)) <= 0.02

    #
    #   Test multiplier search
    #

    def test_multiplier_search(self):
        x = standard_monomials(3)
        """
        The polynomial
            p = (np.sum(x)) ** 2 + a * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        is PSD for a >= 0, and positive definite for a > 0.

        When using the MOSEK solver, we can certify nonnegativity with SAGE polynomials for
        a >= 0.28 when "level == 1", and a >= 0.15 when "level == 2".

        When using ECOS, we run into numerical issues for the smallest choices of "a". So
        instead we test with a == 0.5 and a == 0.25.
        """
        p = (np.sum(x)) ** 2 + 0.5 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        res1 = sage.sage_poly_multiplier_search(p, level=1).solve(solver='ECOS', max_iters=10000)
        assert abs(res1) < 1e-8
        p -= 0.25 * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        res2 = sage.sage_poly_multiplier_search(p, level=2).solve(solver='ECOS', max_iters=10000)
        assert abs(res2) < 1e-8

if __name__ == '__main__':
    unittest.main()
