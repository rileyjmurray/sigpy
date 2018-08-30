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
import unittest
import numpy as np
from sigpy import relaxations
from sigpy.signomials import Signomial


def primal_dual_vals(f, level):
    p = relaxations.sage_primal(f, level).solve(solver='ECOS')
    d = relaxations.sage_dual(f, level).solve(solver='ECOS')
    return [p, d]


class TestSAGERelaxations(unittest.TestCase):

    def test_unconstrained_sage_1(self):
        alpha = np.array([[0, 0],
                          [1, 0],
                          [0, 1],
                          [1, 1],
                          [0.5, 0],
                          [0, 0.5]])
        c = np.array([0, 3, 2, 1, -4, -2])
        s = Signomial(alpha, c)
        expected = [-1.83333, -1.746505595]
        pd0 = primal_dual_vals(s, 0)
        assert abs(pd0[0] - expected[0]) < 1e-4 and abs(pd0[1] - expected[0]) < 1e-4
        pd1 = primal_dual_vals(s, 1)
        assert abs(pd1[0] - expected[1]) < 1e-4 and abs(pd1[1] - expected[1]) < 1e-4

    def test_unconstrained_sage_2(self):
        alpha = np.array([[0, 0],
                          [1, 0],
                          [0, 1],
                          [1, 1],
                          [0.5, 1],
                          [1, 0.5]])
        c = np.array([0, 1, 1, 1.9, -2, -2])
        s = Signomial(alpha, c)
        expected = [-np.inf, -0.122211863]
        pd0 = primal_dual_vals(s, 0)
        assert pd0[0] == expected[0] and pd0[1] == expected[0]
        pd1 = primal_dual_vals(s, 1)
        assert abs(pd1[0] - expected[1]) < 1e-5 and abs(pd1[1] - expected[1]) < 1e-5

    def test_unconstrained_sage_3(self):
        s = Signomial({(1, 0, 0): 1,
                       (0, 1, 0): -1,
                       (0, 0, 1): -1})
        s = s ** 2
        expected = -np.inf
        pd0 = primal_dual_vals(s, 0)
        assert pd0[0] == expected and pd0[1] == expected
        pd1 = primal_dual_vals(s, 1)
        assert pd1[0] == expected and pd1[1] == expected

    def test_unconstrained_sage_4(self):
        s = Signomial({(3,): 1, (2,): -4, (1,): 7, (-1,): 1})
        expected = [3.464102, 4.60250026, 4.6217973]
        pds = [primal_dual_vals(s, ell) for ell in range(3)]
        for ell in range(3):
            assert abs(pds[ell][0] == expected[ell]) < 1e-5
            assert abs(pds[ell][1] == expected[ell]) < 1e-5

    def test_unconstrained_sage_5(self):
        alpha = np.array([[0., 1.],
                         [0.21, 0.08],
                         [0.16, 0.54],
                         [0., 0.],
                         [1., 0.],
                         [0.3, 0.58]])
        c = np.array([1., -57.75, -40.37, 33.94, 67.29, 38.28])
        s = Signomial(alpha, c)
        expected = [-24.054866, -21.31651]
        pd0 = primal_dual_vals(s, 0)
        assert abs(pd0[0] - expected[0]) < 1e-4 and abs(pd0[1] - expected[0]) < 1e-4
        pd1 = primal_dual_vals(s, 1)
        assert abs(pd1[0] - expected[1]) < 1e-4 and abs(pd1[1] - expected[1]) < 1e-4

    def test_unconstrained_sage_6(self):
        alpha = np.array([[0., 1.],
                         [0., 0.],
                         [0.52, 0.15],
                         [1., 0.],
                         [2., 2.],
                         [1.3, 1.38]])
        c = np.array([2.55, 0.31, -1.48, 0.85, 0.65, -1.73])
        s = Signomial(alpha, c)
        expected = [0.00354263, 0.13793126]
        pd0 = primal_dual_vals(s, 0)
        assert abs(pd0[0] - expected[0]) < 1e-6 and abs(pd0[1] - expected[0]) < 1e-6
        pd1 = primal_dual_vals(s, 1)
        assert abs(pd1[0] - expected[1]) < 1e-6 and abs(pd1[1] - expected[1]) < 1e-6

    def test_sage_feasibility(self):
        s = Signomial({(-1,): 1, (1,): -1})
        s = s ** 2
        s.remove_terms_with_zero_as_coefficient()
        status = relaxations.sage_feasibility(s).solve(solver='ECOS')
        assert status == 0
        s = s ** 2
        status = relaxations.sage_feasibility(s).solve(solver='ECOS')
        assert status == -np.inf

    def test_sage_multiplier_search(self):
        s = Signomial({(1,): 1, (-1,): -1}) ** 4
        s.remove_terms_with_zero_as_coefficient()
        val0 = relaxations.sage_multiplier_search(s, level=1).solve(solver='ECOS')
        assert val0 == -np.inf
        s_star = relaxations.sage_primal(s, level=1).solve(solver='ECOS')
        s = s - 0.5 * s_star
        val1 = relaxations.sage_multiplier_search(s, level=1).solve(solver='ECOS')
        assert val1 == 0

    def test_constrained_sage(self):
        s0 = Signomial({(10.2, 0, 0): 10, (0, 9.8, 0): 10, (0, 0, 8.2): 10})
        s1 = Signomial({(1.5089, 1.0981, 1.3419): -14.6794})
        s2 = Signomial({(1.0857, 1.9069, 1.6192): -7.8601})
        s3 = Signomial({(1.0459, 0.0492, 1.6245): 8.7838})
        f = s0 + s1 + s2 + s3
        g = Signomial({(10.2, 0, 0): -8,
                       (0, 9.8, 0): -8,
                       (0, 0, 8.2): -8,
                       (1.0857, 1.9069, 1.6192): -6.4,
                       (0, 0, 0): 1})
        gs = [g]
        expected = -0.6147
        actual = [relaxations.constrained_sage_primal(f, gs, p=0, q=1).solve(solver='ECOS'),
                  relaxations.constrained_sage_dual(f, gs, p=0, q=1).solve(solver='ECOS')]
        assert abs(actual[0] - expected) < 1e-4 and abs(actual[1] - expected) < 1e-4


if __name__ == '__main__':
    unittest.main()
