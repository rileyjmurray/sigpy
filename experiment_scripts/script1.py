import coniclifts as cl
import sigpy
import numpy as np

alpha = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1],
                  [0.5, 0],
                  [0, 0.5]])
c = np.array([0, 3, 2, 1, -4, -2])
s = sigpy.Signomial(alpha, c)

# initial low-level problem data
obj, constrs, variables = sigpy.sage_primal(s, level=0)
c, A, b, K, var_name_to_locs = cl.compile_problem(obj, constrs)

# ECOS problem data
G, h, cones, A_ecos, b_ecos = cl.ecos_format(A, b, K)

# Try to solve!
sol = ecos.solve(c, G, h, cones, A_ecos, b_ecos, verbose=True)
