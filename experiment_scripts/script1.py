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

# First level
obj, constrs, variables = sigpy.sage_dual(s, level=0)
v = variables[0]
A, b, K, var_name_to_locs = cl.compile_system(constrs, [v])

# Second level
obj1, constrs1, variables1 = sigpy.sage_dual(s, level=1)
v1 = variables1[0]
A1, b1, K1, var_name_to_locs1 = cl.compile_system(constrs1, [v1])
