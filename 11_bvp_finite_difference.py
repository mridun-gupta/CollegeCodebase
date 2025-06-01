# -----------------------------------------------------------------------------
# Copyright (C) 2025 Mridun Gupta
# Email: gmridun@gmail.com
# LinkedIn: https://www.linkedin.com/in/mridungupta
# GitHub: http://github.com/mridun-gupta
#
# This file is part of a project licensed under the GNU General Public License
# version 3 (GPLv3). You are free to use, modify, and distribute this code under
# the terms of GPLv3, provided that:
#   - Derivative works remain licensed under GPLv3.
#   - Proper attribution is given to the original author.
#   - The LICENSE file is included with any distribution.
#
# For full license details, see LICENSE or visit
# https://www.gnu.org/licenses/gpl-3.0.html.
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import math

# Step 1: Define problem parameters
# L: Length of the domain (0 to 1)
# Nx: Number of spatial grid points
L = 1.0
Nx = 100

# Step 2: Calculate spatial step size
# dx: Distance between grid points
dx = L / (Nx - 1)

# Step 3: Create spatial grid
# x: Array of spatial points from 0 to L
x = np.linspace(0, L, Nx)

# Step 4: Define the source term f(x)
# f(x) = sin(pi * x) for this example
def f(x):
    return math.sin(math.pi * x)

# Step 5: Initialize the right-hand side vector
# b: Vector to store f(x) at interior points
b = np.zeros(Nx - 2)
for i in range(Nx - 2):
    b[i] = f(x[i + 1]) * (dx ** 2)  # Scale by dx^2 for the finite difference equation

# Step 6: Construct the tridiagonal matrix A
# The finite difference scheme for -u'' = f(x) gives:
# (u[i+1] - 2*u[i] + u[i-1]) / dx^2 = f(x[i])
# This forms a tridiagonal system A * u = b
main_diag = 2.0 * np.ones(Nx - 2)  # Main diagonal: 2
off_diag = -1.0 * np.ones(Nx - 3)  # Off-diagonals: -1
A = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

# Step 7: Solve the linear system A * u = b
# u_interior: Solution at interior points (u[1] to u[Nx-2])
u_interior = np.linalg.solve(A, b)

# Step 8: Initialize full solution array
# u: Include boundary points u[0] = u[Nx-1] = 0
u = np.zeros(Nx)
u[1:Nx-1] = u_interior
u[0] = 0  # Boundary condition u(0) = 0
u[-1] = 0  # Boundary condition u(1) = 0

# Step 9: Compute analytical solution for comparison
# For f(x) = sin(pi * x), the analytical solution is u(x) = sin(pi * x) / (pi^2)
u_analytical = np.zeros(Nx)
for i in range(Nx):
    u_analytical[i] = math.sin(math.pi * x[i]) / (math.pi ** 2)

# Step 10: Plot the numerical and analytical solutions
plt.figure(figsize=(10, 6))
plt.plot(x, u, 'b-', label='Numerical Solution (Finite Difference)')
plt.plot(x, u_analytical, 'r--', label='Analytical Solution')
plt.title('BVP Solution: -u\'\' = sin(πx), u(0) = u(1) = 0')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True)
plt.savefig('bvp_finite_difference.png')