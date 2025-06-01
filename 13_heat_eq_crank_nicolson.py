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
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import math
from tabulate import tabulate

# Step 1: Define problem parameters
# L: Length of the rod (in meters)
# T: Total simulation time (in seconds)
# Nx: Number of spatial grid points
# Nt: Number of time steps
# alpha: Thermal diffusivity (m^2/s)
L = 1.0
T = 0.1
Nx = 50
Nt = 100
alpha = 0.01

# Step 2: Calculate step sizes
# dx: Spatial step size (distance between grid points)
# dt: Time step size
# r: Stability parameter for Crank-Nicolson (alpha * dt / dx^2)
dx = L / (Nx - 1)
dt = T / Nt
r = alpha * dt / (dx ** 2)

# Step 3: Create spatial grid
# x: Array of spatial points from 0 to L
x = np.linspace(0, L, Nx)

# Step 4: Initialize temperature array
# u: 2D array to store temperature at each time step and position
# Rows represent time steps, columns represent spatial points
u = np.zeros((Nt + 1, Nx))

# Step 5: Set initial condition
# u(x,0) = sin(pi * x / L) at t=0 for all spatial points
for i in range(Nx):
    u[0, i] = math.sin(math.pi * x[i] / L)

# Step 6: Set boundary conditions
# u(0,t) = u(L,t) = 0 for all time steps (Dirichlet conditions)
u[:, 0] = 0
u[:, -1] = 0

# Step 7: Construct tridiagonal matrices for Crank-Nicolson
# The method solves A * u^{n+1} = B * u^n
# A: Matrix for implicit part
# B: Matrix for explicit part
# Main diagonals and off-diagonals are defined based on r
main_diag_A = 1 + r
off_diag_A = -r / 2
main_diag_B = 1 - r
off_diag_B = r / 2

# Step 8: Create sparse matrices A and B
# Use scipy.sparse.diags to create tridiagonal matrices
# Only interior points (Nx-2) are solved, as boundaries are fixed
A = diags([off_diag_A, main_diag_A, off_diag_A], [-1, 0, 1], shape=(Nx-2, Nx-2)).tocsr()
B = diags([off_diag_B, main_diag_B, off_diag_B], [-1, 0, 1], shape=(Nx-2, Nx-2)).tocsr()

# Step 9: Time-stepping loop
# For each time step, compute u^{n+1} from u^n
for n in range(0, Nt):
    # Compute right-hand side: B * u^n for interior points
    b = B.dot(u[n, 1:-1])
    # Solve linear system A * u^{n+1} = b for interior points
    u[n+1, 1:-1] = spsolve(A, b)

# Step 10: Plot the results
# Create a figure to visualize temperature distribution
plt.figure(figsize=(10, 6))
# Plot temperature at selected time steps: t=0, t=T/4, t=T/2, t=T
times_to_plot = [0, Nt//4, Nt//2, Nt]
for n in times_to_plot:
    plt.plot(x, u[n, :], label=f't={n*dt:.3f}')
# Add title, labels, legend, and grid to the plot
plt.title('Heat Equation Solution using Crank-Nicolson Method')
plt.xlabel('x')
plt.ylabel('Temperature u(x,t)')
plt.legend()
plt.grid(True)
# Save the plot to a file
plt.savefig('heat_equation_crank_nicolson.png')
plt.show()

# Step 11: Print table with tabulate
# Select spatial indices to display (e.g., every 10th point)
spatial_indices = list(range(0, Nx, 10))
# Prepare headers: x values + selected times
headers = ["x \\ t"] + [f"{n*dt:.3f}" for n in times_to_plot]
# Prepare rows: for each spatial position, temperature values at chosen times
table_data = []
for i in spatial_indices:
    row = [f"{x[i]:.2f}"]  # spatial location
    for n in times_to_plot:
        row.append(f"{u[n, i]:.6f}")
    table_data.append(row)
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))