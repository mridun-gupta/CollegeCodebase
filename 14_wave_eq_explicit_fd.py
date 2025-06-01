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
# L: Length of the spatial domain (e.g., length of string)
# T: Total simulation time
# c: Wave speed
# Nx: Number of spatial grid points
# Nt: Number of time steps
L = 1.0
T = 0.5
c = 1.0
Nx = 100
Nt = 200

# Step 2: Calculate step sizes
# dx: Spatial step size
# dt: Time step size
# r: Courant number (c * dt / dx), must be <= 1 for stability
dx = L / (Nx - 1)
dt = T / Nt
r = c * dt / dx
# Check stability condition
if r > 1:
    print(f"Warning: r = {r:.3f} > 1, solution may be unstable!")
else:
    print(f"Courant number: r = {r:.3f}")

# Step 3: Create spatial and time grids
# x: Spatial points from 0 to L
# t: Time points from 0 to T
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt + 1)

# Step 4: Initialize solution array
# u: 2D array for displacement u(x,t) at each time step and position
u = np.zeros((Nt + 1, Nx))

# Step 5: Set initial condition for displacement
# u(x,0) = sin(pi * x / L) at t=0
for i in range(Nx):
    u[0, i] = math.sin(math.pi * x[i] / L)

# Step 6: Set initial condition for velocity
# du/dt(x,0) = 0, approximated for first time step
# Use central difference: u[1,i] = u[-1,i] to enforce zero initial velocity
# For explicit scheme, compute u[1,i] using the wave equation
for i in range(1, Nx - 1):
    u[1, i] = u[0, i] + 0.5 * (r**2) * (u[0, i + 1] - 2 * u[0, i] + u[0, i - 1])

# Step 7: Set boundary conditions
# u(0,t) = u(L,t) = 0 for all time steps (fixed ends)
u[:, 0] = 0
u[:, -1] = 0

# Step 8: Explicit finite difference time-stepping loop
# Update: u[n+1,i] = 2*u[n,i] - u[n-1,i] + r^2 * (u[n,i+1] - 2*u[n,i] + u[n,i-1])
for n in range(1, Nt):
    for i in range(1, Nx - 1):
        u[n + 1, i] = (2 * u[n, i] - u[n - 1, i] +
                       (r**2) * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]))

# Step 9: Plot the results
plt.figure(figsize=(10, 6))
# Plot displacement at selected time steps: t=0, t=T/4, t=T/2, t=T
times_to_plot = [0, Nt//4, Nt//2, Nt]
for n in times_to_plot:
    plt.plot(x, u[n, :], label=f't={t[n]:.3f}')
plt.title('1D Wave Equation Solution using Explicit Finite Difference')
plt.xlabel('x')
plt.ylabel('Displacement u(x,t)')
plt.legend()
plt.grid(True)
plt.savefig('wave_equation_explicit.png')