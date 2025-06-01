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
from tabulate import tabulate

# Step 1: Define problem parameters
# T: Total time (0 to 1)
# Nt: Number of time steps
# u0: Initial condition u(0) = 1
T = 1.0
Nt = 100
u0 = 1.0

# Step 2: Calculate time step size
# dt: Time step size
dt = T / Nt

# Step 3: Create time grid
# t: Array of time points from 0 to T
t = np.linspace(0, T, Nt + 1)

# Step 4: Define the ODE function
# f(t, u) = -2u + t (the derivative du/dt)
def f(t, u):
    return -2.0 * u + t

# Step 5: Initialize solution array
# u: Array to store solution at each time step
u = np.zeros(Nt + 1)
u[0] = u0  # Set initial condition

# Step 6: Modified Euler method time-stepping loop
# Update: u[n+1] = u[n] + (dt/2) * [f(t[n], u[n]) + f(t[n+1], u[n] + dt * f(t[n], u[n]))]
for n in range(Nt):
    # Compute slope at current point
    k1 = f(t[n], u[n])
    # Predict u at next point using Euler step
    u_predict = u[n] + dt * k1
    # Compute slope at predicted point
    k2 = f(t[n + 1], u_predict)
    # Update using average of slopes
    u[n + 1] = u[n] + (dt / 2) * (k1 + k2)

# Step 7: Compute analytical solution for comparison
# For du/dt = -2u + t, u(0) = 1, the analytical solution is:
# u(t) = (1/4)t - (1/8) + (9/8)e^(-2t)
u_analytical = np.zeros(Nt + 1)
for i in range(Nt + 1):
    u_analytical[i] = (1/4) * t[i] - (1/8) + (9/8) * math.exp(-2 * t[i])

# Step 8: Plot the numerical and analytical solutions
plt.figure(figsize=(10, 6))
plt.plot(t, u, 'b-', label='Numerical Solution (Modified Euler Method)')
plt.plot(t, u_analytical, 'r--', label='Analytical Solution')
plt.title('IVP Solution: du/dt = -2u + t, u(0) = 1')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend()
plt.grid(True)
plt.savefig('ivp_modified_euler_method.png')
plt.show()

# Step 9: Print table with tabulate
indices = range(0, Nt + 1, 10)
table_data = []
for i in indices:
    time_val = f"{t[i]:.2f}"
    num_val = f"{u[i]:.6f}"
    ana_val = f"{u_analytical[i]:.6f}"
    abs_err = f"{abs(u[i] - u_analytical[i]):.6e}"
    table_data.append([time_val, num_val, ana_val, abs_err])
headers = ["t", "Modified Euler", "Analytical", "Absolute Error"]
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))