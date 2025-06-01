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

# Step 1: Define the ODE function
# f(t, u) = -2u + t (the derivative du/dt)
def f(t, u):
    return -2.0 * u + t

# Step 2: Define problem parameters
# T: Total time (0 to 1)
# Nt: Number of time steps
# u0: Initial condition u(0) = 1
T = 1.0
Nt = 100
u0 = 1.0

# Step 3: Calculate time step size
# dt: Time step size
dt = T / Nt

# Step 4: Create time grid
# t: Array of time points from 0 to T
t = np.linspace(0, T, Nt + 1)

# Step 5: Define RK2 solver function
def rk2_step(t_n, u_n, dt, method):
    # Define Butcher tableau coefficients for different RK2 methods
    methods = {
        'heun': {'c2': 1.0, 'a21': 1.0, 'b1': 0.5, 'b2': 0.5},
        'midpoint': {'c2': 0.5, 'a21': 0.5, 'b1': 0.0, 'b2': 1.0},
        'ralston': {'c2': 2/3, 'a21': 2/3, 'b1': 0.25, 'b2': 0.75},
        'classical': {'c2': 1.0, 'a21': 1.0, 'b1': 0.0, 'b2': 1.0}
    }
    coeffs = methods[method]

    # Compute k1 = f(t_n, u_n)
    k1 = f(t_n, u_n)
    # Compute k2 = f(t_n + c2*dt, u_n + a21*k1*dt)
    k2 = f(t_n + coeffs['c2'] * dt, u_n + coeffs['a21'] * k1 * dt)
    # Update: u_{n+1} = u_n + dt * (b1*k1 + b2*k2)
    u_next = u_n + dt * (coeffs['b1'] * k1 + coeffs['b2'] * k2)
    return u_next

# Step 6: Initialize solution arrays for each method
u_heun = np.zeros(Nt + 1)
u_midpoint = np.zeros(Nt + 1)
u_ralston = np.zeros(Nt + 1)
u_classical = np.zeros(Nt + 1)
u_heun[0] = u_midpoint[0] = u_ralston[0] = u_classical[0] = u0

# Step 7: Time-stepping loop for each RK2 method
for n in range(Nt):
    u_heun[n + 1] = rk2_step(t[n], u_heun[n], dt, 'heun')
    u_midpoint[n + 1] = rk2_step(t[n], u_midpoint[n], dt, 'midpoint')
    u_ralston[n + 1] = rk2_step(t[n], u_ralston[n], dt, 'ralston')
    u_classical[n + 1] = rk2_step(t[n], u_classical[n], dt, 'classical')

# Step 8: Compute analytical solution for comparison
# For du/dt = -2u + t, u(0) = 1, the analytical solution is:
# u(t) = (1/4)t - (1/8) + (9/8)e^(-2t)
u_analytical = np.zeros(Nt + 1)
for i in range(Nt + 1):
    u_analytical[i] = (1/4) * t[i] - (1/8) + (9/8) * math.exp(-2 * t[i])

# Step 9: Plot the numerical and analytical solutions
plt.figure(figsize=(10, 6))
plt.plot(t, u_heun, 'b-', label='Heun’s Method')
plt.plot(t, u_midpoint, 'g-', label='Midpoint Method')
plt.plot(t, u_ralston, 'c-', label='Ralston’s Method')
plt.plot(t, u_classical, 'm-', label='Classical RK2')
plt.plot(t, u_analytical, 'r--', label='Analytical Solution')
plt.title('IVP Solution: du/dt = -2u + t, u(0) = 1 (RK2 Methods)')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend()
plt.grid(True)
plt.savefig('rk2_methods.png')