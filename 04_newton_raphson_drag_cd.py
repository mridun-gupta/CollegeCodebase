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
# a, b: Constants in the equation 1/Cd = a*log(Re*Cd) + b
# Re: Reynolds number
# Cd0: Initial guess for Cd
# tol: Tolerance for convergence
# max_iter: Maximum number of iterations
a = 2.5
b = 0.5
Re = 10000
Cd0 = 0.1
tol = 1e-6
max_iter = 100

# Step 2: Define the function f(Cd) = 1/Cd - a*log(Re*Cd) - b
def f(Cd):
    if Cd <= 0:
        return float('inf')  # Prevent invalid Cd values
    return (1 / Cd) - a * math.log10(Re * Cd) - b

# Step 3: Define the derivative f'(Cd)
# f'(Cd) = -1/Cd^2 - a/(Cd * ln(10))
def f_prime(Cd):
    if Cd <= 0:
        return float('inf')  # Prevent invalid Cd values
    return -1 / (Cd**2) - a / (Cd * math.log(10))

# Step 4: Initialize lists to store iteration history for plotting
Cd_values = [Cd0]  # Store Cd approximations
f_values = [f(Cd0)]  # Store f(Cd) values

# Step 5: Newton-Raphson method loop
Cd_n = Cd0
for i in range(max_iter):
    # Compute f(Cd_n) and f'(Cd_n)
    fx_n = f(Cd_n)
    fpx_n = f_prime(Cd_n)

    # Check if derivative is zero or very small to avoid division issues
    if abs(fpx_n) < 1e-10:
        print("Error: Derivative is too small, method fails!")
        break

    # Update Cd_n using Newton-Raphson formula
    Cd_next = Cd_n - fx_n / fpx_n

    # Ensure Cd_next is positive (valid drag coefficient)
    if Cd_next <= 0:
        print("Error: Negative or zero Cd encountered!")
        break

    # Store new values for plotting
    Cd_values.append(Cd_next)
    f_values.append(f(Cd_next))

    # Check convergence
    if abs(f(Cd_next)) < tol or abs(Cd_next - Cd_n) < tol:
        print(f"Root found at Cd = {Cd_next:.6f} after {i+1} iterations")
        break

    Cd_n = Cd_next
else:
    print(f"Did not converge within {max_iter} iterations")

# Step 6: Create points for plotting the function
Cd_plot = np.linspace(0.01, 0.2, 100)  # Range around expected Cd
f_plot = np.array([f(Cd) for Cd in Cd_plot])

# Step 7: Plot the function and iterations
plt.figure(figsize=(10, 6))
plt.plot(Cd_plot, f_plot, 'b-', label='f(Cd) = 1/Cd - a*log10(Re*Cd) - b')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)  # x-axis
plt.scatter(Cd_values, f_values, color='red', label='Newton-Raphson Iterations', zorder=5)
plt.plot(Cd_values, f_values, 'r--', alpha=0.5)  # Connect iteration points
plt.title('Newton-Raphson Method for Drag Coefficient')
plt.xlabel('Cd (Drag Coefficient)')
plt.ylabel('f(Cd)')
plt.legend()
plt.grid(True)
plt.savefig('newton_raphson_drag_coefficient.png')