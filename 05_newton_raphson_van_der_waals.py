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
from tabulate import tabulate

# Step 1: Define problem parameters
# P: Pressure (atm)
# T: Temperature (K)
# a: Van der Waals constant a (L^2 atm/mol^2)
# b: Van der Waals constant b (L/mol)
# R: Gas constant (L atm/(mol K))
# V0: Initial guess for molar volume (L/mol)
# tol: Tolerance for convergence
# max_iter: Maximum number of iterations
P = 1.0
T = 300.0
a = 1.39
b = 0.039
R = 0.0821
V0 = 24.0  # Initial guess based on ideal gas law: V ≈ RT/P
tol = 1e-6
max_iter = 100

# Step 2: Define the function f(V) = P*V + a/V - a*b/V^2 - P*b - R*T
def f(V):
    if V <= b:
        return float('inf')  # Prevent invalid V values (V > b)
    return P * V + a / V - a * b / (V**2) - P * b - R * T

# Step 3: Define the derivative f'(V)
# f'(V) = P - a/V^2 + 2*a*b/V^3
def f_prime(V):
    if V <= b:
        return float('inf')  # Prevent invalid V values
    return P - a / (V**2) + 2 * a * b / (V**3)

# Step 4: Initialize lists to store iteration history for plotting
V_values = [V0]  # Store V approximations
f_values = [f(V0)]  # Store f(V) values

# Step 5: Newton-Raphson method loop
V_n = V0
for i in range(max_iter):
    # Compute f(V_n) and f'(V_n)
    fx_n = f(V_n)
    fpx_n = f_prime(V_n)

    # Check if derivative is too small to avoid division issues
    if abs(fpx_n) < 1e-10:
        print("Error: Derivative is too small, method fails!")
        break

    # Update V_n using Newton-Raphson formula
    V_next = V_n - fx_n / fpx_n

    # Ensure V_next is physically valid (V > b)
    if V_next <= b:
        print("Error: Invalid molar volume encountered (V <= b)!")
        break

    # Store new values for plotting
    V_values.append(V_next)
    f_values.append(f(V_next))

    # Check convergence
    if abs(f(V_next)) < tol or abs(V_next - V_n) < tol:
        print(f"Molar volume found at V = {V_next:.6f} L/mol after {i+1} iterations")
        break

    V_n = V_next
else:
    print(f"Did not converge within {max_iter} iterations")

# Step 6: Create points for plotting the function
V_plot = np.linspace(max(b + 0.01, 20), 28, 100)  # Range around expected V
f_plot = np.array([f(V) for V in V_plot])

# Step 7: Plot the function and iterations
plt.figure(figsize=(10, 6))
plt.plot(V_plot, f_plot, 'b-', label='f(V) = P*V + a/V - a*b/V^2 - P*b - RT')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)  # x-axis
plt.scatter(V_values, f_values, color='red', label='Newton-Raphson Iterations', zorder=5)
plt.plot(V_values, f_values, 'r--', alpha=0.5)  # Connect iteration points
plt.title('Newton-Raphson Method for Van der Waals Equation')
plt.xlabel('V (Molar Volume, L/mol)')
plt.ylabel('f(V)')
plt.legend()
plt.grid(True)
plt.savefig('newton_raphson_vander_waals.png')
plt.show()

# Step 8: Print iteration table for Van der Waals
table_data = []
for i, (V_i, fV_i) in enumerate(zip(V_values, f_values)):
    if i == 0:
        delta = 0.0
    else:
        delta = abs(V_values[i] - V_values[i - 1])
    table_data.append([i, round(V_i, 8), round(fV_i, 8), round(delta, 8)])
headers = ["Iteration", "V (Approx.)", "f(V)", "Change in V"]
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))