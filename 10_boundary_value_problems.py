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

# Step 1: Define the function and its derivative
# f(x) = x^3 - x - 2
def f(x):
    return x**3 - x - 2

# f'(x) = 3x^2 - 1 (derivative of f(x))
def f_prime(x):
    return 3 * x**2 - 1

# Step 2: Define problem parameters
# x0: Initial guess for the root
# tol: Tolerance for convergence
# max_iter: Maximum number of iterations
x0 = 1.5
tol = 1e-6
max_iter = 100

# Step 3: Initialize lists to store iteration history for plotting
x_values = [x0]  # Store x approximations
f_values = [f(x0)]  # Store f(x) values

# Step 4: Newton-Raphson method loop
x_n = x0
for i in range(max_iter):
    # Compute f(x_n) and f'(x_n)
    fx_n = f(x_n)
    fpx_n = f_prime(x_n)

    # Check if derivative is zero to avoid division by zero
    if abs(fpx_n) < 1e-10:
        print("Error: Derivative is zero, method fails!")
        break

    # Update x_n using Newton-Raphson formula
    x_next = x_n - fx_n / fpx_n

    # Store new values for plotting
    x_values.append(x_next)
    f_values.append(f(x_next))

    # Check convergence
    if abs(f(x_next)) < tol or abs(x_next - x_n) < tol:
        print(f"Root found at x = {x_next:.6f} after {i+1} iterations")
        break

    x_n = x_next
else:
    print(f"Did not converge within {max_iter} iterations")

# Step 5: Create points for plotting the function
x_plot = np.linspace(0.5, 2.5, 100)  # Range around the root
y_plot = f(x_plot)

# Step 6: Plot the function and iterations
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', label='f(x) = x^3 - x - 2')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)  # x-axis
plt.scatter(x_values, f_values, color='red', label='Newton-Raphson Iterations', zorder=5)
plt.plot(x_values, f_values, 'r--', alpha=0.5)  # Connect iteration points
plt.title('Newton-Raphson Method for Root Finding')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.savefig('newton_raphson_method.png')