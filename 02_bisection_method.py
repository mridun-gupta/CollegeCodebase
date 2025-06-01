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

# Step 1: Define the function whose root we want to find
# f(x) = x^3 - x - 2
def f(x):
    return x**3 - x - 2

# Step 2: Define problem parameters
# a, b: Initial interval [a, b] where f(a) * f(b) < 0 (root exists)
# tol: Tolerance for convergence
# max_iter: Maximum number of iterations
a = 1.0
b = 2.0
tol = 1e-6
max_iter = 100

# Step 3: Check if a root exists in [a, b]
# Verify that f(a) * f(b) < 0
if f(a) * f(b) >= 0:
    print("Error: f(a) and f(b) must have opposite signs!")
    exit()

# Step 4: Initialize lists to store iteration history for plotting
x_values = []  # Store x approximations
f_values = []  # Store f(x) values

# Step 5: Bisection method loop
for i in range(max_iter):
    # Compute the midpoint of the interval
    x_n = (a + b) / 2
    x_values.append(x_n)
    f_values.append(f(x_n))

    # Check if the solution is close enough
    if abs(f(x_n)) < tol or (b - a) / 2 < tol:
        print(f"Root found at x = {x_n:.6f} after {i+1} iterations")
        break

    # Update the interval [a, b] based on the sign of f(x_n)
    if f(x_n) * f(a) < 0:
        b = x_n  # Root is in [a, x_n]
    else:
        a = x_n  # Root is in [x_n, b]

else:
    print(f"Did not converge within {max_iter} iterations")

# Step 6: Create points for plotting the function
x_plot = np.linspace(0.5, 2.5, 100)  # Range around the root
y_plot = f(x_plot)

# Step 7: Plot the function and iterations
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', label='f(x) = x^3 - x - 2')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)  # x-axis
plt.scatter(x_values, f_values, color='red', label='Bisection Iterations', zorder=5)
plt.plot(x_values, f_values, 'r--', alpha=0.5)  # Connect iteration points
plt.title('Bisection Method for Root Finding')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.savefig('bisection_method.png')