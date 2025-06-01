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

# Step 1: Define the data points for interpolation
# x_data: Known x-values (e.g., points where we know the function values)
# y_data: Known y-values (e.g., sin(x) at those x-values)
x_data = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # 5 points for interpolation
y_data = np.array([math.sin(x) for x in x_data])  # y = sin(x) at these points

# Step 2: Define the Lagrange interpolation function
def lagrange_interpolation(x, x_data, y_data):
    # Initialize the result
    result = 0.0
    n = len(x_data)  # Number of data points
    # Loop over each data point
    for i in range(n):
        # Compute the Lagrange basis polynomial L_i(x)
        term = y_data[i]  # Start with y_i
        for j in range(n):
            if j != i:
                # L_i(x) = product of (x - x_j)/(x_i - x_j) for j != i
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

# Step 3: Create points for plotting the interpolated function
# x_plot: Fine grid of x-values for smooth plotting
# y_plot: Interpolated y-values
x_plot = np.linspace(0, 2, 100)  # 100 points from 0 to 2
y_plot = np.array([lagrange_interpolation(x, x_data, y_data) for x in x_plot])

# Step 4: Compute the true function for comparison
# y_true: True values of sin(x) at x_plot points
y_true = np.sin(x_plot)

# Step 5: Print table for a subset of x_plot
table_data = []
for i in range(0, len(x_plot), 10):
    x_val = x_plot[i]
    interp_val = y_plot[i]
    true_val = y_true[i]
    error = abs(interp_val - true_val)
    table_data.append([round(x_val, 3), round(interp_val, 6), round(true_val, 6), round(error, 6)])

headers = ["x", "Interpolated y", "True sin(x)", "Absolute Error"]
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

# Step 6: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', label='Lagrange Interpolation')
plt.plot(x_plot, y_true, 'r--', label='True Function (sin(x))')
plt.scatter(x_data, y_data, color='black', label='Data Points', zorder=5)
plt.title('Lagrange Interpolation of sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('lagrange_interpolation.png')
plt.show()