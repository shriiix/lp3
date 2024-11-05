import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the function and its gradient
def function(x):
    return (x + 3)**2

def gradient(x):
    return 2 * (x + 3)  # Derivative of (x + 3)^2

# Step 2: Initialize the starting point and parameters
x_start = 2  # Starting point
learning_rate = 0.1  # Learning rate
tolerance = 1e-6  # Tolerance for stopping
max_iterations = 1000  # Maximum number of iterations

# Gradient Descent Algorithm
x = x_start
for i in range(max_iterations):
    grad = gradient(x)  # Calculate the gradient
    x_new = x - learning_rate * grad  # Update the point

    # Check for convergence
    if abs(x_new - x) < tolerance:
        break
    x = x_new

# Result
print(f"Local minima found at x = {x:.6f}")
print(f"Function value at local minima: y = {function(x):.6f}")

# Step 3: Optional - Visualize the function and the convergence
x_vals = np.linspace(-6, 3, 100)
y_vals = function(x_vals)

plt.plot(x_vals, y_vals, label='y = (x + 3)²', color='blue')
plt.scatter(x, function(x), color='red', label='Local Minima', zorder=5)
plt.title('Gradient Descent to Find Local Minima')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()
