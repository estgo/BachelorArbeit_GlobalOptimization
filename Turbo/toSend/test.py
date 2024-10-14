'''import numpy as np
import matplotlib.pyplot as plt
import torch

# Define the function
def f(X, negate):
    result = (1.0 - X[0])**2 + 100.0 * ((X[1] - X[0]**2)**2)
    return result if not negate else -result

# Bounds for x1 and x2
bounds = torch.tensor([
    [-1.5, -0.5],  # Lower bounds for x1 and x2
    [1.5, 2.5]     # Upper bounds for x1 and x2
], dtype=torch.float)

# Extract lower and upper bounds for x1 and x2
lb, ub = bounds[0].numpy(), bounds[1].numpy()

# Generate a grid of values within the bounds
x1 = np.linspace(lb[0], ub[0], 10)
x2 = np.linspace(lb[1], ub[1], 10)
X1, X2 = np.meshgrid(x1, x2)

# Evaluate the function on the grid
Z = np.array([f([x1, x2], negate=False) for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
Z = Z.reshape(X1.shape)

# Create the contour plot
plt.contour(X1, X2, Z, levels=20, cmap='viridis')

# Add labels and a color bar
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Contour plot of f(X1, X2)')
plt.colorbar(label='f(X1, X2)')

plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt


# Step 1: Define the Branin function
def branin(x1, x2):
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


# Step 2: Generate a grid of values
x1 = np.linspace(-5, 15, 400)
x2 = np.linspace(-5, 20, 400)
X1, X2 = np.meshgrid(x1, x2)

# Step 3: Evaluate the function on the grid
Z = branin(X1, X2)

# Step 4: Create the contour plot
plt.contour(X1, X2, Z, levels=50, cmap='viridis')

# Add labels and a color bar
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Contour plot of the Branin function')
plt.colorbar(label='f(X1, X2)')

plt.show()
