import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your setup

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Base State class
class State:
    def __init__(self):
        self.visible = True  # Visibility property for all states

    def draw(self, ax):
        raise NotImplementedError("Subclasses should implement this!")


# Generating State: draws multiple points in a specific color
class GeneratingState(State):
    def __init__(self, X, color='blue'):
        super().__init__()
        self.X = X  # Multidimensional array of points
        self.color = color

    def draw(self, ax):
        if self.visible:
            for point in self.X:
                ax.scatter(point[0], point[1], color=self.color)

# Select State: draws the best point in a different color
class SelectState(State):
    def __init__(self, X_best, color='red', marker='*'):
        super().__init__()
        self.X_best = X_best  # Best X point (multidimensional)
        self.color = color
        self.marker = marker

    def draw(self, ax):
        if self.visible:
            ax.scatter(self.X_best[0], self.X_best[1], color=self.color, marker=self.marker, s=100)


# Update Points State: draws points with a different color
class UpdatePointsState(GeneratingState):
    def __init__(self, X, color='green'):
        super().__init__(X, color)


# Update Rectangle State: draws a rectangle around the best point
class UpdateRectangleState(State):
    def __init__(self, X_best, length, color='purple'):
        super().__init__()
        self.X_best = X_best
        self.length = length
        self.color = color

    def draw(self, ax):
        if self.visible:
            x, y = self.X_best
            rect = plt.Rectangle((x - self.length, y - self.length), 2 * self.length, 2 * self.length,
                                 edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)


# Example usage for animation
fig, ax = plt.subplots()
ax.set_xlim(0, 11)
ax.set_ylim(0, 11)

# Create several states
X_test = [[1, 10], [1, 1], [10, 10], [10, 1], [5, 5],
              [7, 6], [6, 7], [6, 6], [6.5, 6.5], [6.5, 7],
              [4, 4], [3, 3], [3, 4], [4, 3], [5.5, 5.5]]

Y_test = np.array([50, 49, 70, 88, 5, 80, 90, 70, 30, 55, 100, 80, 200, 300, 4])
boundaries = [[0,0], [10,10]]
batch_size_test = 5
length_test = 2.5  # Length of the rectangle sides


states = [
    GeneratingState([[1, 10], [1, 1], [10, 10], [10, 1], [5, 5]]),
    SelectState([5, 5]),
    UpdatePointsState([[1, 10], [1, 1], [10, 10], [10, 1], [5, 5]]),
    UpdateRectangleState([5, 5], 2.5),

    GeneratingState([[7, 6], [6, 7], [6, 6], [6.5, 6.5], [6.5, 7]]),
    SelectState([5, 5]),
    UpdatePointsState([[7, 6], [6, 7], [6, 6], [6.5, 6.5], [6.5, 7]]),
    UpdateRectangleState([5, 5], 2.5),

    GeneratingState([[4, 4], [3, 3], [3, 4], [4, 3], [5.5, 5.5]]),
    SelectState([5.5, 5.5]),
    UpdatePointsState([[4, 4], [3, 3], [3, 4], [4, 3], [5.5, 5.5]]),
    UpdateRectangleState([5.5, 5.5], 5)
]

# Function to update the plot for animation
def update(frame):
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)

    # Draw the current state
    states[frame].draw(ax)

# Create an animation: update every 500ms
ani = FuncAnimation(fig, update, frames=len(states), interval=1000, repeat=False)

plt.show()
