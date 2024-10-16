import matplotlib

matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your setup

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import scbo
from typing import List, Tuple
import torch
from torch import Tensor
import math

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Change to 'cuda' if using GPU
}



# Base State class
class State:
    def __init__(self, state_id):
        self.visible = True  # Visibility property for all states
        self.state_id = state_id  # Add an ID property for the state

    def draw(self, ax):
        raise NotImplementedError("Subclasses should implement this!")


# Generating State: draws multiple points in a specific color
class GeneratingState(State):
    def __init__(self, state_id, X, color='blue', zorder=1):
        super().__init__(state_id)
        self.X = X  # Multidimensional array of points
        self.color = color
        self.zorder = zorder  # Set the zorder for GenerateState
        self.scatter_plots = []  # Store the scatter plots

    def draw(self, ax):
        if not self.scatter_plots:  # If scatter points are not created
            for point in self.X:
                scatter_plot = ax.scatter(point[0], point[1], color=self.color, zorder=self.zorder)
                self.scatter_plots.append(scatter_plot)
        for scatter_plot in self.scatter_plots:
            scatter_plot.set_visible(self.visible)


# Select State: draws the best point in a different color
class SelectState(State):
    def __init__(self, state_id, X_best, color='red', marker='*', zorder=3):
        super().__init__(state_id)
        self.X_best = X_best  # Best X point (multidimensional)
        self.color = color
        self.marker = marker
        self.zorder = zorder  # Set the zorder for SelectState
        self.scatter_plot = None  # Store the scatter plot

    def draw(self, ax):
        if not self.scatter_plot:
            self.scatter_plot = ax.scatter(self.X_best[0], self.X_best[1], color=self.color, marker=self.marker, s=100,
                                           zorder=self.zorder)
        self.scatter_plot.set_visible(self.visible)


# Update Points State: draws points with a different color
class UpdatePointsState(GeneratingState):
    def __init__(self, state_id, X, color='green', zorder=2):
        super().__init__(state_id, X, color, zorder)


# Update Rectangle State: draws a rectangle around the best point
class UpdateRectangleState(State):
    def __init__(self, state_id, X_best, length, color='purple', zorder=4):
        super().__init__(state_id)
        self.X_best = X_best
        self.length = length
        self.color = color
        self.zorder = zorder  # Set the zorder for UpdateRectangleState
        self.rect = None  # Store the rectangle

    def draw(self, ax):
        if not self.rect:
            x, y = self.X_best
            self.rect = plt.Rectangle((x - self.length, y - self.length), 2 * self.length, 2 * self.length,
                                      edgecolor=self.color, facecolor='none', linestyle='--', zorder=self.zorder)
            ax.add_patch(self.rect)
        self.rect.set_visible(self.visible)


# Function to create 2D contour plot
def create_2D_contour_plot(fun, bounds=[[0, 0], [1, 1]], ax=None):
    x1 = np.linspace(bounds[0][0], bounds[1][0], 100)
    x2 = np.linspace(bounds[0][1], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = fun([X1, X2])
    contour = ax.contour(X1, X2, Z, levels=50, cmap='viridis', zorder=0)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Contour plot of the function')
    ax.figure.colorbar(contour, ax=ax, label='f(X1, X2)')


# Define a sample function for the contour plot
def sample_function(X):
    x1, x2 = X
    return (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(
        x1) + 10

def f(X, negate = False):
    result = (1.0 - X[0]) ** 2 + 100.0 * ((X[1] - X[0] ** 2) ** 2)
    return result if not negate else -result

# Example usage for animation
fig, ax = plt.subplots()
ax.set_xlim(0, 11)
ax.set_ylim(0, 11)

# Plot the contour function in the background
#create_2D_contour_plot(f, bounds=[[-1.5, -0.5], [1.5, 2.5]], ax=ax)
create_2D_contour_plot(sample_function, bounds=[[0, 0], [11, 11]], ax=ax)

'''class ScboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(2, **tkwargs) * torch.inf
    restart_triggered: bool = False
    previous_best: tuple = (0, 0)  # Initialize previous best
    current_best: tuple = (0, 0)  # Initialize current best
    Y: torch.Tensor  # Store the Y values for tracking the best points

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))

    def update_length(self):
        """Update the length based on success and failure counters."""
        if self.success_counter >= self.success_tolerance:
            self.success_counter = 0
            self.length /= 2
        elif self.failure_counter >= self.failure_tolerance:
            self.failure_counter = 0
            self.length *= 2

    def best_point(self, start: int, end: int):
        """Find the best point in Y from start to end, and update counters."""
        max_y = self.current_best

        for i in range(start, end):
            if self.Y[i] > max_y[0]:
                max_y = [self.Y[i], i]

        if self.current_best[1] != max_y[1]:
            self.success_counter += 1
        else:
            self.failure_counter += 1

        self.update_length()
        self.previous_best = self.current_best
        self.current_best = max_y


def parse_scbo_output(X: torch.Tensor, Y: torch.Tensor, batch_size: int, state_id: int) -> List[
    State]:
    states = []  # List to store all states
    scbo_state = scbo.ScboState(dim = 2, batch_size=5)
    # Check if X size is divisible by batch_size
    if X.size(0) % batch_size != 0:
        raise ValueError(f"The number of points {X.size(0)} is not divisible by batch size {batch_size}.")

    # Iterate through the batches
    for i in range(0, X.size(0), batch_size):
        # Extract the batch
        batch_X = X[i:i + batch_size]  # Get the current batch of points
        batch_Y = Y[i:i + batch_size]  # Get the corresponding batch of values

        # Create GeneratingState with these points
        generating_state = GeneratingState(state_id, [tuple(batch_X[j].tolist()) for j in range(batch_size)])
        states.append(generating_state)

        # Find the index of the best point in Y (minimum value for the Rosenbrock function)
        best_index = batch_Y.argmin().item()  # Get the index of the minimum value in Y
        best_value = batch_Y[best_index].item()  # Best value
        best_point = tuple(batch_X[best_index].tolist())  # Corresponding point in X

        # Create SelectState with the best point
        select_state = SelectState(state_id, best_point)
        states.append(select_state)

        # Call the best_point method to update current_best and length
        scbo_state.Y = Y  # Set the Y values for tracking
        scbo_state.best_point(i, i + batch_size)

        # Optionally create and append UpdatePointsState here if needed
        update_state = UpdatePointsState(state_id, [tuple(batch_X[j].tolist()) for j in range(batch_size)])
        states.append(update_state)

        # Create UpdateRectangleState around the best point
        update_rectangle_state = UpdateRectangleState(state_id, best_point, scbo_state.length)
        states.append(update_rectangle_state)

    return states  # Return the list of all states


# Usage example with scbo
scbostate = scbo.ScboState(dim=2, batch_size=5)
states_1 = []
states_2 = []
try:
    X1, Y1, C1 = scbo.testOne()
    if X1 is None or Y1 is None:
        raise ValueError("Received None as output from scbo.testOne()")
    print(f":{X1}")
    print(f":{Y1}")
    # Parse output and create states
    states_1 = parse_scbo_output(X1, Y1, scbostate.batch_size, state_id=1)

except Exception as e:
    print(f"An error occurred: {e}")

scbostate = scbo.ScboState(dim=2, batch_size=5)
try:
    X2, Y2, C2 = scbo.testOne()
    if X2 is None or Y2 is None:
        raise ValueError("Received None as output from scbo.testOne()")

    # Parse output and create states
    print(f":{X2}")
    print(f":{Y2}")
    states_2 = parse_scbo_output(X2, Y2, scbostate.batch_size, state_id=2)

except Exception as e:
    print(f"An error occurred: {e}")


import random
from typing import List'''

'''def create_alternating_states_list(states_1: List[State], states_2: List[State], elements_per_state: int = 4) -> List[State]:
    states = []  # List to store the final selected states

    while states_1 or states_2:  # Continue until both lists are empty
        # Randomly choose which list to take from (1 or 2)
        if random.choice([1, 2]) == 1 and states_1:  # Choose from states_1 if it's not empty
            # Pop the first `elements_per_state` from states_1
            states_to_add = states_1[:elements_per_state]  # Get the first 4 elements
            states_1 = states_1[elements_per_state:]  # Remove the first 4 elements from states_1
            states.extend(states_to_add)  # Add them to the final list

        elif states_2:  # Otherwise choose from states_2 if it's not empty
            # Pop the first `elements_per_state` from states_2
            states_to_add = states_2[:elements_per_state]  # Get the first 4 elements
            states_2 = states_2[elements_per_state:]  # Remove the first 4 elements from states_2
            states.extend(states_to_add)  # Add them to the final list

    return states'''

# Create several states with IDs
states = [
    GeneratingState(state_id=1, X=[[1, 10], [1, 1], [10, 10], [10, 1], [5, 5]], zorder=1),
    SelectState(state_id=1, X_best=[5, 5], zorder=3),
    UpdatePointsState(state_id=1, X=[[1, 10], [1, 1], [10, 10], [10, 1], [5, 5]], zorder=2),
    UpdateRectangleState(state_id=1, X_best=[5, 5], length=2.5, zorder=4),

    GeneratingState(state_id=2, X=[[7, 6], [6, 7], [6, 6], [6.5, 6.5], [6.5, 7]], zorder=1),
    SelectState(state_id=2, X_best=[6, 6], color = 'yellow' ,zorder=3),
    UpdatePointsState(state_id=2, X=[[7, 6], [6, 7], [6, 6], [6.5, 6.5], [6.5, 7]], zorder=2),
    UpdateRectangleState(state_id=2, X_best=[6, 6], length=2.5, color='red', zorder=4),

    GeneratingState(state_id=1, X=[[4, 4], [3, 3], [3, 4], [4, 3], [5.5, 5.5]], zorder=1),
    SelectState(state_id=1, X_best=[5.5, 5.5], zorder=3),
    UpdatePointsState(state_id=1, X=[[4, 4], [3, 3], [3, 4], [4, 3], [5.5, 5.5]], zorder=2),
    UpdateRectangleState(state_id=1, X_best=[5.5, 5.5], length=5, zorder=4),
]
#states = create_alternating_states_list(states_1,states_2,elements_per_state=4)
print(states)
# Variable to store the current frame
current_frame = 0


# Function to create a mapping from state IDs to frames in batches of 4
# Function to create a mapping from state IDs to frames in batches of 4
def create_id_frame_map(states, current_frame):
    id_frame_map = {}

    # Iterate through all states in batches of 4
    for i in range(0, current_frame + 1, 4):  # Step by 4
        # Determine the batch range
        batch_end = min(i + 4, current_frame + 1)  # Include the current frame

        # Get the state IDs in the current batch
        batch_ids = {states[j].state_id for j in range(i, batch_end)}

        for state_id in batch_ids:
            # Initialize the list for this state ID if it doesn't exist
            if state_id not in id_frame_map:
                id_frame_map[state_id] = []

            # Get the current batch of frames for this state ID
            batch_frames = [j for j in range(i, batch_end) if states[j].state_id == state_id]

            # Replace the frame list for this ID with the current batch frames (max 4)
            id_frame_map[state_id] = batch_frames[-4:]  # Keep only the last 4 frames

    return id_frame_map


# Function to update the plot for animation
def update(frame):
    # Hide all states initially
    for state in states:
        state.visible = False
        state.draw(ax)

    # Create a mapping of state IDs to their respective frame ranges
    id_frame_map = create_id_frame_map(states, frame)

    # Show frames for all IDs up to the current frame
    for state_id, frame_indices in id_frame_map.items():
        for f in frame_indices:
            states[f].visible = True
            states[f].draw(ax)

    # Always show all UpdatePointsState up to the current frame
    for i in range(frame + 1):
        if isinstance(states[i], UpdatePointsState):
            states[i].visible = True
            states[i].draw(ax)
    print(id_frame_map)
    # Force re-rendering of the plot to reflect changes in visibility
    plt.draw()




# Function to handle key press events for forward/backward movement
def on_key(event):
    global current_frame
    if event.key == 'right':  # Move forward
        if current_frame < len(states) - 1:
            current_frame += 1
            update(current_frame)
    elif event.key == 'left':  # Move backward
        if current_frame > 0:
            current_frame -= 1
            update(current_frame)


# Connect the key press event to the figure
fig.canvas.mpl_connect('key_press_event', on_key)

# Create an animation: start at the first frame
update(current_frame)

plt.show()
