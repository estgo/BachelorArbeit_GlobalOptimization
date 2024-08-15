import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from enum import Enum, auto
import torch


class State(Enum):
    GENERATING = auto()
    DRAW = auto()
    SEARCHING = auto()
    IDLE = auto()

class AnimateTurbo:
    def __init__(self, X, Y, batch_size, length, bounds, negate=False, success = 2, failure = 3):

        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(Y):
            Y = Y.cpu().numpy()
        if torch.is_tensor(bounds):
            bounds = bounds.cpu().numpy()
        self.X = np.array(X)
        self.Y = np.squeeze(np.asarray(Y))
        if self.Y.ndim > 1:
            self.Y = self.Y.flatten()
        if isinstance(self.Y[0], np.ndarray):
            self.Y = np.array([np.asscalar(y) for y in self.Y])
        self.Y = np.array([float(y) for y in self.Y])
        self.batch_size = batch_size
        self.length = length
        self.bounds = bounds
        self.negate = negate
        self.current_best = [float('-inf'), -1]
        self.previous_best = [float('-inf'), -1]
        self.success_counter = 0
        self.success_tolerance = success
        self.failure_counter = 0
        self.failure_tolerance = failure

        self.current_state = State.GENERATING
        self.previous_state = State.IDLE
        self.transitions = {
            State.IDLE: State.GENERATING,
            State.GENERATING: State.DRAW,
            State.DRAW: State.SEARCHING,
            State.SEARCHING: State.GENERATING
        }
        self.current_frame = 1
        self.max_frame = len(X)/batch_size
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.scatter = self.ax.scatter([], [], color='blue')
        self.rectangles = []
        self.ani = None

        # Set up the plot
        self.ax.set_xlim(bounds[0][0], bounds[1][0])
        self.ax.set_ylim(bounds[0][1], bounds[1][1])
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_title('Animation with Rectangles')

    def update_length(self):
        if self.success_counter >= self.success_tolerance:
            self.success_counter = 0
            self.length /= 2
        elif self.failure_counter >= self.failure_tolerance:
            self.failure_counter = 0
            self.length *= 2

    def best_point(self, start, end):
        max_y = self.current_best

        for i in range(start, end):
            if self.Y[i] > max_y[0]:
                max_y = [self.Y[i], i]
        if self.current_best[1] != max_y[1]:
            self.success_counter += 1
        else:
            self.failure_counter +=1
        self.update_length()
        self.previous_best = self.current_best
        self.current_best = max_y

    def update_state(self):
        if self.current_frame < 1:
            self.current_state = State.GENERATING
            self.previous_state = State.IDLE
            return
        self.previous_state = self.current_state
        self.current_state = self.transitions[self.current_state]


    def draw_past_points(self, start, end):
        print(self.X[start:end, 0])
        self.ax.plot(self.X[start:end, 0], self.X[start:end, 1], 'go')
    def update_plot(self):
        # Clear previous rectangles
        print(self.current_best)
        print((self.current_frame-1)*self.batch_size)
        print((self.current_frame)*self.batch_size)
        for rect in self.rectangles:
            rect.set_visible(False)
        current_indices = range((self.current_frame-1) * self.batch_size , min(self.current_frame * self.batch_size, len(self.X)))
        print(self.current_frame)
        print(current_indices)
        #if self.current_state == State.GENERATING:
        self.ax.plot(self.X[current_indices, 0], self.X[current_indices, 1], 'bo')  # 'bo' for blue dots
        self.draw_past_points(0, (self.current_frame-1)*self.batch_size)

        if self.current_state == State.DRAW:
            self.best_point((self.current_frame-1)*self.batch_size, (self.current_frame)*self.batch_size)
            if self.current_best[1] != self.previous_best[1] and self.previous_best[1] >= 0:
                self.ax.plot(self.X[self.previous_best[1]][0], self.X[self.previous_best[1]][1], 'bo')


        if self.current_frame > 1 and self.current_best[1] >= 0:
            self.ax.plot(self.X[self.current_best[1]][0], self.X[self.current_best[1]][1], 'ro')
        self.update_state()
        print(self.current_best)

        if self.current_best[1] >= 0:
            x, y = self.X[self.current_best[1]]
            rect = plt.Rectangle((x - self.length, y - self.length), 2 * self.length, 2 * self.length,
                                 edgecolor='red', facecolor='none', linestyle='--')
            self.ax.add_patch(rect)
            self.rectangles.append(rect)


        # Update scatter plot with the selected points
        current_points = np.array([self.X[current_indices, 0], self.X[current_indices, 1]]).T
        self.scatter.set_offsets(current_points)
        # Redraw the canvas
        self.fig.canvas.draw_idle()


    def create_animation(self):
        def update(frame):
            # Determine the indices to show
            start_idx = frame * self.batch_size
            end_idx = start_idx + self.batch_size
            self.current_indices = range(start_idx, min(end_idx, len(self.X)))
            #self.update_plot()
            return self.scatter,

        self.ani = FuncAnimation(self.fig, update, frames=(len(self.X) // self.batch_size + 1),
                                interval=500, blit=True, repeat=False)

    def embed_animation_in_frame(self, master_frame):
        canvas = FigureCanvasTkAgg(self.fig, master=master_frame)
        canvas.draw()
        tk_canvas = canvas.get_tk_widget()
        tk_canvas.pack(fill='both', expand=True)
        return canvas

    def start_animation(self):
        if self.ani:
            self.ani.event_source.start()

    def pause_animation(self):
        if self.ani:
            self.ani.event_source.stop()

    def step_forward(self):
        if self.current_frame <= self.max_frame:
            if self.current_state == State.SEARCHING:
                self.current_frame = self.current_frame + 1
            self.update_plot()
            self.fig.canvas.draw_idle()

    def step_backward(self):
        if self.current_frame > 1:
            if self.current_state == State.GENERATING:
                self.current_frame = self.current_frame - 1
            self.update_plot()
            self.fig.canvas.draw_idle()

    def update_data(self, X, Y, batch_size,length, bounds, negate=False, success = 2, failure = 3):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(Y):
            Y = Y.cpu().numpy()
        if torch.is_tensor(bounds):
            bounds = bounds.cpu().numpy()
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.bounds = bounds
        self.length = length
        self.bounds = bounds
        self.negate = negate
        self.current_best = [float('-inf'), -1]
        self.current_state = State.GENERATING
        self.previous_state = State.IDLE
        self.current_frame = 1
        self.max_frame = len(X) / batch_size
        self.success_tolerance = success
        self.failure_tolerance = failure

        self.update_animation(X,Y)

    def restart_animation(self):
        self.current_state = State.GENERATING
        self.previous_state = State.IDLE
        self.current_frame = 1
        self.update_animation(self.X, self.Y)
    def update_animation(self, X, Y):
        # Clear the previous plot elements
        self.ax.clear()
        self.rectangles.clear()

        # Reset the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.scatter = self.ax.scatter([], [], color='blue')

        # Set up the plot with the new bounds
        self.ax.set_xlim(self.bounds[0][0], self.bounds[1][0])
        self.ax.set_ylim(self.bounds[0][1], self.bounds[1][1])
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_title('Animation with Rectangles')

        # Redraw the plot with the updated data
        #self.update_plot()

# Example usage with Tkinter

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Animation Example")

    X_test = [[1, 10], [1, 1], [10, 10], [10, 1], [5, 5],
              [7, 6], [6, 7], [6, 6], [6.5, 6.5], [6.5, 7],
              [4, 4], [3, 3], [3, 4], [4, 3], [5.5, 5.5]]

    Y_test = np.array([50, 49, 70, 88, 5, 80, 90, 70, 30, 55, 100, 80, 200, 300, 4])
    boundaries = [[0,0], [10,10]]
    batch_size_test = 5
    length_test = 2.5  # Length of the rectangle sides

    animation = AnimateTurbo(X_test, -Y_test, batch_size_test, length_test, boundaries)
    animation.create_animation()
    animation.embed_animation_in_frame(root)

    # Adding control buttons
    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X)

    start_button = tk.Button(control_frame, text="Start", command=animation.start_animation)
    start_button.pack(side=tk.LEFT)

    pause_button = tk.Button(control_frame, text="Pause", command=animation.pause_animation)
    pause_button.pack(side=tk.LEFT)

    step_forward_button = tk.Button(control_frame, text="Next Step", command=animation.step_forward)
    step_forward_button.pack(side=tk.LEFT)

    step_backward_button = tk.Button(control_frame, text="Previous Step", command=animation.step_backward)
    step_backward_button.pack(side=tk.LEFT)

    restart_button = tk.Button(control_frame, text="Restart", command=animation.restart_animation)
    restart_button.pack(side=tk.LEFT)

    root.mainloop()
