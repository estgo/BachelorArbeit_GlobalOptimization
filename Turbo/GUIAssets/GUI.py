import tkinter as tk
from tkinter import ttk

import numpy as np

from widgets import create_button, create_combobox
from plotter import create_scatter_plot, create_bar_plot, embed_plot_in_frame
import animation
import ttkbootstrap as ttkb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Turbo.turbo as turbo
import Turbo.evaluations as evaluation
import threading
import Turbo.scbo as scbo

# Static Variables
canvas = None
buttons = {}
comboboxes = {}
global X , Y, ani
left_frame = None
global root
def get_theme_background_color(style, widget='TFrame'):
    """Get the background color from the ttkbootstrap theme for a specific widget."""
    return style.lookup(widget, 'background')

def animation_button_config(state = 'normal'):
    buttons['start_animation'].config(state=state)
    buttons['pause_animation'].config(state=state)
    buttons['forward_animation'].config(state=state)
    buttons['backward_animation'].config(state=state)


def update_canvas(self, tk_canvas):
    # Clear the current canvas
    tk_canvas.pack_forget()

    # Embed the new animation into the frame
    new_canvas = FigureCanvasTkAgg(self.fig, master=tk_canvas.master)
    new_canvas.draw()
    new_tk_canvas = new_canvas.get_tk_widget()
    new_tk_canvas.pack(fill='both', expand=True)

    return new_canvas


def start_turbo_process():
    global canvas, X, Y, root  # Use global variables for canvas, X, and Y
    # Disable the buttons
    buttons['start_turbo'].config(state="disabled")
    animation_button_config('disabled')

    # Define the task to be run in a separate thread
    def run_search():
        global left_frame
        scbostate = scbo.ScboState(dim=2, batch_size=5)

        try:
            X, Y, C = scbo.testOne()
            if X is None or Y is None:
                raise ValueError("Received None as output from scbo.testOne()")
        except Exception as e:
            print(f"An error occurred: {e}")
            buttons['start_turbo'].config(state="normal")
            animation_button_config()
            return

        global ani
        ani.update_data(X,Y, 5, 1, [[-1.5, -0.5], [1.5, 2.5]], True, scbostate.success_tolerance , scbostate.failure_tolerance)
        print(left_frame.winfo_children())
        for widget in left_frame.winfo_children():
            widget.destroy()

            # Embed the new animation into the frame
        print(left_frame.winfo_children())
        new_canvas = ani.embed_animation_in_frame(left_frame)
        print(left_frame.winfo_children())

        # Re-enable the buttons once the task is complete
        buttons['start_turbo'].config(state="normal")
        animation_button_config()

        print("Process completed successfully.")

    # Start the task in a new thread
    threading.Thread(target=run_search).start()

def restart_canvas():
    X_test = [[1, 10], [1, 1], [10, 10], [10, 1], [5, 5],
              [7, 6], [6, 7], [6, 6], [6.5, 6.5], [6.5, 7],
              [4, 4], [3, 3], [3, 4], [4, 3], [5.5, 5.5]]

    Y_test = np.array([50, 49, 70, 88, 5, 80, 90, 70, 30, 55, 100, 80, 200, 300, 4])
    boundaries = [[0, 0], [10, 10]]
    ani.update_data(X_test, -Y_test, batch_size=5, length=2.5, bounds=boundaries)
    for widget in left_frame.winfo_children():
        widget.destroy()
    new_canvas = ani.embed_animation_in_frame(left_frame)


def main():
    global canvas, ani
    global root, left_frame
    root = tk.Tk()
    root.title("Matplotlib Animation with Controls")
    root.geometry("800x600")

    # Apply superhero theme
    style = ttkb.Style(theme='superhero')

    # Create a notebook (tab control)
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")

    # Create the frames for each tab
    view1_frame = tk.Frame(notebook)
    view2_frame = tk.Frame(notebook)

    # Add the frames to the notebook with titles
    notebook.add(view1_frame, text="View 1")
    notebook.add(view2_frame, text="View 2")

    # View 1 Layout: Split into left and right frames
    left_frame = tk.Frame(view1_frame, width=400, height=600, bg="white")
    left_frame.pack(side="left", fill="both", expand=True)

    right_frame = tk.Frame(view1_frame, width=400, height=600, bg="lightgray")
    right_frame.pack(side="right", fill="both", expand=True)

    # Prepare Values for animation
    X_test = [[1, 10], [1, 1], [10, 10], [10, 1], [5, 5],
              [7, 6], [6, 7], [6, 6], [6.5, 6.5], [6.5, 7],
              [4, 4], [3, 3], [3, 4], [4, 3], [5.5, 5.5]]

    Y_test = np.array([50, 49, 70, 88, 5, 80, 90, 70, 30, 55, 100, 80, 200, 300, 4])
    boundaries = [[0, 0], [10, 10]]

    # Create animation
    ani = animation.AnimateTurbo(X_test, -Y_test, batch_size=5, length=2.5, bounds=boundaries)
    ani.create_animation()
    canvas = ani.embed_animation_in_frame(left_frame)
    print(left_frame.winfo_children())

    # ComboBox for Selection
    options_1 = ["--[name to be decided]--", "Turbo", "ScoBO"]
    comboboxes['Select State'] = create_combobox(right_frame, options_1, x=10, y=10, width=20)

    # Create the second combobox with the function options
    options_2 = ["--Select function--", "Ackley", "Michael", "Robot"]
    comboboxes['Select Function'] = create_combobox(right_frame, options_2, x=10, y=50, width=20)

    # Buttons to control the animation
    buttons['start_turbo'] = create_button(
        right_frame,
        text="Start Process",
        x=10,
        y=90,
        command=start_turbo_process
    )
    buttons['start_animation'] = create_button(right_frame, text="Start", x=10, y=200, command=ani.start_animation, state='disabled')
    buttons['restart'] = create_button(right_frame, text="restart", x=80, y=200, command=restart_canvas, state='normal')
    buttons['pause_animation'] = create_button(right_frame, text="Pause", x=10, y=250, command=ani.pause_animation, state='disabled')
    buttons['forward_animation'] = create_button(right_frame, text="Step Forward", x=10, y=290,
                  command=ani.step_forward, state='disabled')
    buttons['backward_animation'] = create_button(right_frame, text="Step Backward", x=10, y=330,
                  command=ani.step_backward, state='disabled')

    # View 2 Content: Scatter Plot and Bar Plot
    scatter_fig = create_scatter_plot()
    embed_plot_in_frame(scatter_fig, view2_frame)

    bar_fig = create_bar_plot()
    embed_plot_in_frame(bar_fig, view2_frame)
    animation_button_config()
    root.mainloop()

if __name__ == "__main__":
    main()
