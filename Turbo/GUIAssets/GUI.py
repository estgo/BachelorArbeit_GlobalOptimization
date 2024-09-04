import tkinter as tk
from tkinter import ttk

import numpy as np
import sys
sys.path.append('C:/Users/estgo/OneDrive/Bureau/Bachelorarbeit/Turbo/')
from widgets import create_button, create_combobox
from plotter import create_scatter_plot, create_bar_plot, embed_plot_in_frame, create_2D_contour_plot
import animation
import ttkbootstrap as ttkb
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import Turbo.scbo as scbo
import Turbo.evaluations as evaluations


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.canvas = None
        self.buttons = {}
        self.comboboxes = {}
        style = ttkb.Style(theme='superhero')
        self.title("Matplotlib Animation with Controls")
        self.geometry("800x600")

        # Create a notebook (tab control)
        self.notebook = tk.ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")

        # Create the frames for each tab
        self.view1_frame = tk.Frame(self.notebook)
        self.view1_left_frame = None
        self.view1_right_frame = None
        self.view2_frame = tk.Frame(self.notebook)
        self.view2_left_frame = None
        self.view2_right_frame = None
        self.notebook.add(self.view1_frame, text="View 1")
        self.notebook.add(self.view2_frame, text="View 2")

        # Initialize frames
        self.setup_view1()
        self.setup_view2()

        self.animation_button_config()

    def setup_view1(self):
        self.view1_left_frame = tk.Frame(self.view1_frame, width=400, height=600, bg="white")
        self.view1_left_frame.pack(side="left", fill="both", expand=True)
        self.view1_right_frame = tk.Frame(self.view1_frame, width=400, height=600, bg="lightgray")
        self.view1_right_frame.pack(side="right", fill="both", expand=True)

        # ComboBox for Selection
        options_1 = ["--[name to be decided]--", "Turbo", "ScoBO"]
        self.comboboxes['Select State'] = create_combobox(self.view1_right_frame, options_1, x=10, y=10, width=20)

        # Create the second combobox with the function options
        options_2 = ["--Select function--", "Ackley", "Michael", "Robot"]
        self.comboboxes['Select Function'] = create_combobox(self.view1_right_frame, options_2, x=10, y=50, width=20)

        # Buttons to control the animation
        self.buttons['start_turbo'] = create_button(
            self.view1_right_frame,
            text="Start Process",
            x=10,
            y=90,
            command=self.start_turbo_process
        )
        X_test = [[1, 10], [1, 1], [10, 10], [10, 1], [5, 5],
                  [7, 6], [6, 7], [6, 6], [6.5, 6.5], [6.5, 7],
                  [4, 4], [3, 3], [3, 4], [4, 3], [5.5, 5.5]]

        Y_test = np.array([50, 49, 70, 88, 5, 80, 90, 70, 30, 55, 100, 80, 200, 300, 4])
        boundaries = [[0, 0], [10, 10]]

        # Create animation
        self.ani = animation.AnimateTurbo(X_test, -Y_test, batch_size=5, length=2.5, bounds=boundaries)
        self.ani.create_animation()
        self.canvas = self.ani.embed_animation_in_frame(self.view1_left_frame)

        self.buttons['start_animation'] = create_button(self.view1_right_frame, text="Start", x=10, y=200, command=self.ani.start_animation,
                                                        state='disabled')
        self.buttons['restart'] = create_button(self.view1_right_frame, text="restart", x=80, y=200, command=self.restart_canvas,
                                                state='normal')
        self.buttons['pause_animation'] = create_button(self.view1_right_frame, text="Pause", x=10, y=250, command=self.ani.pause_animation,
                                                        state='disabled')
        self.buttons['forward_animation'] = create_button(self.view1_right_frame, text="Step Forward", x=10, y=290,
                                                          command=self.ani.step_forward, state='disabled')
        self.buttons['backward_animation'] = create_button(self.view1_right_frame, text="Step Backward", x=10, y=330,
                                                           command=self.ani.step_backward, state='disabled')

    def setup_view2(self):
        self.view2_left_frame = tk.Frame(self.view2_frame, width=600, height=600, bg="white")
        self.view2_left_frame.pack(side="left", fill="both", expand=True)
        self.view2_right_frame = tk.Frame(self.view2_frame, width=200, height=600, bg="lightgray")
        self.view2_right_frame.pack(side="right", fill="both", expand=True)

        bounds = [
            [-5, 0],
            [10, 15]
        ]
        # Scatter and Bar plots
        contour_fig = create_2D_contour_plot(fun=evaluations.Branin().fun, bounds=bounds)
        embed_plot_in_frame(contour_fig, self.view2_left_frame)

        bar_fig = create_bar_plot()
        embed_plot_in_frame(bar_fig, self.view2_left_frame)

        # Button in the right frame of view2
        clear_button = tk.Button(self.view2_right_frame, text="Clear View 2 Left", command=self.clear_view2_left_frame)
        clear_button.pack(pady=20, padx=10, anchor='center')

    def clear_view2_left_frame(self):
        for child in self.view2_left_frame.winfo_children():
            child.destroy()

    def restart_canvas(self):
        X_test = [[1, 9], [1, 1], [9, 9], [9, 1], [5, 5],
                  [7, 6], [6, 7], [6, 6], [6.5, 6.5], [6.5, 7],
                  [4, 4], [3, 3], [3, 4], [4, 3], [5.5, 5.5]]

        Y_test = np.array([50, 49, 70, 88, 5, 80, 90, 70, 30, 55, 100, 80, 200, 300, 4])
        boundaries = [[0, 0], [10, 10]]
        self.ani.update_data(X_test, -Y_test, batch_size=5, length=2.5, bounds=boundaries)
        for widget in self.view1_left_frame.winfo_children():
            widget.destroy()
        new_canvas = self.ani.embed_animation_in_frame(self.view1_left_frame)


    def animation_button_config(self, state = 'normal'):
        self.buttons['start_animation'].config(state=state)
        self.buttons['pause_animation'].config(state=state)
        self.buttons['forward_animation'].config(state=state)
        self.buttons['backward_animation'].config(state=state)



    def start_turbo_process(self):
        global X, Y  # Use global variables for canvas, X, and Y
        # Disable the buttons
        self.buttons['start_turbo'].config(state="disabled")
        self.animation_button_config('disabled')

        # Define the task to be run in a separate thread
        def run_search():
            scbostate = scbo.ScboState(dim=2, batch_size=5)

            try:
                X, Y, C = scbo.testOne()
                if X is None or Y is None:
                    raise ValueError("Received None as output from scbo.testOne()")
            except Exception as e:
                print(f"An error occurred: {e}")
                self.buttons['start_turbo'].config(state="normal")
                self.animation_button_config()
                return

            self.ani.update_data(X, Y, 5, 1, [[-1.5, -0.5], [1.5, 2.5]], True, scbostate.success_tolerance,
                            scbostate.failure_tolerance)
            print(self.view1_left_frame.winfo_children())
            for widget in self.view1_left_frame.winfo_children():
                widget.destroy()

                # Embed the new animation into the frame
            print(self.view1_left_frame.winfo_children())
            new_canvas = self.ani.embed_animation_in_frame(self.view1_left_frame)
            print(self.view1_left_frame.winfo_children())

            # Re-enable the buttons once the task is complete
            self.buttons['start_turbo'].config(state="normal")
            self.animation_button_config()

            print("Process completed successfully.")

        # Start the task in a new thread
        threading.Thread(target=run_search).start()


if __name__ == "__main__":
    app = Application()
    app.mainloop()
