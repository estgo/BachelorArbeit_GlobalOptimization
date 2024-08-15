# plotter.py
import matplotlib.pyplot as plt
import numpy as np

def create_scatter_plot():
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    x = np.random.rand(50)
    y = np.random.rand(50)
    ax.scatter(x, y, color='blue')
    return fig

def create_bar_plot():
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    categories = ['A', 'B', 'C', 'D']
    values = [4, 7, 1, 8]
    ax.bar(categories, values, color='green')
    return fig

def embed_plot_in_frame(fig, frame):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
