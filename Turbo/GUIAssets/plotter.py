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

def create_2D_contour_plot(fun, bounds = [[0,0], [1,1]]):
    # Step 2: Generate a grid of values
    x1 = np.linspace(bounds[0][0], bounds[1][0], 100)
    x2 = np.linspace(bounds[0][1], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Step 3: Evaluate the function on the grid
    Z = fun([X1, X2])

    # Step 4: Create the contour plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    contour = ax.contour(X1, X2, Z, levels=50, cmap='viridis')

    # Add labels and a color bar
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Contour plot of the Branin function')
    fig.colorbar(contour, ax=ax, label='f(X1, X2)')

    # Return the figure
    return fig