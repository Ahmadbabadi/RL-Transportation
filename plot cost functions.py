
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D


def f(distance, value_price):
    v, p = 1, 0
    cost = (distance/v)**1.9 + distance**0.5 * p * value_price
    return -costCcc

def g(distance, value_price):
    v, p = 3, 1
    cost = (distance/v)**1.9 + distance**0.5 * p * value_price
    return -cost

def h(distance, value_price):
    v, p = 9, 3
    cost = (distance/v)**1.9 + distance**0.5 * p * value_price
    return -cost


def plot_function_maxima(x_range=(-5, 5), y_range=(-5, 5), resolution=200):
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    F = f(X, Y)
    G = g(X, Y)
    H = h(X, Y)
    
    functions = np.stack([F, G, H], axis=-1)
    max_indices = np.argmax(functions, axis=-1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    cmap = ListedColormap(colors)
    
    im1 = ax1.contourf(X, Y, max_indices, levels=[-0.5, 0.5, 1.5, 2.5], 
                       colors=colors, alpha=0.8)
    ax1.contour(X, Y, max_indices, levels=[-0.5, 0.5, 1.5, 2.5], 
                colors='black', linewidths=0.5, alpha=0.3)
    
    ax1.set_xlabel('Distance', fontsize=12)
    ax1.set_ylabel('value of price', fontsize=12)
    # ax1.set_title(r'Regions where each function is maximum $ (\frac{distance}{velocity})^{1.9} + (distance)^{0.5)*price*value price} $ ', fontsize=14, fontweight='bold')
    ax1.set_title(r'Regions where each function is maximum', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    
    legend_elements = [Patch(facecolor=colors[0], label='f(x,y) is max'),
                      Patch(facecolor=colors[1], label='g(x,y) is max'),
                      Patch(facecolor=colors[2], label='h(x,y) is max')]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    
    ax2.remove()
    ax2 = fig.add_subplot(122, projection='3d')
    
    x_3d = np.linspace(x_range[0], x_range[1], 50)
    y_3d = np.linspace(y_range[0], y_range[1], 50)
    X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
    
    F_3d = f(X_3d, Y_3d)
    G_3d = g(X_3d, Y_3d)
    H_3d = h(X_3d, Y_3d)
    
    surf1 = ax2.plot_surface(X_3d, Y_3d, F_3d, color=colors[0], alpha=0.6, label='f(x,y)')
    surf2 = ax2.plot_surface(X_3d, Y_3d, G_3d, color=colors[1], alpha=0.6, label='g(x,y)')
    surf3 = ax2.plot_surface(X_3d, Y_3d, H_3d, color=colors[2], alpha=0.6, label='h(x,y)')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Function value')
    ax2.set_title('3D view of all functions')
    
    ax2.text2D(0.05, 0.95, "Red: f(x,y)\nTeal: g(x,y)\nBlue: h(x,y)", 
               transform=ax2.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    unique, counts = np.unique(max_indices, return_counts=True)
    total_points = resolution * resolution
    
    function_names = ['f(x,y)', 'g(x,y)', 'h(x,y)']
    for i, count in enumerate(counts):
        if i < len(unique):
            percentage = (count / total_points) * 100
            print(f"{function_names[unique[i]]} is maximum in {percentage:.1f}% of the region ({count} points)")


if __name__ == "__main__":
    plot_function_maxima(x_range=(0, 20), y_range=(0, 4), resolution=80)
    