# Matplotlib Cheat Sheet

![Matplotlib Logo](https://matplotlib.org/stable/_images/sphx_glr_logos2_003.png)

## Table of Contents
- [Installation](#installation)
- [Basic Plotting](#basic-plotting)
- [Figure and Axes](#figure-and-axes)
- [Plot Customization](#plot-customization)
- [Multiple Plots](#multiple-plots)
- [Special Plot Types](#special-plot-types)
- [Saving and Exporting](#saving-and-exporting)
- [Working with Images](#working-with-images)
- [3D Plotting](#3d-plotting)
- [Tips and Best Practices](#tips-and-best-practices)

## Installation

```python
# Using pip
pip install matplotlib

# Using conda
conda install matplotlib
```

Import conventions:

```python
# Standard imports
import matplotlib.pyplot as plt
import numpy as np

# For 3D plotting
from mpl_toolkits.mplot3d import Axes3D

# For interactive plotting
%matplotlib inline  # For Jupyter Notebooks
```

## Basic Plotting

### Line Plot

```python
# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()
```

### Scatter Plot

```python
# Create data
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
area = (30 * np.random.rand(50))**2

# Create plot
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### Bar Plot

```python
# Create data
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 55, 15]

# Create plot
plt.bar(categories, values)
plt.title('Bar Plot')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()
```

### Histogram

```python
# Create data
data = np.random.randn(1000)

# Create plot
plt.hist(data, bins=30, alpha=0.7, color='b')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

## Figure and Axes

### Creating a Figure and Axes

```python
# Method 1: Using plt.subplots()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)
ax.set_title('Title')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')

# Method 2: Using object-oriented API
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(x, y)
```

### Figure Size and DPI

```python
# Set figure size and DPI
plt.figure(figsize=(10, 6), dpi=100)

# Or for subplots
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
```

### Tight Layout

```python
# Automatically adjust subplot params for better layout
plt.tight_layout()

# Or with specific padding
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
```

## Plot Customization

### Colors, Markers, and Line Styles

```python
# Line color, marker, and style
plt.plot(x, y, color='red', marker='o', linestyle='--', linewidth=2, markersize=8)

# Using shorthand notation (fmt)
plt.plot(x, y, 'ro--')  # red, circle marker, dashed line
```

Common format strings:
- Colors: 'b' (blue), 'g' (green), 'r' (red), 'c' (cyan), 'm' (magenta), 'y' (yellow), 'k' (black), 'w' (white)
- Markers: '.' (point), 'o' (circle), 's' (square), '*' (star), '+' (plus), 'x' (x)
- Line styles: '-' (solid), '--' (dashed), ':' (dotted), '-.' (dash-dot)

### Labels and Title

```python
plt.title('Main Title', fontsize=16, fontweight='bold')
plt.xlabel('X Axis', fontsize=12)
plt.ylabel('Y Axis', fontsize=12)
plt.suptitle('Super Title', fontsize=18)  # Figure-level title
```

### Legends

```python
plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')
plt.legend(loc='best')  # Automatically choose best location

# Other legend locations
plt.legend(loc='upper right')
plt.legend(loc='lower left')

# Custom legend properties
plt.legend(loc='upper left', frameon=False, fontsize=12, ncol=2)
```

### Axis Limits and Ticks

```python
# Set axis limits
plt.xlim(-5, 15)
plt.ylim(-1.5, 1.5)

# Set tick positions
plt.xticks([0, 2, 4, 6, 8, 10])
plt.yticks([-1, -0.5, 0, 0.5, 1])

# Set tick labels
plt.xticks([0, 2, 4, 6, 8, 10], ['0', '2', '4', '6', '8', '10'], rotation=45)
```

### Grid

```python
# Simple grid
plt.grid(True)

# Customized grid
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Grid on specific axes
plt.grid(axis='y')  # Only horizontal grid lines
```

### Text and Annotations

```python
# Add text at specific coordinates
plt.text(x=5, y=0.5, s="Important point", fontsize=12)

# Annotate a point
plt.annotate('Local maximum', xy=(3, 1), xytext=(4, 1.3),
             arrowprops=dict(facecolor='black', shrink=0.05))
```

## Multiple Plots

### Subplots

```python
# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot on each subplot
axs[0, 0].plot(x, y)
axs[0, 0].set_title('Plot 1')

axs[0, 1].scatter(x, y)
axs[0, 1].set_title('Plot 2')

axs[1, 0].bar(categories, values)
axs[1, 0].set_title('Plot 3')

axs[1, 1].hist(np.random.randn(1000))
axs[1, 1].set_title('Plot 4')

# Add a title to the figure
fig.suptitle('Multiple Plots', fontsize=16)

# Adjust spacing between subplots
fig.tight_layout()
```

### Subplot2grid

```python
# Create uneven layouts
fig = plt.figure(figsize=(10, 8))

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))

ax1.plot(x, y)
ax2.scatter(x, y)
ax3.hist(np.random.randn(100))
ax4.bar(categories[:3], values[:3])
ax5.plot(x, np.cos(x))
```

### Twin Axes

```python
# Create plot with two y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# First plot (left y-axis)
ax1.plot(x, y, 'b-')
ax1.set_xlabel('x')
ax1.set_ylabel('sin(x)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Second plot (right y-axis)
ax2 = ax1.twinx()
ax2.plot(x, np.cos(x), 'r-')
ax2.set_ylabel('cos(x)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Two scales')
```

## Special Plot Types

### Pie Chart

```python
# Create data
sizes = [30, 25, 20, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E']
explode = (0.1, 0, 0, 0, 0)  # Explode first slice

# Create pie chart
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures circular pie
plt.title('Pie Chart')
plt.show()
```

### Box Plot

```python
# Create data
data = [np.random.normal(0, std, 100) for std in range(1, 5)]

# Create box plot
plt.boxplot(data, notch=True, patch_artist=True)
plt.title('Box Plot')
plt.xlabel('Group')
plt.ylabel('Value')
plt.show()
```

### Heatmap

```python
# Create data
data = np.random.rand(10, 10)

# Create heatmap
plt.imshow(data, cmap='viridis')
plt.colorbar(label='Value')
plt.title('Heatmap')
plt.show()
```

### Contour Plot

```python
# Create data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create contour plot
plt.contour(X, Y, Z, 20, cmap='RdGy')
plt.colorbar(label='Value')
plt.title('Contour Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show()
```

## Saving and Exporting

```python
# Save figure with different formats and DPI
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.savefig('plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('plot.svg', format='svg', bbox_inches='tight')
plt.savefig('plot.jpg', format='jpg', dpi=300, quality=95, bbox_inches='tight')
```

Parameters for `savefig()`:
- `dpi`: Resolution in dots per inch
- `bbox_inches='tight'`: Trim extra whitespace around the figure
- `transparent=True`: Transparent background
- `facecolor='w'`: Background color
- `pad_inches=0.1`: Padding around the figure

## Working with Images

```python
# Load an image
from matplotlib import image
img = image.imread('image.png')

# Display the image
plt.imshow(img)
plt.axis('off')  # Turn off axis
plt.title('Image')
plt.show()

# Display grayscale image
plt.imshow(img_gray, cmap='gray')
```

## 3D Plotting

```python
# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create data
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Surface plot
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot')

plt.show()
```

Other 3D plot types:

```python
# 3D scatter plot
ax.scatter3D(xs, ys, zs, c=zs, cmap='viridis')

# 3D line plot
ax.plot3D(xs, ys, zs, 'gray')

# 3D wireframe
ax.plot_wireframe(X, Y, Z, color='black')

# 3D contour plot
ax.contour3D(X, Y, Z, 50, cmap='binary')
```

## Tips and Best Practices

### Styling

```python
# See available styles
plt.style.available

# Use a predefined style
plt.style.use('ggplot')
plt.style.use('seaborn-darkgrid')
plt.style.use('fivethirtyeight')

# Reset to default style
plt.style.use('default')
```

### Context managers for temporary style changes

```python
# Temporarily use a different style
with plt.style.context('dark_background'):
    plt.plot(x, y)
    plt.title('Dark style plot')
    plt.show()
```

### Setting default parameters

```python
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
```

### Scale Formatting

```python
# Log scale
plt.xscale('log')
plt.yscale('log')

# Symmetric log scale
plt.yscale('symlog', linthresh=0.1)

# Custom tick formatter
import matplotlib.ticker as ticker
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
```

### Color Maps

```python
# Sequential colormaps (one color, varying intensity)
plt.imshow(data, cmap='viridis')  # The default, perceptually uniform
plt.imshow(data, cmap='plasma')
plt.imshow(data, cmap='inferno')

# Diverging colormaps (two colors, meet in the middle)
plt.imshow(data, cmap='coolwarm')
plt.imshow(data, cmap='RdBu')
plt.imshow(data, cmap='PiYG')

# Qualitative colormaps (distinct colors)
plt.imshow(data, cmap='tab10')
plt.imshow(data, cmap='Set3')
```

### Performance Tips

```python
# For large datasets, use plot() with 'o' marker instead of scatter()
plt.plot(large_x, large_y, 'o', markersize=1)  # Faster than scatter for large data

# Use rasterized=True for vector outputs with dense data
plt.plot(x, y, rasterized=True)

# Avoid unnecessary calls to draw/show
fig, ax = plt.subplots()
for i in range(n_plots):
    # Avoid plt.show() inside the loop
    ax.plot(x, y[i])
plt.show()  # Call only once at the end
```

### Animations

```python
import matplotlib.animation as animation

# Create a figure and axis
fig, ax = plt.subplots()

# Initialize plot with blank data
line, = ax.plot([], [], 'r-')
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1, 1)

# Define update function for animation
def update(frame):
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x + frame/10)
    line.set_data(x, y)
    return line,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Save animation
ani.save('animation.gif', writer='pillow', fps=20)

plt.show()
```