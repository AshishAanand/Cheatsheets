# Plotly Cheat Sheet

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Plotly-logo.png" width="500" alt="Plotly logo">
</div>

## Table of Contents

- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Basic Concepts](#basic-concepts)
- [Line and Scatter Plots](#line-and-scatter-plots)
- [Bar Charts](#bar-charts)
- [Statistical Charts](#statistical-charts)
- [3D Charts](#3d-charts)
- [Maps and Geospatial](#maps-and-geospatial)
- [Subplots and Multiple Axes](#subplots-and-multiple-axes)
- [Interactive Elements](#interactive-elements)
- [Animation](#animation)
- [Styling and Theming](#styling-and-theming)
- [Export and Deployment](#export-and-deployment)
- [Plotly Express](#plotly-express)
- [Dash Framework](#dash-framework)
- [Resources](#resources)

## Introduction

Plotly is a powerful Python library for creating interactive, publication-quality graphs and dashboards. It's built on top of plotly.js, which is built on D3.js and stack.gl, making it suitable for web-based data visualization. Plotly offers two main interfaces: a low-level interface using `plotly.graph_objects` and a high-level interface using `plotly.express`.

## Installation and Setup

```python
# Installation
pip install plotly

# For notebooks (to display plots inline)
pip install "notebook>=5.3" "ipywidgets>=7.5"

# For interactive widgets in JupyterLab
pip install jupyterlab "ipywidgets>=7.5"
jupyter labextension install jupyterlab-plotly

# For static image export
pip install -U kaleido
```

### Basic imports

```python
# Main imports
import plotly.graph_objects as go  # Low-level interface
import plotly.express as px        # High-level interface
import plotly.io as pio            # I/O functions
from plotly.subplots import make_subplots  # Subplots
import numpy as np
import pandas as pd

# Set default template
pio.templates.default = "plotly_white"  # Other options: "plotly", "plotly_dark", "ggplot2", etc.
```

## Basic Concepts

### Figure structure

```python
# Creating a figure with graph_objects
fig = go.Figure(
    data=[go.Scatter(x=[1, 2, 3], y=[1, 3, 2])],  # List of trace objects
    layout=go.Layout(                              # Layout configuration
        title="Simple Plot",
        xaxis=dict(title="X-axis"),
        yaxis=dict(title="Y-axis")
    )
)

# Alternative syntax
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 3, 2]))
fig.update_layout(title="Simple Plot")

# Showing the figure
fig.show()
```

### Common Figure Methods

```python
# Add trace
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))

# Update layout
fig.update_layout(
    title="Updated Title",
    xaxis_title="X Axis",
    yaxis_title="Y Axis",
    width=800,
    height=500
)

# Update traces
fig.update_traces(marker=dict(size=10, color="red"), selector=dict(mode="markers"))

# Update axes
fig.update_xaxes(range=[0, 5], showgrid=True)
fig.update_yaxes(type="log", showticklabels=False)

# Add shapes, annotations, etc.
fig.add_shape(type="rect", x0=1, y0=1, x1=2, y1=3, line=dict(color="black"))
fig.add_annotation(x=2, y=2, text="Important Point", showarrow=True)
```

## Line and Scatter Plots

### Scatter Plot

```python
# Using graph_objects
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[1, 3, 2, 4, 3],
    mode='markers',  # 'markers', 'lines', 'lines+markers'
    name='Basic Scatter',
    marker=dict(
        size=12,
        color='blue',
        symbol='circle',  # 'circle', 'square', 'diamond', 'cross', etc.
        line=dict(
            color='black',
            width=1
        )
    )
))
fig.update_layout(title='Scatter Plot')

# Using plotly express (simpler)
df = px.data.iris()  # Load sample dataset
fig = px.scatter(
    df, 
    x='sepal_width', 
    y='sepal_length',
    color='species',              # Color by group
    size='petal_length',          # Size by value
    hover_data=['petal_width'],   # Show on hover
    title='Iris Dataset'
)
```

### Line Plot

```python
# Using graph_objects
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5],
    y=[0, 1, 4, 9, 16, 25],
    mode='lines',
    name='Quadratic',
    line=dict(
        color='firebrick',
        width=3,
        dash='solid'  # 'solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'
    )
))
fig.update_layout(title='Line Plot')

# Using plotly express
df = px.data.gapminder().query("country=='Canada'")
fig = px.line(
    df, 
    x='year', 
    y='lifeExp', 
    title='Life expectancy in Canada'
)

# Multiple lines
df = px.data.gapminder().query("continent=='Europe'")
fig = px.line(
    df, 
    x='year', 
    y='lifeExp', 
    color='country', 
    line_group='country', 
    title='Life Expectancy in Europe'
)
```

### Bubble Chart

```python
# Using graph_objects
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 11, 12, 13],
    mode='markers',
    marker=dict(
        size=[40, 60, 80, 100],  # Size of bubbles
        color=[0, 1, 2, 3],      # Color of bubbles
        colorscale='Viridis',    # Color scale
        showscale=True           # Show color scale
    ),
    text=['A', 'B', 'C', 'D'],  # Hover text
))

# Using plotly express
df = px.data.gapminder().query("year==2007")
fig = px.scatter(
    df, 
    x='gdpPercap', 
    y='lifeExp',
    size='pop',                 # Size by population
    color='continent',          # Color by continent
    hover_name='country',       # Show country name on hover
    log_x=True,                 # Log scale for x-axis
    size_max=60,                # Maximum bubble size
    title='GDP vs Life Expectancy in 2007'
)
```

## Bar Charts

### Simple Bar Chart

```python
# Using graph_objects
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['A', 'B', 'C', 'D'],
    y=[10, 15, 13, 17],
    text=[10, 15, 13, 17],
    textposition='auto',
    marker_color='royalblue',  # Either a single color or an array
))
fig.update_layout(title='Simple Bar Chart')

# Using plotly express
df = px.data.gapminder().query("country == 'Canada'")
fig = px.bar(
    df, 
    x='year', 
    y='pop',
    title='Population of Canada Over Time'
)
```

### Grouped and Stacked Bar Charts

```python
# Grouped bar chart (graph_objects)
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['A', 'B', 'C'],
    y=[10, 15, 13],
    name='Group 1'
))
fig.add_trace(go.Bar(
    x=['A', 'B', 'C'],
    y=[8, 11, 9],
    name='Group 2'
))
fig.update_layout(
    title='Grouped Bar Chart',
    barmode='group'  # 'group', 'stack', 'relative', 'overlay'
)

# Stacked bar chart (plotly express)
df = px.data.gapminder().query("continent == 'Europe' and year == 2007")
fig = px.bar(
    df, 
    x='country', 
    y='pop',
    color='gdpPercap',
    title='Population by Country (Europe, 2007)',
    barmode='stack'
)
```

### Horizontal Bar Chart

```python
# Using graph_objects
fig = go.Figure()
fig.add_trace(go.Bar(
    y=['A', 'B', 'C', 'D'],  # Note that y is now categories
    x=[10, 15, 13, 17],      # And x is values
    orientation='h',         # Horizontal orientation
    marker_color='indianred'
))
fig.update_layout(title='Horizontal Bar Chart')

# Using plotly express
df = px.data.gapminder().query("year == 2007").sort_values(by='lifeExp')
fig = px.bar(
    df.head(10), 
    y='country',     # Note y is categories
    x='lifeExp',     # And x is values
    orientation='h', # Horizontal orientation
    color='continent',
    title='10 Countries with Lowest Life Expectancy (2007)'
)
```

## Statistical Charts

### Box Plot

```python
# Using graph_objects
fig = go.Figure()
fig.add_trace(go.Box(
    y=[0, 1, 1, 2, 3, 5, 8, 13, 21],
    boxpoints='all',  # Display all points: 'all', 'outliers', 'suspectedoutliers', False
    jitter=0.3,       # Add horizontal jitter to points
    pointpos=-1.8,    # Offset points horizontally
    marker_color='rgb(107, 174, 214)',
    name='Data Points'
))
fig.update_layout(title='Box Plot')

# Using plotly express
df = px.data.tips()
fig = px.box(
    df, 
    x='day', 
    y='total_bill', 
    color='smoker',
    notched=True,  # Add a notch to the box plot
    title='Box Plot of Total Bill by Day and Smoker Status'
)
```

### Violin Plot

```python
# Using graph_objects
fig = go.Figure()
fig.add_trace(go.Violin(
    y=[0, 1, 1, 2, 3, 5, 8, 13, 21],
    box_visible=True,        # Show box plot inside violin
    meanline_visible=True,   # Show mean line
    points='all',            # Show all points: 'all', 'outliers', 'suspectedoutliers', False
    name='Distribution'
))
fig.update_layout(title='Violin Plot')

# Using plotly express
df = px.data.tips()
fig = px.violin(
    df, 
    x='day', 
    y='tip', 
    color='sex',
    box=True,        # Show box plot inside violin
    points="all",    # Show all data points
    title='Violin Plot of Tips by Day and Sex'
)
```

### Histogram

```python
# Using graph_objects
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=[1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5],
    nbinsx=10,             # Number of bins
    histnorm='probability', # Normalization: '', 'percent', 'probability', 'density', 'probability density'
    marker_color='rgb(55, 83, 109)',
    opacity=0.7
))
fig.update_layout(title='Histogram')

# Using plotly express
df = px.data.tips()
fig = px.histogram(
    df, 
    x='total_bill',
    color='sex',
    marginal='rug',             # Add a rug plot at the margin: 'rug', 'box', 'violin', None
    histnorm='percent',         # Normalization
    barmode='overlay',          # 'overlay', 'stack', 'group', 'relative'
    title='Histogram of Total Bill'
)
```

### Density Heatmap

```python
# Using graph_objects
x = np.random.normal(2, 1, 1000)
y = np.random.normal(1, 1, 1000)
fig = go.Figure()
fig.add_trace(go.Histogram2d(
    x=x,
    y=y,
    colorscale='Blues',
    nbinsx=20,
    nbinsy=20,
    colorbar=dict(title='Count')
))
fig.update_layout(title='2D Histogram (Density Heatmap)')

# Using plotly express
df = px.data.tips()
fig = px.density_heatmap(
    df, 
    x='total_bill', 
    y='tip',
    marginal_x='histogram',  # Add histogram at the margin: 'histogram', 'box', 'violin', 'rug', None
    marginal_y='histogram',
    title='Density Heatmap of Tips vs Total Bill'
)
```

## 3D Charts

### 3D Scatter Plot

```python
# Using graph_objects
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=[1, 2, 3, 4],
    y=[1, 3, 2, 4],
    z=[1, 2, 3, 1],
    mode='markers',
    marker=dict(
        size=12,
        color=[0, 1, 2, 3],  # Set color by point
        colorscale='Viridis', # Color scale
        opacity=0.8
    )
))
fig.update_layout(
    title='3D Scatter Plot',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    )
)

# Using plotly express
df = px.data.iris()
fig = px.scatter_3d(
    df, 
    x='sepal_length', 
    y='sepal_width', 
    z='petal_width',
    color='species',
    size='petal_length',
    title='3D Iris Dataset Visualization'
)
```

### 3D Surface Plot

```python
# Using graph_objects
x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
y = x.copy().T  # transpose
z = np.sin(x ** 2 + y ** 2)

fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(
    title='3D Surface Plot',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    )
)

# Using plotly express
df = px.data.election()  # Sample dataset with 3 columns we can use
fig = px.scatter_3d(
    df, 
    x='Joly', 
    y='Coderre', 
    z='Bergeron',
    color='winner',
    size='total',
    symbol='winner'
)
```

### 3D Line Plot

```python
# Using graph_objects
t = np.linspace(0, 10, 100)
x = np.cos(t)
y = np.sin(t)
z = t

fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='lines',
    line=dict(
        color=t,
        width=6,
        colorscale='Viridis'
    )
)])
fig.update_layout(title='3D Line Plot (Helix)')
```

## Maps and Geospatial

### Choropleth Map

```python
# Using graph_objects
fig = go.Figure(data=go.Choropleth(
    locations=['AZ', 'CA', 'NY', 'TX'],  # Spatial coordinates (state codes)
    z=[10, 20, 40, 30],                  # Data to be color-coded
    locationmode='USA-states',           # Set of locations
    colorscale='Reds',
    colorbar_title='Value',
))
fig.update_layout(
    title='US States',
    geo_scope='usa',  # Limit map scope to USA
)

# Using plotly express
df = px.data.gapminder().query("year==2007")
fig = px.choropleth(
    df, 
    locations='iso_alpha',    # Column containing ISO country codes
    color='lifeExp',          # Color by life expectancy
    hover_name='country',     # Hover information
    color_continuous_scale=px.colors.sequential.Plasma,
    title='Life Expectancy by Country (2007)'
)
```

### Scatter Geo

```python
# Using graph_objects
fig = go.Figure(data=go.Scattergeo(
    lon=[0, 75, -100],            # Longitude
    lat=[0, 20, 40],              # Latitude
    text=['Point 1', 'Point 2', 'Point 3'],
    mode='markers',
    marker=dict(
        size=[10, 20, 30],
        color=['blue', 'red', 'green'],
        line_color='black',
        line_width=1,
        sizemode='diameter'
    )
))
fig.update_layout(
    title='World Map with Points',
    geo=dict(
        showland=True,
        landcolor='rgb(243, 243, 243)',
        countrycolor='rgb(204, 204, 204)',
    )
)

# Using plotly express
df = px.data.gapminder().query("year==2007")
fig = px.scatter_geo(
    df, 
    locations='iso_alpha',
    color='continent',              # Color by continent
    hover_name='country',           # Show country name on hover
    size='pop',                     # Size by population
    projection='natural earth',     # Map projection
    title='World Population in 2007'
)
```

### Mapbox (requires a Mapbox token for custom maps)

```python
# Using plotly express with open street map (no token required)
df = px.data.carshare()
fig = px.scatter_mapbox(
    df, 
    lat='centroid_lat', 
    lon='centroid_lon',
    color='peak_hour',
    size='car_hours',
    hover_name='hood_name',
    zoom=10,
    mapbox_style='open-street-map',  # No token required for this style
    title='Car Share Locations'
)

# With a Mapbox token (replace with your own token)
token = "your_mapbox_token"  # Replace with your token
px.set_mapbox_access_token(token)

df = px.data.carshare()
fig = px.scatter_mapbox(
    df, 
    lat='centroid_lat', 
    lon='centroid_lon',
    color='peak_hour',
    size='car_hours',
    hover_name='hood_name',
    zoom=11,
    mapbox_style='light',  # 'basic', 'streets', 'outdoors', 'light', 'dark', 'satellite', 'satellite-streets'
    title='Car Share Locations'
)
```

## Subplots and Multiple Axes

### Basic Subplots

```python
# Create subplots: 2 rows, 2 columns
fig = make_subplots(rows=2, cols=2, subplot_titles=('Plot 1', 'Plot 2', 'Plot 3', 'Plot 4'))

# Add traces
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=1)
fig.add_trace(go.Bar(x=[1, 2, 3], y=[7, 8, 9]), row=1, col=2)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[10, 11, 12]), row=2, col=1)
fig.add_trace(go.Bar(x=[1, 2, 3], y=[13, 14, 15]), row=2, col=2)

fig.update_layout(height=600, width=800, title_text="Multiple Subplots")
```

### Subplots with Different Types

```python
# Create subplots with different specs
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "scatter"}, {"type": "histogram"}],
           [{"type": "box"}, {"type": "heatmap"}]],
    subplot_titles=('Scatter', 'Histogram', 'Box Plot', 'Heatmap')
)

# Add different traces to different subplots
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=1)
fig.add_trace(go.Histogram(x=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]), row=1, col=2)
fig.add_trace(go.Box(y=[1, 2, 3, 4, 5, 6, 7, 8, 9]), row=2, col=1)
fig.add_trace(go.Heatmap(z=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]), row=2, col=2)

fig.update_layout(height=700, width=700, title_text="Different Plot Types")
```

### Subplots with Shared Axes

```python
# Create subplots with shared axes
fig = make_subplots(
    rows=2, cols=2,
    shared_xaxes=True,    # Share x axes
    shared_yaxes=True,    # Share y axes
    vertical_spacing=0.1  # Adjust vertical spacing
)

# Add traces
fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]), row=1, col=1)
fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]), row=1, col=2)
fig.add_trace(go.Scatter(x=[3, 4], y=[1, 2]), row=2, col=1)
fig.add_trace(go.Scatter(x=[3, 4], y=[3, 4]), row=2, col=2)

fig.update_layout(height=600, width=600, title_text="Subplots with Shared Axes")
```

### Secondary Y-Axis

```python
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[40, 50, 60], name="Line 1"),
    secondary_y=False,  # Use primary y-axis
)
fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="Line 2"),
    secondary_y=True,   # Use secondary y-axis
)

# Set axis titles
fig.update_xaxes(title_text="X Axis")
fig.update_yaxes(title_text="Primary Y Axis", secondary_y=False)
fig.update_yaxes(title_text="Secondary Y Axis", secondary_y=True)

fig.update_layout(title_text="Double Y Axis Example")
```

## Interactive Elements

### Dropdown Menus

```python
# Create figure
fig = go.Figure()

# Add traces - each one will be shown/hidden by the dropdown
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="Line 1"))
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[7, 8, 9], name="Line 2"))
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[10, 11, 12], name="Line 3"))

# Create dropdown buttons
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="All",
                     method="update",
                     args=[{"visible": [True, True, True]},
                           {"title": "All Lines"}]),
                dict(label="Line 1",
                     method="update",
                     args=[{"visible": [True, False, False]},
                           {"title": "Line 1 Only"}]),
                dict(label="Line 2",
                     method="update",
                     args=[{"visible": [False, True, False]},
                           {"title": "Line 2 Only"}]),
                dict(label="Line 3",
                     method="update",
                     args=[{"visible": [False, False, True]},
                           {"title": "Line 3 Only"}]),
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.15,
            yanchor="top"
        ),
    ]
)

fig.update_layout(title="Dropdown Menu Example", height=600)
```

### Sliders

```python
# Create data with multiple frames
x = np.arange(10)
y_frames = [np.sin(x + i/10) for i in range(10)]

# Create figure
fig = go.Figure(
    data=[go.Scatter(x=x, y=y_frames[0], name="sinusoid")],
    layout=go.Layout(
        title="Sinusoid Animation",
        xaxis=dict(range=[0, 10], autorange=False),
        yaxis=dict(range=[-1.5, 1.5], autorange=False),
    )
)

# Create frames
frames = [go.Frame(data=[go.Scatter(x=x, y=y)], name=str(i)) for i, y in enumerate(y_frames)]
fig.frames = frames

# Add slider
fig.update_layout(
    sliders=[dict(
        active=0,
        steps=[
            dict(
                method="animate",
                args=[[str(i)], dict(frame=dict(duration=300, redraw=True), transition=dict(duration=300))],
                label=str(i)
            )
            for i in range(10)
        ],
        transition=dict(duration=300),
        x=0.1,
        xanchor="left",
        y=0,
        yanchor="top",
        pad=dict(b=10, t=50),
        currentvalue=dict(
            font=dict(size=12),
            prefix="Phase: ",
            visible=True,
            xanchor="right"
        ),
        len=0.9
    )]
)
```

### Buttons

```python
# Create figure
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode="markers", name="Points"))

# Add buttons
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.1,
            y=1.15,
            buttons=list([
                dict(
                    label="Markers",
                    method="update",
                    args=[{"marker": {"size": 10, "color": "blue"}},
                          {"mode": "markers"}]
                ),
                dict(
                    label="Lines",
                    method="update",
                    args=[{"mode": "lines", "line": {"width": 3, "color": "red"}}]
                ),
                dict(
                    label="Both",
                    method="update",
                    args=[{"mode": "lines+markers",
                           "marker": {"size": 8, "color": "green"},
                           "line": {"width": 2, "color": "green"}}]
                ),
            ]),
        )
    ]
)

fig.update_layout(title="Button Example")
```

## Animation

### Frame-by-Frame Animation

```python
# Create data for animation
import numpy as np
n_frames = 30
x = np.linspace(-2, 2, 100)
r = np.linspace(0, 2, n_frames)

# Create figure
fig = go.Figure(
    data=[go.Scatter(x=x, y=np.sin(x*r[0]), mode="lines", name="sin(x*r)")],
    layout=go.Layout(
        title="Animated Sinusoids",
        xaxis=dict(range=[-2, 2], autorange=False),
        yaxis=dict(range=[-2, 2], autorange=False),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 100, "redraw": True},
                             "fromcurrent": True, "transition": {"duration": 50}}]
            )]
        )]
    )
)

# Create and add frames
frames = [go.Frame(data=[go.Scatter(x=x, y=np.sin(x*r_val))], name=str(i))
          for i, r_val in enumerate(r)]
fig.frames = frames
```

### Using Plotly Express for Animation

```python
# Using plotly express for animation
df = px.data.gapminder()
fig = px.scatter(
    df, 
    x="gdpPercap", 
    y="lifeExp", 
    animation_frame="year",   # Column to animate by
    animation_group="country", # Column to track across animation
    size="pop", 
    color="continent", 
    hover_name="country",
    log_x=True, 
    size_max=55, 
    range_x=[100, 100000], 
    range_y=[25, 90],
    title="Gapminder: GDP vs