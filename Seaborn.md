# Seaborn Cheat Sheet

<div align="center">
  <img src="https://github.com/mwaskom/seaborn/raw/master/doc/_static/logo-wide-lightbg.svg" width="500" alt="Seaborn logo">
</div>

## Table of Contents

- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Data Preparation](#data-preparation)
- [Figure Aesthetics](#figure-aesthetics) 
- [Distribution Plots](#distribution-plots)
- [Categorical Plots](#categorical-plots)
- [Relational Plots](#relational-plots)
- [Regression Plots](#regression-plots)
- [Matrix Plots](#matrix-plots)
- [Multi-plot Grids](#multi-plot-grids)
- [Color Palettes](#color-palettes)
- [Common Customizations](#common-customizations)
- [Statistical Estimation](#statistical-estimation)
- [Resources](#resources)

## Introduction

Seaborn is a Python data visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics. It integrates closely with pandas data structures.

## Installation and Setup

```python
# Installation
pip install seaborn

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_theme()  # Default theme
sns.set_style("whitegrid")  # Common alternatives: "white", "dark", "darkgrid", "ticks"

# Figure size and resolution
plt.figure(figsize=(10, 6), dpi=100)
```

## Data Preparation

Seaborn works best with tidy pandas DataFrames:

```python
# Load built-in datasets
tips_df = sns.load_dataset("tips")
flights_df = sns.load_dataset("flights")
iris_df = sns.load_dataset("iris")

# Basic DataFrame inspection
tips_df.head()
tips_df.info()
tips_df.describe()
```

## Figure Aesthetics

```python
# Available styles
sns.set_style("whitegrid")  # Grid lines on white background
sns.set_style("white")      # White background without grid lines
sns.set_style("darkgrid")   # Grid lines on dark background
sns.set_style("dark")       # Dark background without grid lines
sns.set_style("ticks")      # Just the axes and ticks

# Context styles
sns.set_context("paper")    # Small figures and thin lines
sns.set_context("notebook") # Default - medium sized figures
sns.set_context("talk")     # Larger figures for presentations
sns.set_context("poster")   # Very large text for posters

# Save and show
plt.tight_layout()
plt.savefig("plot_name.png", dpi=300)
plt.show()
```

## Distribution Plots

### Histograms and KDE

```python
# Histogram
sns.histplot(data=tips_df, x="total_bill", bins=20)

# KDE (Kernel Density Estimation)
sns.kdeplot(data=tips_df, x="total_bill", shade=True)

# Univariate distributions with rugplot
sns.displot(data=tips_df, x="total_bill", kind="hist", kde=True, rug=True)

# Bivariate distribution
sns.displot(data=tips_df, x="total_bill", y="tip", kind="kde")

# ECDF (Empirical Cumulative Distribution Function)
sns.ecdfplot(data=tips_df, x="total_bill")
```

### Box and Violin Plots

```python
# Boxplot
sns.boxplot(data=tips_df, x="day", y="total_bill")
sns.boxplot(data=tips_df, x="day", y="total_bill", hue="smoker")

# Violin plot
sns.violinplot(data=tips_df, x="day", y="total_bill")
sns.violinplot(data=tips_df, x="day", y="total_bill", hue="smoker", split=True)

# Swarm plot (categorical scatterplot)
sns.swarmplot(data=tips_df, x="day", y="total_bill")

# Boxen plot (enhanced box plot for larger datasets)
sns.boxenplot(data=tips_df, x="day", y="total_bill")
```

## Categorical Plots

```python
# Strip plot (scatter)
sns.stripplot(data=tips_df, x="day", y="total_bill", jitter=True)

# Combined box and swarm plot
sns.boxplot(data=tips_df, x="day", y="total_bill")
sns.swarmplot(data=tips_df, x="day", y="total_bill", color=".25")

# Bar plot (with confidence intervals)
sns.barplot(data=tips_df, x="day", y="total_bill")

# Count plot (bar plot of counts)
sns.countplot(data=tips_df, x="day")

# Point plot (showing means with confidence intervals)
sns.pointplot(data=tips_df, x="day", y="total_bill", hue="smoker")

# Categorical plot (combines several types)
sns.catplot(data=tips_df, x="day", y="total_bill", kind="box")
# Kinds: "strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"
```

## Relational Plots

```python
# Scatter plot
sns.scatterplot(data=tips_df, x="total_bill", y="tip")
sns.scatterplot(data=tips_df, x="total_bill", y="tip", hue="smoker", size="size")

# Line plot
sns.lineplot(data=flights_df, x="year", y="passengers")
sns.lineplot(data=flights_df, x="year", y="passengers", hue="month")

# Relational plot (combines scatter and line)
sns.relplot(data=tips_df, x="total_bill", y="tip", kind="scatter")
sns.relplot(data=flights_df, x="year", y="passengers", kind="line")
```

## Regression Plots

```python
# Basic regression plot
sns.regplot(data=tips_df, x="total_bill", y="tip")

# Regression with polynomial fits
sns.regplot(data=tips_df, x="total_bill", y="tip", order=2)

# Binned regression
sns.regplot(data=tips_df, x="total_bill", y="tip", x_bins=5)

# Residuals plot
sns.residplot(data=tips_df, x="total_bill", y="tip")

# Multiple regression with factor variables
sns.lmplot(data=tips_df, x="total_bill", y="tip", hue="smoker")
sns.lmplot(data=tips_df, x="total_bill", y="tip", col="smoker", row="time")
```

## Matrix Plots

```python
# Correlation heatmap
corr = iris_df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")

# Clustered heatmap
sns.clustermap(corr, cmap="coolwarm", standard_scale=1)

# Heatmap from pivoted DataFrame
flights_pivot = flights_df.pivot("month", "year", "passengers")
sns.heatmap(flights_pivot, cmap="YlGnBu")
```

## Multi-plot Grids

### FacetGrid - plot same relationship across subset of data

```python
# Basic facet grid
g = sns.FacetGrid(tips_df, col="time", row="smoker")
g.map(sns.scatterplot, "total_bill", "tip")

# Add another layer and customize
g = sns.FacetGrid(tips_df, col="day", height=4, aspect=.7)
g.map(sns.histplot, "total_bill", kde=True)
g.add_legend()
```

### PairGrid - plot multiple pairwise relationships

```python
# Basic pair grid
g = sns.PairGrid(iris_df)
g.map(sns.scatterplot)

# Different plots on diagonal vs. off-diagonal
g = sns.PairGrid(iris_df)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)

# Add hue
g = sns.PairGrid(iris_df, hue="species")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()
```

### JointGrid - plot bivariate and univariate distributions

```python
# Basic joint plot
sns.jointplot(data=tips_df, x="total_bill", y="tip")

# Change kind
sns.jointplot(data=tips_df, x="total_bill", y="tip", kind="reg")
# Kinds: "scatter", "kde", "hist", "hex", "reg", "resid"

# Add hue
sns.jointplot(data=tips_df, x="total_bill", y="tip", hue="smoker")
```

## Color Palettes

```python
# Built-in color palettes
sns.color_palette("tab10")     # Default categorical
sns.color_palette("pastel")
sns.color_palette("Set2")      # Qualitative (categorical)
sns.color_palette("viridis")   # Sequential
sns.color_palette("RdBu")      # Diverging

# Setting palette
sns.set_palette("Set2")

# Using palette in plots
sns.scatterplot(data=iris_df, x="sepal_length", y="petal_length", 
                hue="species", palette="viridis")

# Custom palette
custom_pal = sns.color_palette("husl", 8)
sns.palplot(custom_pal)  # Display the palette
```

## Common Customizations

```python
# Figure-level vs. axes-level functions
# Figure-level (creates figure): displot, catplot, lmplot, relplot, etc.
# Axes-level (uses existing axes): histplot, boxplot, scatterplot, etc.

# Working with matplotlib directly
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=tips_df, x="total_bill", y="tip", ax=ax)
ax.set_title("Relationship between Bill and Tip")
ax.set_xlabel("Total Bill Amount ($)")
ax.set_ylabel("Tip Amount ($)")

# Despine (remove top and right spines)
sns.despine()

# Setting figure sizes with seaborn
sns.relplot(data=tips_df, x="total_bill", y="tip", height=6, aspect=1.5)
```

## Statistical Estimation

```python
# Confidence intervals
sns.barplot(data=tips_df, x="day", y="total_bill", errorbar=("ci", 95))  # 95% CI
sns.barplot(data=tips_df, x="day", y="total_bill", errorbar=("pi", 50))  # 50% prediction interval
sns.barplot(data=tips_df, x="day", y="total_bill", errorbar="sd")        # Standard deviation

# Aggregation functions
sns.barplot(data=tips_df, x="day", y="total_bill", estimator=np.median)  # Default is mean

# Bootstrapped confidence intervals
sns.regplot(data=tips_df, x="total_bill", y="tip", ci=95)                # 95% CI for regression
```

## Resources

- [Official Seaborn Documentation](https://seaborn.pydata.org/)
- [Seaborn API Reference](https://seaborn.pydata.org/api.html)
- [Seaborn Example Gallery](https://seaborn.pydata.org/examples/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [GitHub Repository](https://github.com/mwaskom/seaborn)

---

<div align="center">
  <p>Created with ❤️ for data visualization enthusiasts</p>
</div>