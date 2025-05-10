# NumPy Cheat Sheet

![NumPy Logo](https://numpy.org/images/logo.svg)

## Table of Contents
- [Installation](#installation)
- [Creating Arrays](#creating-arrays)
- [Array Properties](#array-properties)
- [Array Manipulation](#array-manipulation)
- [Indexing and Slicing](#indexing-and-slicing)
- [Mathematical Operations](#mathematical-operations)
- [Statistical Functions](#statistical-functions)
- [Broadcasting](#broadcasting)
- [Linear Algebra](#linear-algebra)
- [Random Numbers](#random-numbers)
- [Advanced Indexing](#advanced-indexing)
- [Input and Output](#input-and-output)
- [Performance Tips](#performance-tips)
- [Working with Other Libraries](#working-with-other-libraries)

## Installation

```python
# Using pip
pip install numpy

# Using conda
conda install numpy
```

Import convention:

```python
import numpy as np
```

## Creating Arrays

### Basic Arrays

```python
# From a list or tuple
a = np.array([1, 2, 3, 4])
b = np.array([[1, 2, 3], [4, 5, 6]])

# With specific data type
a = np.array([1, 2, 3], dtype=np.float64)
b = np.array([1, 2, 3], dtype=np.complex128)

# From a range
a = np.arange(10)  # 0-9
b = np.arange(1, 10)  # 1-9
c = np.arange(1, 10, 2)  # 1, 3, 5, 7, 9

# Linearly spaced
a = np.linspace(0, 1, 5)  # 5 values from 0 to 1
b = np.linspace(0, 2*np.pi, 100)  # Useful for functions like sin, cos

# Logarithmically spaced
a = np.logspace(0, 2, 5)  # 5 values from 10^0 to 10^2
```

### Special Arrays

```python
# Zeros and ones
a = np.zeros(5)  # 1D array of zeros
b = np.zeros((2, 3))  # 2D array of zeros
c = np.ones(5)  # 1D array of ones
d = np.ones((2, 3))  # 2D array of ones

# Full of specific value
a = np.full(5, 7)  # 1D array of 7s
b = np.full((2, 3), 7)  # 2D array of 7s

# Identity matrix
a = np.eye(3)  # 3x3 identity matrix
b = np.identity(3)  # Another way to create identity matrix

# Diagonal matrix
a = np.diag([1, 2, 3])  # Diagonal elements, zeros elsewhere

# Empty arrays (uninitialized, values may be random)
a = np.empty(5)
b = np.empty((2, 3))
```

### Arrays from Existing Data

```python
# Copy of an array
b = np.copy(a)
b = a.copy()

# View of an array (shares the same data)
b = a.view()

# Memory buffer
from_buffer = np.frombuffer(buffer, dtype=float)

# From existing files
data = np.loadtxt('file.txt')
data = np.genfromtxt('file.csv', delimiter=',')
```

## Array Properties

```python
# Basic properties
a.shape  # Shape (dimensions)
a.ndim  # Number of dimensions
a.size  # Total number of elements
a.dtype  # Data type
a.itemsize  # Size of each element in bytes
a.nbytes  # Total size in bytes

# Memory layout
a.flags  # Information about memory layout
a.strides  # Bytes to step in each dimension when traversing

# Checking array properties
np.iscomplex(a)  # True if any element is complex
np.isreal(a)  # True if all elements are real
np.isscalar(42)  # True if argument is a scalar
np.isfinite(a)  # True for finite elements
np.isinf(a)  # True for infinite elements
np.isnan(a)  # True for NaN (Not a Number) elements
```

## Array Manipulation

### Changing Shape

```python
# Reshape
b = a.reshape(3, 4)  # Reshape to 3x4 array
b = a.reshape(3, -1)  # 3 rows, columns auto-calculated
b = np.reshape(a, (3, 4))  # Function form

# Flattening
b = a.flatten()  # Returns a copy
b = a.ravel()  # Returns a view when possible

# Transposing
b = a.T  # Transpose
b = np.transpose(a)  # Function form
b = a.transpose(1, 0, 2)  # Permute dimensions (for 3D arrays)
```

### Adding/Removing Elements

```python
# Resize (changes the array itself)
a.resize(5, 4)  # Can change size, not just shape

# Append elements
b = np.append(a, [7, 8, 9])  # Appends to flattened array by default
b = np.append(a, [[7, 8, 9]], axis=0)  # Append row
b = np.append(a, [[7], [8]], axis=1)  # Append column

# Insert elements
b = np.insert(a, 1, 5)  # Insert 5 at index 1 (flattened)
b = np.insert(a, 1, [5, 6, 7], axis=0)  # Insert row at index 1

# Delete elements
b = np.delete(a, 1)  # Delete element at index 1 (flattened)
b = np.delete(a, 1, axis=0)  # Delete row at index 1
b = np.delete(a, [1, 3], axis=1)  # Delete columns at indices 1 and 3
```

### Combining Arrays

```python
# Concatenate
c = np.concatenate([a, b])  # Concatenate flattened arrays
c = np.concatenate([a, b], axis=0)  # Row-wise (default)
c = np.concatenate([a, b], axis=1)  # Column-wise

# Vertical and Horizontal Stacking
c = np.vstack((a, b))  # Stack vertically (rows)
c = np.hstack((a, b))  # Stack horizontally (columns)
c = np.dstack((a, b))  # Stack along depth (3rd dimension)

# Stack along new axis
c = np.stack([a, b])  # Create new axis (default: axis=0)
c = np.stack([a, b], axis=1)  # Create new axis at position 1

# Column stack (1D arrays to 2D columns)
c = np.column_stack((a, b))  # Each 1D array becomes a column

# Row stack (1D arrays to 2D rows)
c = np.row_stack((a, b))  # Each 1D array becomes a row
```

### Splitting Arrays

```python
# Split into equal parts
parts = np.split(a, 3)  # Split into 3 equal parts
parts = np.split(a, [3, 5, 7])  # Split at indices 3, 5, and 7

# Horizontal, vertical, and depth splitting
h_parts = np.hsplit(a, 3)  # Split horizontally into 3
v_parts = np.vsplit(a, 3)  # Split vertically into 3
d_parts = np.dsplit(a, 3)  # Split along depth into 3
```

### Repeating Elements

```python
# Repeat elements
b = np.repeat(a, 3)  # Each element repeated 3 times
b = np.repeat(a, [1, 2, 3])  # Element 0 once, element 1 twice, element 2 thrice
b = np.repeat(a, 3, axis=1)  # Each column repeated 3 times

# Tile arrays
b = np.tile(a, 3)  # Repeat entire array 3 times
b = np.tile(a, (2, 3))  # Repeat in a 2x3 grid
```

## Indexing and Slicing

### Basic Indexing

```python
# 1D arrays
a[0]  # First element
a[-1]  # Last element
a[2:5]  # Elements from index 2 to 4
a[2:]  # Elements from index 2 to end
a[:3]  # Elements from start to index 2
a[::2]  # Every second element
a[::-1]  # All elements, reversed

# Multidimensional arrays
b[0, 0]  # First element of 2D array
b[0, :]  # First row
b[:, 0]  # First column
b[0:2, 1:3]  # Block from rows 0-1 and columns 1-2
b[-1, :]  # Last row
```

### Boolean Indexing

```python
# Create mask
mask = a > 5  # Boolean array of same shape

# Apply mask
filtered = a[mask]  # Returns 1D array of elements where mask is True
filtered = a[a > 5]  # Directly filter with condition
filtered = a[(a > 5) & (a < 10)]  # Multiple conditions with & (AND)
filtered = a[(a < 5) | (a > 15)]  # Multiple conditions with | (OR)
```

### Fancy Indexing

```python
# Index with array of indices
indices = [0, 2, 4]
selected = a[indices]  # Select elements at indices 0, 2, and 4

# Multiple axes
rows = [0, 1]
cols = [1, 2]
selected = b[rows, :]  # Select rows 0 and 1
selected = b[:, cols]  # Select columns 1 and 2
selected = b[rows, cols]  # Select B[0,1] and B[1,2]
selected = b[np.ix_(rows, cols)]  # All combinations (2x2 result)
```

## Mathematical Operations

### Element-wise Operations

```python
# Basic operations
c = a + b  # Addition
c = a - b  # Subtraction
c = a * b  # Multiplication
c = a / b  # Division
c = a // b  # Floor division
c = a % b  # Modulo
c = a ** b  # Exponentiation

# In-place operations
a += b  # Equivalent to a = a + b
a -= b  # Equivalent to a = a - b

# Universal functions
c = np.add(a, b)  # Addition
c = np.subtract(a, b)  # Subtraction
c = np.multiply(a, b)  # Multiplication
c = np.divide(a, b)  # Division
c = np.floor_divide(a, b)  # Floor division
c = np.mod(a, b)  # Modulo
c = np.power(a, b)  # Exponentiation
```

### Mathematical Functions

```python
# Trigonometric functions
y = np.sin(x)
y = np.cos(x)
y = np.tan(x)
y = np.arcsin(x)
y = np.arccos(x)
y = np.arctan(x)
y = np.arctan2(y, x)  # Angle from x-axis to point (x,y)
y = np.deg2rad(angle)  # Convert degrees to radians
y = np.rad2deg(angle)  # Convert radians to degrees

# Hyperbolic functions
y = np.sinh(x)
y = np.cosh(x)
y = np.tanh(x)
y = np.arcsinh(x)
y = np.arccosh(x)
y = np.arctanh(x)

# Exponential and logarithmic functions
y = np.exp(x)  # e^x
y = np.expm1(x)  # e^x - 1
y = np.exp2(x)  # 2^x
y = np.log(x)  # Natural logarithm (ln)
y = np.log10(x)  # Base 10 logarithm
y = np.log2(x)  # Base 2 logarithm
y = np.log1p(x)  # ln(1+x)

# Arithmetic functions
y = np.add(x1, x2)
y = np.reciprocal(x)  # 1/x
y = np.negative(x)  # -x
y = np.positive(x)  # +x
y = np.multiply(x1, x2)
y = np.divide(x1, x2)
y = np.power(x1, x2)
y = np.subtract(x1, x2)

# Rounding
y = np.round(x, decimals=2)  # Round to 2 decimal places
y = np.floor(x)  # Largest integer not greater than x
y = np.ceil(x)  # Smallest integer not less than x
y = np.trunc(x)  # Truncate decimal part (towards zero)
```

### Aggregation Functions

```python
# Sum
s = np.sum(a)  # Sum of all elements
s = a.sum()  # Method form
s = np.sum(a, axis=0)  # Sum of each column
s = np.sum(a, axis=1)  # Sum of each row
s = np.sum(a, keepdims=True)  # Preserve dimensions

# Product
p = np.prod(a)  # Product of all elements
p = a.prod()  # Method form
p = np.prod(a, axis=0)  # Product of each column

# Cumulative sum and product
cs = np.cumsum(a)  # Cumulative sum
cp = np.cumprod(a)  # Cumulative product
cs = np.cumsum(a, axis=0)  # Cumulative sum along first axis

# Min and max
m = np.min(a)  # Minimum value
m = a.min()  # Method form
m = np.min(a, axis=0)  # Minimum of each column
m = np.max(a)  # Maximum value
m = a.max()  # Method form
m = np.max(a, axis=0)  # Maximum of each column

# Argmin and argmax (indices of min/max values)
idx = np.argmin(a)  # Index of minimum value
idx = a.argmin()  # Method form
idx = np.argmin(a, axis=0)  # Index of minimum in each column
idx = np.argmax(a)  # Index of maximum value
idx = a.argmax()  # Method form
idx = np.argmax(a, axis=0)  # Index of maximum in each column
```

## Statistical Functions

```python
# Basic statistics
mean = np.mean(a)
mean = a.mean()
mean = np.mean(a, axis=0)  # Mean of each column

median = np.median(a)
median = np.median(a, axis=0)  # Median of each column

std = np.std(a)  # Standard deviation
std = a.std()
std = np.std(a, axis=0)  # Standard deviation of each column

var = np.var(a)  # Variance
var = a.var()
var = np.var(a, axis=0)  # Variance of each column

# Percentiles and quantiles
q = np.percentile(a, 75)  # 75th percentile
q = np.percentile(a, [25, 50, 75])  # Multiple percentiles
q = np.percentile(a, 75, axis=0)  # 75th percentile of each column

q = np.quantile(a, 0.75)  # Same as percentile but uses 0-1 scale
q = np.quantile(a, [0.25, 0.5, 0.75])  # Multiple quantiles

# Histogram
hist, bin_edges = np.histogram(a, bins=10)
hist, bin_edges = np.histogram(a, bins=[0, 1, 2, 3])  # Custom bins

# Correlation and covariance
corr = np.corrcoef(x, y)  # Correlation coefficient matrix
cov = np.cov(x, y)  # Covariance matrix

# Additional statistics
average = np.average(a, weights=w)  # Weighted average
ptp = np.ptp(a)  # Peak-to-peak (max - min)
```

## Broadcasting

Broadcasting allows NumPy to work with arrays of different shapes during arithmetic operations.

```python
# Broadcasting examples
a = np.array([1, 2, 3])  # Shape (3,)
b = 5  # Scalar
c = a + b  # b is broadcasted to [5, 5, 5]

a = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
b = np.array([10, 20, 30])  # Shape (3,)
c = a + b  # b is broadcasted to [[10, 20, 30], [10, 20, 30]]

a = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
b = np.array([[10], [20]])  # Shape (2, 1)
c = a + b  # b is broadcasted to [[10, 10, 10], [20, 20, 20]]

# Manual broadcasting using np.broadcast_to
b_broadcast = np.broadcast_to(b, (2, 3))
```

Broadcasting rules:
1. Arrays with fewer dimensions are padded with ones on the left.
2. Arrays with shape equal to 1 in a dimension are stretched to match the other array.
3. If shapes don't match and neither is 1, an error is raised.

## Linear Algebra

NumPy provides basic linear algebra operations through the `numpy.linalg` submodule.

```python
# Matrix multiplication
c = a @ b  # Python 3.5+ syntax
c = a.dot(b)  # Method form
c = np.dot(a, b)  # Function form
c = np.matmul(a, b)  # Specialized matrix multiplication

# Matrix power
p = np.linalg.matrix_power(a, 3)  # A³

# Determinant
d = np.linalg.det(a)

# Inverse
a_inv = np.linalg.inv(a)

# Pseudo-inverse (Moore-Penrose)
a_pinv = np.linalg.pinv(a)

# Solving linear systems (x in a·x = b)
x = np.linalg.solve(a, b)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(a)
eigenvalues = np.linalg.eigvals(a)  # Only eigenvalues

# Singular Value Decomposition (SVD)
u, s, vh = np.linalg.svd(a)

# Norms
norm = np.linalg.norm(a)  # Default: Frobenius norm
norm = np.linalg.norm(a, ord=1)  # L1 norm
norm = np.linalg.norm(a, ord=np.inf)  # L-infinity norm

# QR decomposition
q, r = np.linalg.qr(a)

# Cholesky decomposition
l = np.linalg.cholesky(a)  # A must be positive-definite

# Matrix and vector products
inner = np.inner(a, b)  # Inner product
outer = np.outer(a, b)  # Outer product
cross = np.cross(a, b)  # Cross product
kron = np.kron(a, b)  # Kronecker product
tensordot = np.tensordot(a, b)  # Tensor dot product
```

## Random Numbers

NumPy provides a wide range of random number generation capabilities through `numpy.random`.

```python
# Set random seed for reproducibility
np.random.seed(42)

# Basic random sampling
r = np.random.random()  # Single value between [0, 1)
r = np.random.random(5)  # Array of 5 values between [0, 1)
r = np.random.random((2, 3))  # 2x3 array between [0, 1)

# Random integers
r = np.random.randint(10)  # Single int between [0, 10)
r = np.random.randint(3, 10)  # Single int between [3, 10)
r = np.random.randint(3, 10, 5)  # 5 ints between [3, 10)
r = np.random.randint(3, 10, (2, 3))  # 2x3 array of ints between [3, 10)

# Random choice
r = np.random.choice(5)  # Single item from [0, 1, 2, 3, 4]
r = np.random.choice([1, 5, 9])  # Single item from given array
r = np.random.choice(5, 3)  # 3 items from [0, 1, 2, 3, 4]
r = np.random.choice(5, 3, replace=False)  # Without replacement
r = np.random.choice(5, 3, p=[0.1, 0.2, 0.3, 0.2, 0.2])  # With probability weights

# Shuffle
a = np.arange(10)
np.random.shuffle(a)  # In-place shuffle

# Permutation
p = np.random.permutation(10)  # Permutation of [0, 1, ..., 9]
p = np.random.permutation(a)  # Permutation of a (not in-place)

# Distributions
# Normal (Gaussian) distribution
r = np.random.normal()  # Mean=0, SD=1
r = np.random.normal(loc=5, scale=2)  # Mean=5, SD=2
r = np.random.normal(loc=5, scale=2, size=(2, 3))  # 2x3 array

# Uniform distribution
r = np.random.uniform()  # Range [0, 1)
r = np.random.uniform(low=1, high=10)  # Range [1, 10)
r = np.random.uniform(low=1, high=10, size=(2, 3))  # 2x3 array

# Other distributions
r = np.random.binomial(n=10, p=0.5, size=5)  # Binomial
r = np.random.poisson(lam=5, size=5)  # Poisson
r = np.random.exponential(scale=1.0, size=5)  # Exponential
r = np.random.beta(a=1, b=10, size=5)  # Beta
r = np.random.gamma(shape=2, scale=2, size=5)  # Gamma
r = np.random.chisquare(df=2, size=5)  # Chi-square
r = np.random.standard_t(df=10, size=5)  # Student's t
r = np.random.f(dfnum=5, dfden=10, size=5)  # F-distribution
r = np.random.lognormal(mean=0, sigma=1, size=5)  # Log-normal
r = np.random.rayleigh(scale=1, size=5)  # Rayleigh
```

### The New Generator API (NumPy 1.17+)

```python
# Create a generator
rng = np.random.default_rng(42)  # With seed

# Basic random sampling
r = rng.random()
r = rng.random((2, 3))

# Integers
r = rng.integers(10)  # [0, 10] inclusive
r = rng.integers(3, 10, (2, 3))

# Choice
r = rng.choice(5, 3)
r = rng.choice(5, 3, replace=False)
r = rng.choice(5, 3, p=[0.1, 0.2, 0.3, 0.2, 0.2])

# Shuffle and permutation
rng.shuffle(a)  # In-place
p = rng.permutation(10)

# Distributions
r = rng.normal(5, 2, (2, 3))
r = rng.uniform(1, 10, (2, 3))
```

## Advanced Indexing

### Boolean Masking

```python
# Creating masks
mask = (a > 5) & (a < 10)  # AND
mask = (a < 5) | (a > 15)  # OR
mask = ~(a > 5)  # NOT

# Apply masks
filtered = a[mask]
a[mask] = 0  # Set masked elements to 0
```

### Using Where

```python
# Conditional assignment
result = np.where(a > 5, a, 0)  # If a > 5, keep value, else 0
result = np.where(a > 5, a, b)  # If a > 5, use a, else use b

# Find indices where condition is true
indices = np.where(a > 5)
```

### Advanced Selection and Assignment

```python
# Select with a list of points
points = np.array([0, 1, 2])
values = a[points, points]  # Diagonal elements a[0,0], a[1,1], a[2,2]

# Assigning with advanced indexing
a[a < 0] = 0  # Replace negative values with 0
a[[0, 1, 2], [1, 2, 0]] = 10  # Set specific elements to 10
a[[0, 1, 2]] = [10, 20, 30]  # Set specific rows
```

## Input and Output

### Saving and Loading Array Data

```python
# Binary format (.npy)
np.save('array.npy', a)  # Save single array
loaded_a = np.load('array.npy')

# Multiple arrays (.npz)
np.savez('arrays.npz', a=a, b=b)  # Save multiple arrays
loaded = np.load('arrays.npz')
loaded_a = loaded['a']
loaded_b = loaded['b']

# Compressed format
np.savez_compressed('arrays_compressed.npz', a=a, b=b)

# Text formats
np.savetxt('array.txt', a, delimiter=',')
np.savetxt('array.csv', a, delimiter=',')
loaded_a = np.loadtxt('array.txt', delimiter=',')
loaded_a = np.genfromtxt('array.csv', delimiter=',')
```

## Performance Tips

```python
# Pre-allocate arrays
result = np.zeros((1000, 1000))  # Better than growing arrays dynamically

# Use vectorized operations
# Instead of:
# for i in range(len(a)):
#     result[i] = a[i] * 2 + 5
# Use:
result = a * 2 + 5

# Use specialized functions
# Instead of:
# c = np.zeros_like(a)
# for i in range(len(a)-1):
#     c[i] = a[i+1] - a[i]
# Use:
c = np.diff(a)

# Use views instead of copies when possible
b = a[::2]  # View
b = np.ascontiguousarray(a[::2])  # Force a contiguous copy if needed

# Use fast dtypes
a = np.array([1, 2, 3], dtype=np.float32)  # Often faster than float64
b = np.array([1, 2, 3], dtype=np.int32)  # Often faster than int64

# Use stride tricks for advanced operations
from numpy.lib import stride_tricks
sliding_window = stride_tricks.sliding_window_view(a, 3)  # Sliding window view
```

## Working with Other Libraries

### With Pandas

```python
import pandas as pd

# NumPy array from Pandas DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
arr = df.to_numpy()  # Recommended
arr = df.values  # Also works

# Pandas DataFrame from NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
```

### With Matplotlib

```python
import matplotlib.pyplot as plt

# Basic plotting
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()

# Image display
img = np.random.rand(10, 10)
plt.imshow(img, cmap='viridis')
plt.colorbar()
plt.show()
```

### With SciPy

```python
from scipy import stats, optimize, interpolate

# Statistics
z_scores = stats.zscore(a)
p_value = stats.ttest_ind(a, b).pvalue

# Optimization
def f(x): return x**2 + 5*np.sin(x)
result = optimize.minimize(f, x0=0)

# Interpolation
f = interpolate.interp1d(x, y, kind='cubic')
y_new = f(x_new)
```