# Pandas Cheat Sheet

![Pandas Logo](https://pandas.pydata.org/static/img/pandas_mark.svg)

## Table of Contents
- [Installation](#installation)
- [Data Import/Export](#data-importexport)
- [Data Creation](#data-creation)
- [Data Inspection](#data-inspection)
- [Data Selection](#data-selection)
- [Data Cleaning](#data-cleaning)
- [Data Transformation](#data-transformation)
- [String Operations](#string-operations)
- [Time Series](#time-series)
- [Grouping and Aggregation](#grouping-and-aggregation)
- [Merging and Joining](#merging-and-joining)
- [Statistical Functions](#statistical-functions)
- [Plotting](#plotting)
- [Performance Tips](#performance-tips)
- [Best Practices](#best-practices)

## Installation

```python
# Using pip
pip install pandas

# Using conda
conda install pandas
```

Import conventions:

```python
# Standard imports
import pandas as pd
import numpy as np
```

## Data Import/Export

### Reading Data

```python
# CSV
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', sep=';', decimal=',', encoding='utf-8')

# Excel
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df = pd.read_excel('file.xlsx', sheet_name=0)  # First sheet

# JSON
df = pd.read_json('file.json')

# SQL
from sqlalchemy import create_engine
engine = create_engine('sqlite:///database.db')
df = pd.read_sql('SELECT * FROM table', engine)
df = pd.read_sql_query('SELECT * FROM table', engine)

# HTML tables
df = pd.read_html('https://example.com/table.html')[0]  # Returns a list of tables

# Parquet
df = pd.read_parquet('file.parquet')

# HDF5
df = pd.read_hdf('file.h5', key='df')

# Feather
df = pd.read_feather('file.feather')

# Clipboard
df = pd.read_clipboard()
```

### Writing Data

```python
# CSV
df.to_csv('file.csv', index=False)
df.to_csv('file.csv', sep=';', decimal=',', encoding='utf-8')

# Excel
df.to_excel('file.xlsx', sheet_name='Sheet1', index=False)
writer = pd.ExcelWriter('file.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()

# JSON
df.to_json('file.json')
df.to_json('file.json', orient='records')

# SQL
df.to_sql('table_name', engine, if_exists='replace', index=False)

# Parquet
df.to_parquet('file.parquet')

# HDF5
df.to_hdf('file.h5', key='df', mode='w')

# Feather
df.to_feather('file.feather')

# Clipboard
df.to_clipboard()
```

## Data Creation

### Creating DataFrames

```python
# From a dictionary
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c'],
    'C': [1.0, 2.0, 3.0]
})

# From a list of lists
df = pd.DataFrame([
    [1, 'a', 1.0],
    [2, 'b', 2.0],
    [3, 'c', 3.0]
], columns=['A', 'B', 'C'])

# From NumPy array
df = pd.DataFrame(np.random.randn(3, 3), columns=['A', 'B', 'C'])

# With custom indices
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c']
}, index=['x', 'y', 'z'])
```

### Creating Series

```python
# From a list
s = pd.Series([1, 2, 3, 4])

# From a dictionary
s = pd.Series({'a': 1, 'b': 2, 'c': 3})

# With custom index
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

# From scalar value
s = pd.Series(5, index=['a', 'b', 'c'])  # [5, 5, 5]
```

### Date Range

```python
# Create a date range
dates = pd.date_range('2023-01-01', periods=10)  # Daily frequency by default
dates = pd.date_range('2023-01-01', '2023-01-10')
dates = pd.date_range('2023-01-01', periods=10, freq='B')  # Business days
dates = pd.date_range('2023-01-01', periods=10, freq='H')  # Hourly
dates = pd.date_range('2023-01-01', periods=10, freq='M')  # Month end
dates = pd.date_range('2023-01-01', periods=10, freq='MS')  # Month start
```

## Data Inspection

### Basic Information

```python
# Shape, size, and data types
df.shape  # (rows, columns)
df.size   # Total number of elements (rows * columns)
df.dtypes  # Data types of each column

# Basic info
df.info()  # Summary including memory usage

# Memory usage
df.memory_usage(deep=True)  # Detailed memory usage

# Basic statistics
df.describe()  # Statistical summary of numerical columns
df.describe(include='all')  # Summary of all columns

# Unique values
df['column'].unique()  # Array of unique values
df['column'].nunique()  # Count of unique values
df['column'].value_counts()  # Count of occurrences of each value
```

### Viewing Data

```python
# Heads and tails
df.head()  # First 5 rows
df.head(10)  # First 10 rows
df.tail()  # Last 5 rows
df.tail(10)  # Last 10 rows

# Sample
df.sample(5)  # Random sample of 5 rows
df.sample(frac=0.1)  # Random sample of 10% of rows

# Display all columns
pd.set_option('display.max_columns', None)

# Display all rows
pd.set_option('display.max_rows', None)

# Display settings
pd.get_option('display.max_rows')
pd.set_option('display.max_rows', 100)
pd.reset_option('display.max_rows')
```

## Data Selection

### Selection by Position

```python
# Single value by index
df.iloc[0, 0]  # First row, first column

# Rows by index
df.iloc[0]  # First row
df.iloc[-1]  # Last row
df.iloc[0:5]  # First 5 rows
df.iloc[::2]  # Every other row

# Columns by index
df.iloc[:, 0]  # First column
df.iloc[:, -1]  # Last column
df.iloc[:, 0:2]  # First 2 columns

# Subset by row and column indices
df.iloc[0:3, 0:3]  # First 3 rows and columns
df.iloc[[0, 2, 5], [1, 3]]  # Specific rows and columns
```

### Selection by Label

```python
# Single value by label
df.loc['row_label', 'column_label']

# Rows by label
df.loc['row_label']
df.loc[['row1', 'row2']]
df.loc['row1':'row3']  # Inclusive

# Columns by label
df.loc[:, 'column_label']
df.loc[:, ['col1', 'col2']]
df.loc[:, 'col1':'col3']  # Inclusive

# Subset by row and column labels
df.loc['row1':'row3', 'col1':'col3']
```

### Column Selection

```python
# Single column (returns Series)
df['column']  # or df.column for valid python identifiers

# Multiple columns (returns DataFrame)
df[['col1', 'col2']]
```

### Boolean Indexing

```python
# Filter rows by condition
df[df['A'] > 0]
df[(df['A'] > 0) & (df['B'] < 0)]  # AND condition
df[(df['A'] > 0) | (df['B'] < 0)]  # OR condition
df[~(df['A'] > 0)]  # NOT condition

# Filter rows by multiple conditions
mask = (df['A'] > 0) & (df['B'] < 0)
df[mask]

# Filter with isin
df[df['A'].isin([1, 2, 3])]
df[~df['A'].isin([1, 2, 3])]  # NOT in

# Query method (more readable for complex conditions)
df.query('A > 0 and B < 0')
df.query('A in [1, 2, 3]')
```

### Setting Values

```python
# Set value by position
df.iloc[0, 0] = 100

# Set value by label
df.loc['row_label', 'column_label'] = 100

# Set column values
df['column'] = 10
df['column'] = [1, 2, 3, 4, 5]
df['column'] = np.array([1, 2, 3, 4, 5])
df['column'] = pd.Series([1, 2, 3, 4, 5])

# Set values using mask
df.loc[df['A'] > 0, 'B'] = 10
```

## Data Cleaning

### Handling Missing Values

```python
# Check for missing values
df.isna()  # Returns boolean DataFrame
df.isnull()  # Same as isna()
df.isna().sum()  # Count of missing values per column
df.isna().any()  # Check if any value is missing in each column
df.isna().all()  # Check if all values are missing in each column

# Drop missing values
df.dropna()  # Drop rows with any missing values
df.dropna(axis=1)  # Drop columns with any missing values
df.dropna(how='all')  # Drop rows with all missing values
df.dropna(subset=['A', 'B'])  # Drop rows with missing values in specific columns
df.dropna(thresh=2)  # Drop rows with fewer than 2 non-missing values

# Fill missing values
df.fillna(0)  # Fill all missing values with 0
df['A'].fillna(df['A'].mean())  # Fill missing values with mean
df.fillna(method='ffill')  # Forward fill (use previous value)
df.fillna(method='bfill')  # Backward fill (use next value)
df.fillna({'A': 0, 'B': 1})  # Fill different values by column
```

### Duplicate Data

```python
# Check for duplicates
df.duplicated()  # Returns boolean Series
df.duplicated().sum()  # Count of duplicate rows

# Drop duplicates
df.drop_duplicates()  # Drop duplicate rows
df.drop_duplicates(subset=['A', 'B'])  # Consider only specific columns
df.drop_duplicates(keep='first')  # Keep first occurrence (default)
df.drop_duplicates(keep='last')  # Keep last occurrence
df.drop_duplicates(keep=False)  # Drop all duplicates
```

### Data Type Conversion

```python
# Convert data types
df['A'] = df['A'].astype('int64')
df['B'] = df['B'].astype('float64')
df['C'] = df['C'].astype('str')
df['D'] = df['D'].astype('category')

# Convert multiple columns
df = df.astype({'A': 'int64', 'B': 'float64', 'C': 'str'})

# Convert to numeric (handling errors)
pd.to_numeric(df['A'])
pd.to_numeric(df['A'], errors='coerce')  # Invalid values become NaN
pd.to_numeric(df['A'], errors='ignore')  # Invalid values remain as is

# Convert to datetime
pd.to_datetime(df['date'])
pd.to_datetime(df['date'], format='%Y-%m-%d')
pd.to_datetime(df['date'], errors='coerce')  # Invalid dates become NaT
```

## Data Transformation

### Row and Column Operations

```python
# Add a new column
df['new_col'] = df['A'] + df['B']
df['new_col'] = 10
df['new_col'] = np.nan

# Delete columns
df = df.drop(columns=['A', 'B'])
df = df.drop('A', axis=1)
del df['A']

# Rename columns
df = df.rename(columns={'A': 'a', 'B': 'b'})
df.columns = ['a', 'b', 'c']

# Rename index
df = df.rename(index={0: 'x', 1: 'y'})

# Reset index
df = df.reset_index()  # Turn index into column
df = df.reset_index(drop=True)  # Drop old index

# Set index
df = df.set_index('A')
df = df.set_index(['A', 'B'])  # MultiIndex
```

### Sorting

```python
# Sort by values
df = df.sort_values('A')
df = df.sort_values(['A', 'B'])  # Sort by multiple columns
df = df.sort_values('A', ascending=False)
df = df.sort_values(['A', 'B'], ascending=[True, False])

# Sort by index
df = df.sort_index()
df = df.sort_index(ascending=False)
```

### Applying Functions

```python
# Apply function to each element
df = df.applymap(lambda x: x**2)

# Apply function to each column
df = df.apply(np.sum)
df = df.apply(lambda x: x.max() - x.min())

# Apply function to each row
df = df.apply(np.sum, axis=1)
df = df.apply(lambda x: x.max() - x.min(), axis=1)

# Apply function to a single column
df['A'] = df['A'].apply(lambda x: x**2)

# Apply function with additional arguments
df['A'] = df['A'].apply(lambda x, y: x * y, y=10)
```

### Mapping Values

```python
# Map values in a column
mapping = {1: 'a', 2: 'b', 3: 'c'}
df['A'] = df['A'].map(mapping)

# Replace values
df = df.replace(1, 10)
df = df.replace([1, 2, 3], [10, 20, 30])
df = df.replace({1: 10, 2: 20, 3: 30})
```

### Transposing Data

```python
# Transpose DataFrame
df = df.T  # Swap rows and columns
```

## String Operations

String operations are performed on Series with string values using the `str` accessor.

```python
# Basic string methods
s = pd.Series(['foo', 'bar', 'baz'])
s.str.upper()  # ['FOO', 'BAR', 'BAZ']
s.str.len()  # [3, 3, 3]
s.str.strip()  # Remove whitespace
s.str.replace('a', 'z')  # Replace substrings

# Pattern matching
s.str.contains('a')  # Boolean mask
s.str.startswith('f')  # Boolean mask
s.str.endswith('z')  # Boolean mask

# Extracting substrings
s.str[0]  # First character of each string
s.str.slice(0, 2)  # First two characters
s.str.extract('(a)')  # Extract pattern (returns DataFrame)
s.str.extractall('(a)')  # Extract all occurrences (returns DataFrame)

# Split strings
s.str.split('a')  # Split on character
s.str.split('a', expand=True)  # Split and expand into columns

# String concatenation
s.str.cat(sep=', ')  # Concatenate all strings
s.str.cat(['x', 'y', 'z'], sep='-')  # Concatenate with another list
```

## Time Series

### Date and Time Components

```python
# Extract components from datetime
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday  # 0 is Monday, 6 is Sunday
df['weekday_name'] = df['date'].dt.day_name()
df['quarter'] = df['date'].dt.quarter

# Extract time components
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute
df['second'] = df['date'].dt.second

# Boolean properties
df['is_month_end'] = df['date'].dt.is_month_end
df['is_month_start'] = df['date'].dt.is_month_start
df['is_quarter_end'] = df['date'].dt.is_quarter_end
df['is_year_end'] = df['date'].dt.is_year_end
```

### Time Series Operations

```python
# Shift (lag) values
df['previous'] = df['value'].shift(1)  # Shift down by 1
df['next'] = df['value'].shift(-1)  # Shift up by 1

# Difference between values
df['diff'] = df['value'].diff()  # Difference with previous row
df['diff_2'] = df['value'].diff(2)  # Difference with row 2 steps before

# Percentage change
df['pct_change'] = df['value'].pct_change()
df['pct_change_2'] = df['value'].pct_change(2)

# Rolling windows
df['rolling_mean'] = df['value'].rolling(window=3).mean()
df['rolling_std'] = df['value'].rolling(window=3).std()
df['rolling_sum'] = df['value'].rolling(window=3).sum()

# Expanding windows (cumulative)
df['cumsum'] = df['value'].expanding().sum()
df['cummean'] = df['value'].expanding().mean()
df['cummax'] = df['value'].expanding().max()
```

### Resampling Time Series

```python
# Downsample to lower frequency
df_daily = df.resample('D').mean()  # Daily
df_monthly = df.resample('M').mean()  # Monthly (month end)
df_monthly = df.resample('MS').mean()  # Monthly (month start)
df_quarterly = df.resample('Q').mean()  # Quarterly
df_annual = df.resample('A').mean()  # Annual (year end)

# Upsample to higher frequency
df_hourly = df_daily.resample('H').ffill()  # Forward fill
df_hourly = df_daily.resample('H').bfill()  # Backward fill
df_hourly = df_daily.resample('H').interpolate()  # Interpolate values

# Resampling with different methods
df_daily = df.resample('D').sum()
df_daily = df.resample('D').agg(['min', 'max', 'mean'])
df_daily = df.resample('D').agg({'A': 'sum', 'B': 'mean'})
```

## Grouping and Aggregation

### Groupby Operations

```python
# Group by single column
grouped = df.groupby('A')

# Group by multiple columns
grouped = df.groupby(['A', 'B'])

# Iterate through groups
for name, group in grouped:
    print(name)
    print(group)

# Access specific group
grouped.get_group('value')

# Basic aggregations
grouped.count()  # Count non-NA values
grouped.sum()
grouped.mean()
grouped.median()
grouped.min()
grouped.max()
grouped.std()
grouped.var()
grouped.first()  # First value in each group
grouped.last()  # Last value in each group

# Multiple aggregations
grouped.agg(['sum', 'mean', 'count'])
grouped.agg({'A': 'sum', 'B': ['min', 'max']})

# Custom aggregations
grouped.agg(lambda x: x.max() - x.min())
```

### Transformation and Filtering

```python
# Transform each group
df['mean_by_group'] = df.groupby('A')['B'].transform('mean')
df['rank_in_group'] = df.groupby('A')['B'].transform(lambda x: x.rank())

# Filter groups
df.groupby('A').filter(lambda x: x['B'].mean() > 0)
```

### Pivot Tables

```python
# Basic pivot table
pivot = pd.pivot_table(df, values='D', index='A', columns='B', aggfunc='mean')

# Pivot table with multiple values and aggregations
pivot = pd.pivot_table(
    df, 
    values=['D', 'E'], 
    index=['A', 'B'], 
    columns='C', 
    aggfunc={'D': 'mean', 'E': 'sum'},
    fill_value=0,
    margins=True  # Add row and column totals
)

# Reshape from long to wide format
df_wide = df.pivot(index='date', columns='category', values='value')

# Reshape from wide to long format
df_long = pd.melt(
    df_wide, 
    id_vars=['date'],
    value_vars=['A', 'B', 'C'],
    var_name='category',
    value_name='value'
)
```

## Merging and Joining

### Combining DataFrames

```python
# Concatenate DataFrames
combined = pd.concat([df1, df2])  # Vertically (default)
combined = pd.concat([df1, df2], axis=1)  # Horizontally
combined = pd.concat([df1, df2], ignore_index=True)  # Reset indices

# Append rows
df1 = df1.append(df2)  # Deprecated in newer versions
df1 = pd.concat([df1, df2])  # Preferred method

# Append a single row
df = df.append({'A': 1, 'B': 2}, ignore_index=True)
```

### Database-Style Joins

```python
# Inner join (default)
merged = pd.merge(df1, df2, on='key')
merged = pd.merge(df1, df2, on=['key1', 'key2'])

# Left join
merged = pd.merge(df1, df2, on='key', how='left')

# Right join
merged = pd.merge(df1, df2, on='key', how='right')

# Outer join
merged = pd.merge(df1, df2, on='key', how='outer')

# Joining on different column names
merged = pd.merge(df1, df2, left_on='key1', right_on='key2')

# Join on indices
merged = pd.merge(df1, df2, left_index=True, right_index=True)
merged = pd.merge(df1, df2, left_on='key', right_index=True)

# Specify suffixes for overlapping columns
merged = pd.merge(df1, df2, on='key', suffixes=('_left', '_right'))
```

## Statistical Functions

```python
# Basic statistics
df.mean()
df.median()
df.min()
df.max()
df.std()
df.var()
df.sem()  # Standard error of the mean
df.describe()  # Summary statistics

# Correlation and covariance
df.corr()  # Correlation matrix
df.cov()  # Covariance matrix
df['A'].corr(df['B'])  # Correlation between two columns
df['A'].cov(df['B'])  # Covariance between two columns

# Ranking
df.rank()  # Default: average method
df.rank(method='min')  # Minimum rank
df.rank(method='max')  # Maximum rank
df.rank(method='first')  # First occurrence
df.rank(method='dense')  # No gaps in ranking

# Rolling and expanding statistics
df.rolling(window=3).mean()
df.expanding().mean()

# Cumulative statistics
df.cumsum()
df.cumprod()
df.cummax()
df.cummin()
```

## Plotting

Pandas integrates with Matplotlib for easy plotting.

```python
# Import pyplot
import matplotlib.pyplot as plt

# Line plot
df.plot()
df.plot(x='A', y='B')

# Bar plot
df.plot.bar()
df.plot.barh()  # Horizontal bar plot
df['A'].value_counts().plot.bar()

# Pie chart
df.plot.pie(y='A')

# Histogram
df.plot.hist(bins=20)
df['A'].plot.hist(bins=20)

# Box plot
df.plot.box()

# Scatter plot
df.plot.scatter(x='A', y='B')

# Area plot
df.plot.area()

# Density plot
df.plot.density()

# Hexbin plot (for bivariate data)
df.plot.hexbin(x='A', y='B', gridsize=15)

# Customize plot
df.plot(
    figsize=(10, 6),
    title='Plot Title',
    grid=True,
    legend=True,
    style='.-',
    color=['r', 'g', 'b'],
    alpha=0.7
)

# Save plot
plt.savefig('plot.png', dpi=300)
```

## Performance Tips

```python
# Use efficient data types
df['category_col'] = df['category_col'].astype('category')
df['int_col'] = df['int_col'].astype('int32')  # Smaller integer type
df['float_col'] = df['float_col'].astype('float32')  # Smaller float type

# Check memory usage
df.memory_usage(deep=True)

# Use inplace operations (carefully)
df.dropna(inplace=True)
df.fillna(0, inplace=True)
df.reset_index(inplace=True)

# Vectorized operations
# Instead of:
for i in range(len(df)):
    df.iloc[i, 0] = df.iloc[i, 1] + df.iloc[i, 2]
# Use:
df.iloc[:, 0] = df.iloc[:, 1] + df.iloc[:, 2]

# Use query for complex filtering
# Instead of:
df[(df['A'] > 0) & (df['B'] < 0) & (df['C'] == 0)]
# Use:
df.query('A > 0 and B < 0 and C == 0')

# Use numba for complex custom functions
from numba import jit

@jit(nopython=True)
def complex_function(x):
    # calculations
    return result

df['result'] = df['A'].apply(complex_function)
```

## Best Practices

### Style and Readability

```python
# Use method chaining
df = (
    df
    .drop_duplicates()
    .fillna(0)
    .sort_values('A')
    .reset_index(drop=True)
)

# Create intermediate variables for complex operations
mask = (df['A'] > 0) & (df['B'] < 0)
df_filtered = df[mask]
result = df_filtered.mean()

# Use named aggregations (pandas 0.25+)
df.groupby('A').agg(
    min_B=('B', 'min'),
    max_B=('B', 'max'),
    avg_C=('C', 'mean')
)
```

### Working with Large Data

```python
# Read data in chunks
chunks = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in chunks:
    # Process each chunk
    processed_chunk = process_function(chunk)
    # Append to result or write to file
    processed_chunk.to_csv('output.csv', mode='a', header=False)

# Use dask for larger-than-memory data
import dask.dataframe as dd
ddf = dd.read_csv('large_file.csv')
result = ddf.groupby('A').mean().compute()
```

### Debugging

```python
# Use info() to understand data structure
df.info()

# Check for NaN values
df.isna().sum()

# Use sample() to inspect data
df.sample(5)

# Check for duplicate rows
df.duplicated().sum()
```

### Configuration

```python
# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)
pd.set_option('display.float_format', '{:.2f}'.format)  # 2 decimal places

# Reset to default
pd.reset_option('all')
```