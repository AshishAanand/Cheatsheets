# Python Cheat Sheet

A comprehensive reference guide for Python programming essentials.

## Table of Contents
- [Basic Syntax](#basic-syntax)
- [Data Types](#data-types)
- [Variables & Operations](#variables--operations)
- [Control Flow](#control-flow)
- [Functions](#functions)
- [Data Structures](#data-structures)
- [Object-Oriented Programming](#object-oriented-programming)
- [File I/O](#file-io)
- [Error Handling](#error-handling)
- [Modules & Packages](#modules--packages)
- [List Comprehensions](#list-comprehensions)
- [Lambda Functions](#lambda-functions)
- [Decorators](#decorators)
- [Common Built-in Functions](#common-built-in-functions)
- [Popular Libraries](#popular-libraries)

## Basic Syntax

```python
# Comments
# This is a single-line comment

"""
This is a multi-line comment
or docstring
"""

# Print statement
print("Hello, World!")
print("Value:", 42)
print(f"Formatted: {variable}")  # f-string

# Multiple statements on one line
x = 1; y = 2; z = 3

# Line continuation
total = 1 + 2 + 3 + \
        4 + 5 + 6

# Indentation (4 spaces or 1 tab)
if True:
    print("Indented block")
```

## Data Types

```python
# Numbers
integer = 42
float_num = 3.14
complex_num = 3 + 4j

# Strings
single_quote = 'Hello'
double_quote = "World"
multi_line = """This is a
multi-line string"""
raw_string = r"C:\Users\name"  # Raw string

# Boolean
is_true = True
is_false = False

# None
empty_value = None

# Type checking
print(type(42))  # <class 'int'>
print(isinstance(42, int))  # True
```

## Variables & Operations

```python
# Variable assignment
x = 10
y = 20
name = "Python"

# Multiple assignment
a, b, c = 1, 2, 3
x = y = z = 0

# Arithmetic operators
addition = 5 + 3        # 8
subtraction = 5 - 3     # 2
multiplication = 5 * 3  # 15
division = 5 / 3        # 1.666...
floor_division = 5 // 3 # 1
modulus = 5 % 3         # 2
exponent = 5 ** 3       # 125

# Comparison operators
equal = (5 == 3)        # False
not_equal = (5 != 3)    # True
greater = (5 > 3)       # True
less_equal = (5 <= 3)   # False

# Logical operators
and_op = True and False  # False
or_op = True or False    # True
not_op = not True        # False

# Assignment operators
x += 5  # x = x + 5
x -= 3  # x = x - 3
x *= 2  # x = x * 2
```

## Control Flow

```python
# If statements
age = 18
if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

# Ternary operator
status = "Adult" if age >= 18 else "Minor"

# For loops
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

for item in ['a', 'b', 'c']:
    print(item)

for i, value in enumerate(['a', 'b', 'c']):
    print(f"{i}: {value}")

# While loops
count = 0
while count < 5:
    print(count)
    count += 1

# Break and continue
for i in range(10):
    if i == 3:
        continue  # Skip this iteration
    if i == 7:
        break     # Exit loop
    print(i)
```

## Functions

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

result = greet("Alice")

# Default arguments
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Variable arguments
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4))  # 10

# Keyword arguments
def create_profile(**kwargs):
    return kwargs

profile = create_profile(name="John", age=30, city="NYC")

# Mixed arguments
def mixed_func(required, default="default", *args, **kwargs):
    print(f"Required: {required}")
    print(f"Default: {default}")
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

# Lambda functions
square = lambda x: x ** 2
add = lambda x, y: x + y
```

## Data Structures

### Lists
```python
# Creating lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# List operations
numbers.append(6)           # Add to end
numbers.insert(0, 0)        # Insert at index
numbers.remove(3)           # Remove first occurrence
popped = numbers.pop()      # Remove and return last
numbers.extend([7, 8, 9])   # Add multiple items

# List methods
print(len(numbers))         # Length
print(numbers.count(2))     # Count occurrences
print(numbers.index(4))     # Find index
numbers.sort()              # Sort in place
numbers.reverse()           # Reverse in place

# Slicing
print(numbers[1:4])         # Elements 1-3
print(numbers[:3])          # First 3 elements
print(numbers[2:])          # From index 2 to end
print(numbers[-1])          # Last element
print(numbers[::2])         # Every 2nd element
```

### Dictionaries
```python
# Creating dictionaries
person = {"name": "Alice", "age": 30, "city": "NYC"}
empty_dict = {}

# Dictionary operations
person["email"] = "alice@email.com"  # Add/update
age = person.get("age", 0)           # Get with default
del person["city"]                   # Delete key

# Dictionary methods
keys = person.keys()
values = person.values()
items = person.items()

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
```

### Tuples
```python
# Creating tuples
coordinates = (10, 20)
single_item = (42,)  # Note the comma
empty_tuple = ()

# Tuple unpacking
x, y = coordinates
```

### Sets
```python
# Creating sets
fruits = {"apple", "banana", "orange"}
empty_set = set()

# Set operations
fruits.add("grape")
fruits.remove("banana")
fruits.discard("kiwi")  # Won't error if not found

# Set operations
set1 = {1, 2, 3}
set2 = {3, 4, 5}
union = set1 | set2         # {1, 2, 3, 4, 5}
intersection = set1 & set2  # {3}
difference = set1 - set2    # {1, 2}
```

## Object-Oriented Programming

```python
class Animal:
    # Class variable
    species_count = 0
    
    def __init__(self, name, species):
        # Instance variables
        self.name = name
        self.species = species
        Animal.species_count += 1
    
    def speak(self):
        return f"{self.name} makes a sound"
    
    @classmethod
    def get_species_count(cls):
        return cls.species_count
    
    @staticmethod
    def animal_info():
        return "Animals are living organisms"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Canine")
        self.breed = breed
    
    def speak(self):  # Method overriding
        return f"{self.name} barks"

# Usage
dog = Dog("Buddy", "Golden Retriever")
print(dog.speak())  # Buddy barks
print(Animal.get_species_count())  # 1
```

## File I/O

```python
# Reading files
with open("file.txt", "r") as file:
    content = file.read()           # Read entire file
    
with open("file.txt", "r") as file:
    lines = file.readlines()        # Read all lines
    
with open("file.txt", "r") as file:
    for line in file:               # Read line by line
        print(line.strip())

# Writing files
with open("output.txt", "w") as file:
    file.write("Hello, World!")
    file.writelines(["Line 1\n", "Line 2\n"])

# Appending to files
with open("output.txt", "a") as file:
    file.write("\nAppended text")

# Working with CSV
import csv
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# Working with JSON
import json
data = {"name": "Alice", "age": 30}
with open("data.json", "w") as file:
    json.dump(data, file)

with open("data.json", "r") as file:
    loaded_data = json.load(file)
```

## Error Handling

```python
# Basic try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
try:
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Invalid input!")
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")

# Finally block
try:
    file = open("file.txt", "r")
    # Process file
except FileNotFoundError:
    print("File not found!")
finally:
    if 'file' in locals() and not file.closed:
        file.close()

# Raising exceptions
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age
```

## Modules & Packages

```python
# Importing modules
import math
import datetime as dt
from random import randint, choice
from collections import *

# Using imported modules
print(math.pi)
now = dt.datetime.now()
random_num = randint(1, 10)

# Creating your own module (mymodule.py)
def my_function():
    return "Hello from my module!"

PI = 3.14159

# Using your module
import mymodule
print(mymodule.my_function())
print(mymodule.PI)
```

## List Comprehensions

```python
# Basic list comprehension
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Nested comprehension
matrix = [[i*j for j in range(3)] for i in range(3)]

# Dictionary comprehension
word_lengths = {word: len(word) for word in ["hello", "world", "python"]}

# Set comprehension
unique_squares = {x**2 for x in range(-5, 6)}
```

## Lambda Functions

```python
# Basic lambda
square = lambda x: x**2
add = lambda x, y: x + y

# With map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))

# With filter
evens = list(filter(lambda x: x % 2 == 0, numbers))

# With sorted
students = [("Alice", 85), ("Bob", 90), ("Charlie", 78)]
sorted_by_grade = sorted(students, key=lambda x: x[1])
```

## Decorators

```python
# Basic decorator
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")
```

## Common Built-in Functions

```python
# Essential built-in functions
len([1, 2, 3])              # Length: 3
max([1, 5, 3])              # Maximum: 5
min([1, 5, 3])              # Minimum: 1
sum([1, 2, 3])              # Sum: 6
abs(-5)                     # Absolute value: 5
round(3.14159, 2)           # Round: 3.14

# Type conversion
int("123")                  # String to int
float("3.14")               # String to float
str(123)                    # Number to string
list("hello")               # String to list
tuple([1, 2, 3])            # List to tuple

# Iteration functions
range(5)                    # 0, 1, 2, 3, 4
enumerate(["a", "b", "c"])  # (0, 'a'), (1, 'b'), (2, 'c')
zip([1, 2, 3], ["a", "b", "c"])  # (1, 'a'), (2, 'b'), (3, 'c')

# Functional programming
map(lambda x: x**2, [1, 2, 3])      # Apply function to each item
filter(lambda x: x > 0, [-1, 0, 1, 2])  # Filter items
all([True, True, False])             # All items True? False
any([False, False, True])            # Any item True? True
```

## Popular Libraries

### NumPy
```python
import numpy as np

# Arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Array operations
np.zeros(5)         # Array of zeros
np.ones((2, 3))     # 2x3 array of ones
np.arange(0, 10, 2) # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)# 5 evenly spaced numbers
```

### Pandas
```python
import pandas as pd

# DataFrames
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NYC', 'LA', 'Chicago']
})

# Basic operations
df.head()           # First 5 rows
df.info()           # DataFrame info
df.describe()       # Statistics
df['Age'].mean()    # Column average
```

### Requests
```python
import requests

# HTTP requests
response = requests.get('https://api.github.com/users/octocat')
data = response.json()
print(response.status_code)
```

### Matplotlib
```python
import matplotlib.pyplot as plt

# Basic plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('My Plot')
plt.show()
```

## Quick Tips

### String Formatting
```python
name = "Alice"
age = 30

# f-strings (Python 3.6+)
print(f"My name is {name} and I'm {age} years old")

# .format() method
print("My name is {} and I'm {} years old".format(name, age))

# % formatting (older style)
print("My name is %s and I'm %d years old" % (name, age))
```

### Working with Dates
```python
from datetime import datetime, timedelta

now = datetime.now()
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
parsed = datetime.strptime("2023-01-01", "%Y-%m-%d")
tomorrow = now + timedelta(days=1)
```

### Regular Expressions
```python
import re

pattern = r'\d+'  # One or more digits
text = "I have 5 apples and 3 oranges"
numbers = re.findall(pattern, text)  # ['5', '3']
```

## Useful Resources

- **Official Documentation:** [docs.python.org](https://docs.python.org)
- **Python Package Index:** [pypi.org](https://pypi.org)
- **Style Guide:** [PEP 8](https://pep8.org)
- **Interactive Learning:** [python.org/shell](https://python.org/shell)