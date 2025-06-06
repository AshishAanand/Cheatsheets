# Python Data Structure Methods

A comprehensive reference for all methods available on Python's core data structures: Lists, Tuples, Sets, and Dictionaries.

## Table of Contents
- [List Methods](#list-methods)
- [Tuple Methods](#tuple-methods)
- [Set Methods](#set-methods)
- [Dictionary Methods](#dictionary-methods)
- [Common Operations](#common-operations)
- [Method Comparison Table](#method-comparison-table)

## List Methods

Lists are mutable, ordered collections that allow duplicate elements.

### Modifying Methods (Change the list)

```python
# append(item) - Add item to end
my_list = [1, 2, 3]
my_list.append(4)           # [1, 2, 3, 4]

# insert(index, item) - Insert item at specific index
my_list.insert(1, 'hello')  # [1, 'hello', 2, 3, 4]

# extend(iterable) - Add all items from iterable
my_list.extend([5, 6])      # [1, 'hello', 2, 3, 4, 5, 6]

# remove(item) - Remove first occurrence of item
my_list.remove('hello')     # [1, 2, 3, 4, 5, 6]

# pop(index) - Remove and return item at index (default: last)
last_item = my_list.pop()   # Returns 6, list becomes [1, 2, 3, 4, 5]
second_item = my_list.pop(1) # Returns 2, list becomes [1, 3, 4, 5]

# clear() - Remove all items
my_list.clear()             # []

# sort(key=None, reverse=False) - Sort list in place
numbers = [3, 1, 4, 1, 5]
numbers.sort()              # [1, 1, 3, 4, 5]
numbers.sort(reverse=True)  # [5, 4, 3, 1, 1]

# Sort with key function
words = ['banana', 'pie', 'Washington', 'book']
words.sort(key=len)         # ['pie', 'book', 'banana', 'Washington']

# reverse() - Reverse list in place
numbers.reverse()           # [1, 1, 3, 4, 5]
```

### Non-modifying Methods (Return information)

```python
# count(item) - Count occurrences of item
numbers = [1, 2, 3, 2, 2, 4]
count_2 = numbers.count(2)  # Returns 3

# index(item, start, end) - Find index of first occurrence
index_of_3 = numbers.index(3)     # Returns 2
index_of_2 = numbers.index(2, 2)  # Returns 3 (search from index 2)

# copy() - Create shallow copy
original = [1, 2, [3, 4]]
copied = original.copy()    # Creates new list with same elements
```

## Tuple Methods

Tuples are immutable, ordered collections that allow duplicate elements.

```python
# count(item) - Count occurrences of item
my_tuple = (1, 2, 3, 2, 2, 4)
count_2 = my_tuple.count(2)  # Returns 3

# index(item, start, end) - Find index of first occurrence
index_of_3 = my_tuple.index(3)     # Returns 2
index_of_2 = my_tuple.index(2, 2)  # Returns 3 (search from index 2)

# Note: Tuples have only 2 methods because they are immutable
```

## Set Methods

Sets are mutable, unordered collections of unique elements.

### Modifying Methods (Change the set)

```python
# add(item) - Add single item
my_set = {1, 2, 3}
my_set.add(4)               # {1, 2, 3, 4}

# update(iterable) - Add multiple items from iterable
my_set.update([5, 6, 7])    # {1, 2, 3, 4, 5, 6, 7}
my_set.update('abc')        # {1, 2, 3, 4, 5, 6, 7, 'a', 'b', 'c'}

# remove(item) - Remove item (raises KeyError if not found)
my_set.remove(1)            # {2, 3, 4, 5, 6, 7, 'a', 'b', 'c'}

# discard(item) - Remove item (no error if not found)
my_set.discard(999)         # No error, set unchanged
my_set.discard(2)           # {3, 4, 5, 6, 7, 'a', 'b', 'c'}

# pop() - Remove and return arbitrary item
item = my_set.pop()         # Returns and removes random item

# clear() - Remove all items
my_set.clear()              # set()
```

### Set Operations (Mathematical operations)

```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# union(*others) - Return union of sets
union_set = set1.union(set2)              # {1, 2, 3, 4, 5, 6}
# Alternative: set1 | set2

# intersection(*others) - Return intersection
intersection_set = set1.intersection(set2) # {3, 4}
# Alternative: set1 & set2

# difference(*others) - Return difference
difference_set = set1.difference(set2)     # {1, 2}
# Alternative: set1 - set2

# symmetric_difference(other) - Return symmetric difference
sym_diff = set1.symmetric_difference(set2) # {1, 2, 5, 6}
# Alternative: set1 ^ set2

# intersection_update(*others) - Update with intersection
set1.intersection_update(set2)  # set1 becomes {3, 4}

# difference_update(*others) - Update with difference
set1 = {1, 2, 3, 4}
set1.difference_update(set2)    # set1 becomes {1, 2}

# symmetric_difference_update(other) - Update with symmetric difference
set1 = {1, 2, 3, 4}
set1.symmetric_difference_update(set2)  # set1 becomes {1, 2, 5, 6}
```

### Comparison Methods (Return boolean)

```python
set1 = {1, 2}
set2 = {1, 2, 3, 4}
set3 = {5, 6}

# issubset(other) - Test if every element is in other
is_subset = set1.issubset(set2)     # True
# Alternative: set1 <= set2

# issuperset(other) - Test if every element of other is in set
is_superset = set2.issuperset(set1) # True
# Alternative: set2 >= set1

# isdisjoint(other) - Test if no elements in common
is_disjoint = set1.isdisjoint(set3) # True
```

### Non-modifying Methods

```python
# copy() - Create shallow copy
original_set = {1, 2, 3}
copied_set = original_set.copy()    # {1, 2, 3}
```

## Dictionary Methods

Dictionaries are mutable, unordered collections of key-value pairs.

### Accessing Methods

```python
my_dict = {'a': 1, 'b': 2, 'c': 3}

# get(key, default=None) - Get value for key
value = my_dict.get('a')        # Returns 1
value = my_dict.get('d', 0)     # Returns 0 (default)

# keys() - Get all keys
keys = my_dict.keys()           # dict_keys(['a', 'b', 'c'])

# values() - Get all values
values = my_dict.values()       # dict_values([1, 2, 3])

# items() - Get all key-value pairs
items = my_dict.items()         # dict_items([('a', 1), ('b', 2), ('c', 3)])
```

### Modifying Methods

```python
# update(other) - Update with key-value pairs from other
my_dict.update({'d': 4, 'e': 5})    # {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
my_dict.update([('f', 6), ('g', 7)]) # Can also use list of tuples

# setdefault(key, default=None) - Get key or set default if not exists
value = my_dict.setdefault('h', 8)   # Returns 8, adds 'h': 8 to dict
value = my_dict.setdefault('a', 99)  # Returns 1 (existing value)

# pop(key, default) - Remove key and return value
value = my_dict.pop('a')        # Returns 1, removes 'a' from dict
value = my_dict.pop('z', 0)     # Returns 0 (default, key not found)

# popitem() - Remove and return arbitrary (key, value) pair
item = my_dict.popitem()        # Returns ('g', 7) or similar

# clear() - Remove all items
my_dict.clear()                 # {}

# copy() - Create shallow copy
original = {'x': 1, 'y': 2}
copied = original.copy()        # {'x': 1, 'y': 2}
```

### Class Methods

```python
# fromkeys(iterable, value=None) - Create dict from keys
keys = ['a', 'b', 'c']
new_dict = dict.fromkeys(keys, 0)       # {'a': 0, 'b': 0, 'c': 0}
new_dict = dict.fromkeys(keys)          # {'a': None, 'b': None, 'c': None}
```

## Common Operations

### Length and Membership

```python
# Works for all data structures
my_list = [1, 2, 3]
my_tuple = (1, 2, 3)
my_set = {1, 2, 3}
my_dict = {'a': 1, 'b': 2}

# len() - Get number of elements
print(len(my_list))     # 3
print(len(my_tuple))    # 3
print(len(my_set))      # 3
print(len(my_dict))     # 2

# in operator - Check membership
print(2 in my_list)     # True
print(2 in my_tuple)    # True
print(2 in my_set)      # True
print('a' in my_dict)   # True (checks keys)
print(1 in my_dict.values())  # True (check values)
```

### Conversion Between Types

```python
# Convert between data structures
my_list = [1, 2, 3, 2]
my_tuple = tuple(my_list)       # (1, 2, 3, 2)
my_set = set(my_list)           # {1, 2, 3} - duplicates removed
back_to_list = list(my_set)     # [1, 2, 3]

# Dictionary from other structures
keys = ['a', 'b', 'c']
values = [1, 2, 3]
my_dict = dict(zip(keys, values))  # {'a': 1, 'b': 2, 'c': 3}
```

## Method Comparison Table

| Operation | List | Tuple | Set | Dictionary |
|-----------|------|-------|-----|------------|
| Add single item | `append(item)` | ❌ | `add(item)` | `dict[key] = value` |
| Add multiple items | `extend(iterable)` | ❌ | `update(iterable)` | `update(other)` |
| Remove item | `remove(item)` | ❌ | `remove(item)` | `pop(key)` |
| Remove item safely | ❌ | ❌ | `discard(item)` | `pop(key, default)` |
| Remove and return | `pop(index)` | ❌ | `pop()` | `pop(key)` |
| Count occurrences | `count(item)` | `count(item)` | ❌ | ❌ |
| Find index | `index(item)` | `index(item)` | ❌ | ❌ |
| Clear all | `clear()` | ❌ | `clear()` | `clear()` |
| Copy | `copy()` | ❌ | `copy()` | `copy()` |
| Sort | `sort()` | ❌ | ❌ | ❌ |
| Reverse | `reverse()` | ❌ | ❌ | ❌ |

## Quick Reference Summary

### Lists (Mutable, Ordered, Duplicates allowed)
**Modifying:** `append()`, `insert()`, `extend()`, `remove()`, `pop()`, `clear()`, `sort()`, `reverse()`
**Non-modifying:** `count()`, `index()`, `copy()`

### Tuples (Immutable, Ordered, Duplicates allowed)
**Methods:** `count()`, `index()`

### Sets (Mutable, Unordered, No duplicates)
**Modifying:** `add()`, `update()`, `remove()`, `discard()`, `pop()`, `clear()`
**Set operations:** `union()`, `intersection()`, `difference()`, `symmetric_difference()`
**Comparisons:** `issubset()`, `issuperset()`, `isdisjoint()`
**Non-modifying:** `copy()`

### Dictionaries (Mutable, Ordered as of Python 3.7+, Unique keys)
**Accessing:** `get()`, `keys()`, `values()`, `items()`
**Modifying:** `update()`, `setdefault()`, `pop()`, `popitem()`, `clear()`, `copy()`
**Class method:** `fromkeys()`

## Performance Notes

- **List append()**: O(1) average, O(n) worst case
- **List insert()**: O(n) - elements need to shift
- **Set add()/remove()**: O(1) average
- **Dictionary get()/pop()**: O(1) average
- **List/Tuple index()**: O(n) - linear search