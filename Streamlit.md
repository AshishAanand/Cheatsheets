# Streamlit Cheat Sheet

A comprehensive reference guide for building interactive web applications with Streamlit.

## Table of Contents
- [Installation & Setup](#installation--setup)
- [Basic App Structure](#basic-app-structure)
- [Text & Display Elements](#text--display-elements)
- [Input Widgets](#input-widgets)
- [Data Display](#data-display)
- [Charts & Visualizations](#charts--visualizations)
- [Layout & Containers](#layout--containers)
- [State Management](#state-management)
- [File Operations](#file-operations)
- [Deployment](#deployment)

## Installation & Setup

```bash
# Install Streamlit
pip install streamlit

# Run your app
streamlit run app.py

# Run on specific port
streamlit run app.py --server.port 8080
```

## Basic App Structure

```python
import streamlit as st

# Basic app template
st.title("My Streamlit App")
st.write("Hello, World!")

# Add favicon and page config (must be first Streamlit command)
st.set_page_config(
    page_title="My App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

## Text & Display Elements

```python
# Headers and text
st.title("Main Title")
st.header("Header")
st.subheader("Subheader")
st.text("Fixed width text")
st.markdown("**Bold** and *italic* text")
st.write("Universal display function")

# Code display
st.code("print('Hello, World!')", language='python')

# Alerts and messages
st.success("Success message")
st.info("Info message")
st.warning("Warning message")
st.error("Error message")
st.exception(RuntimeError("Error details"))
```

## Input Widgets

```python
# Text inputs
name = st.text_input("Enter your name")
text_area = st.text_area("Enter description", height=100)

# Numbers
age = st.number_input("Enter age", min_value=0, max_value=120, value=25)
slider_val = st.slider("Select value", 0, 100, 50)

# Selections
option = st.selectbox("Choose option", ["Option 1", "Option 2", "Option 3"])
multi_select = st.multiselect("Choose multiple", ["A", "B", "C", "D"])
radio = st.radio("Pick one", ["Yes", "No", "Maybe"])

# Boolean inputs
checkbox = st.checkbox("Check me")
toggle = st.toggle("Toggle me")

# Date and time
date = st.date_input("Select date")
time = st.time_input("Select time")

# File upload
uploaded_file = st.file_uploader("Upload file", type=['csv', 'txt', 'json'])

# Buttons
if st.button("Click me"):
    st.write("Button was clicked!")
```

## Data Display

```python
import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40]
})

# Display dataframe
st.dataframe(df)  # Interactive table
st.table(df)      # Static table

# Display metrics
st.metric("Temperature", "70°F", "1.2°F")

# Display JSON
st.json({"key": "value", "number": 123})
```

## Charts & Visualizations

```python
import matplotlib.pyplot as plt
import plotly.express as px

# Line chart
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)
st.line_chart(chart_data)

# Bar chart
st.bar_chart(chart_data)

# Area chart
st.area_chart(chart_data)

# Scatter chart
st.scatter_chart(chart_data)

# Matplotlib
fig, ax = plt.subplots()
ax.hist(np.random.randn(100), bins=20)
st.pyplot(fig)

# Plotly
fig = px.scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13])
st.plotly_chart(fig)

# Map
map_data = pd.DataFrame({
    'lat': [37.76, 37.77, 37.78],
    'lon': [-122.4, -122.41, -122.42]
})
st.map(map_data)
```

## Layout & Containers

```python
# Columns
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Column 1")
with col2:
    st.write("Column 2")
with col3:
    st.write("Column 3")

# Sidebar
st.sidebar.title("Sidebar")
sidebar_input = st.sidebar.slider("Sidebar slider", 0, 100)

# Containers
container = st.container()
with container:
    st.write("Inside container")

# Expander
with st.expander("See details"):
    st.write("Hidden content here")

# Tabs
tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])
with tab1:
    st.write("Content of tab 1")
with tab2:
    st.write("Content of tab 2")
```

## State Management

```python
# Session state
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# Button with state
if st.button("Increment"):
    st.session_state.counter += 1

st.write(f"Counter: {st.session_state.counter}")

# Callbacks
def increment_counter():
    st.session_state.counter += 1

st.button("Increment with callback", on_click=increment_counter)
```

## File Operations

```python
# File upload and processing
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

# Download button
csv = df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="data.csv",
    mime="text/csv"
)
```

## Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy!

### Local Development
```bash
# Create requirements.txt
pip freeze > requirements.txt

# Create .streamlit/config.toml for configuration
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## Essential Tips

### Performance Optimization
```python
# Cache data loading
@st.cache_data
def load_data():
    return pd.read_csv("large_file.csv")

# Cache resource initialization
@st.cache_resource
def init_model():
    return load_model("model.pkl")
```

### Form Handling
```python
# Forms prevent rerun on every widget interaction
with st.form("my_form"):
    name = st.text_input("Name")
    age = st.number_input("Age")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.write(f"Hello {name}, you are {age} years old")
```

### Progress Indicators
```python
# Progress bar
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)

# Spinner
with st.spinner("Loading..."):
    time.sleep(3)
st.success("Done!")
```

## Common Patterns

### Multi-page App Structure
```python
# pages/home.py
import streamlit as st

def show():
    st.title("Home Page")
    st.write("Welcome to the home page!")

# main.py
import streamlit as st
from pages import home, about

st.set_page_config(page_title="Multi-page App")

pages = {
    "Home": home,
    "About": about
}

selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
pages[selected_page].show()
```

### Dynamic Filtering
```python
# Filter dataframe based on user input
df = load_data()
categories = st.multiselect("Filter by category", df['category'].unique())
if categories:
    filtered_df = df[df['category'].isin(categories)]
else:
    filtered_df = df

st.dataframe(filtered_df)
```

## Useful Resources

- **Official Documentation:** [docs.streamlit.io](https://docs.streamlit.io)
- **Gallery:** [streamlit.io/gallery](https://streamlit.io/gallery)
- **Community:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub:** [github.com/streamlit/streamlit](https://github.com/streamlit/streamlit)