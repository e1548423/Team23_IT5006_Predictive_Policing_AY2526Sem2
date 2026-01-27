import streamlit as st
import pandas as pd

# Set the title of the app
st.title("My First Streamlit App 🚀")

# Add some text
st.write("Welcome to my first dashboard created in VS Code!")

# Create a simple dataframe
df = pd.DataFrame({
    'Column A': [1, 2, 3, 4],
    'Column B': [10, 20, 30, 40]
})

# Display the table
st.subheader("Simple Data Table")
st.write(df)

# Add an interactive widget
user_input = st.text_input("What is your name?")
if user_input:
    st.write(f"Hello, {user_input}!")