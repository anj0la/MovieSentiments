"""
File: 3_Epochs_Metrics.py

Author: Anjola Aina
Date Modified: October 30th, 2024

Displays model testing metrics (accuracy, precision, recall, and F1 score) at selected epochs 
and learning rates.

Functions:
    display_metrics_table: Formats and displays testing metrics as a table.
    run_page: Initializes the Epoch Metrics page and renders the table.

External Dependencies:
    - pandas: For data manipulation and formatting.
    - streamlit: Used for displaying content in a web application.
"""
import pandas as pd
import streamlit as st

def display_metrics_table() -> None:
    """
    Creates and displays a sorted table of testing metrics for various epochs and learning rates.

    Metrics include accuracy, precision, recall, and F1 score, and are formatted to two decimal places.
    """
    data = {
    'Learning Rate': ['0.1'] * 4 + ['0.01'] * 4,
    'Epochs': [10, 25, 50, 100, 10, 25, 50, 100],
    'Test Acc (%)': [82.12, 83.91, 85.50, 87.16, 79.84, 80.63, 81.13, 82.03],
    'Precision (%)': [80.65, 81.24, 83.13, 85.82, 81.61, 79.99, 79.39, 80.37],
    'Recall (%)': [84.88, 88.51, 89.36, 89.26, 77.44, 82.10, 84.48, 85.14],
    'F1 Score (%)': [82.71, 84.72, 86.13, 87.51, 79.47, 81.03, 81.86, 82.68]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by=['Epochs', 'Learning Rate'], ascending=[True, False]).reset_index(drop=True)
    
    # Format numerical columns to two decimal places as strings
    df_sorted = df_sorted.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x)
    
    # Display the table in Streamlit
    st.markdown(df_sorted.style.hide(axis='index').to_html(), unsafe_allow_html=True)
    
def run_page() -> None:
    """
    Initializes the Epoch Metrics page and displays testing metrics for model evaluation.
    """
    # Initalize the page and session variables
    st.set_page_config(page_title='Epoch Metrics', layout='wide')
    
    # Set page title and information
    st.title('Epoch Metrics')
    st.write('Listed below is the table representing testing metrics over 10, 25, 50 and 100 epochs with the chosen learning rates **0.1** and **0.01**.')
    
    display_metrics_table()
    
# Display the page
run_page()