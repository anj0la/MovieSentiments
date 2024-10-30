import pandas as pd
import streamlit as st
# from moviesentiments_lr.model.predict import make_prediction

def run_page():
    # Initalize the page and session variables
    st.set_page_config(page_title='Epoch Metrics', layout='wide')
    
    # Set page title and information
    st.title('Epoch Metrics')
    st.write('Listed below is the table representing testing metrics over 10, 25, 50 and 100 epochs with the chosen learning rates **0.1** and **0.01**.')

    
run_page()