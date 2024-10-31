"""
File: 1_Make_Prediction.py

Author: Anjola Aina
Date Modified: October 30th, 2024

Allows users to input text and receive a sentiment prediction, including probabilities for each class.

Functions:
    init_session_state_vars: Initializes session variables for prediction and score tracking.
    get_prediction: Simulates a prediction based on user input and sets session variables.
    display_input_form: Displays a form for users to input text and submit for prediction.
    display_prediction_results: Displays prediction results based on user input.
    run_page: Initializes the Prediction page and sets up prediction input/output.

External Dependencies:
    - pandas: For organizing and displaying prediction results.
    - streamlit: Used for displaying content in a web application.
"""
import pandas as pd
import streamlit as st
# from moviesentiments.model.predict import make_prediction

def init_session_state_vars() -> None:
    """
    Initializes session state variables for prediction and confidence score tracking.
    """
    if 'prediction' not in st.session_state:
        st.session_state.prediction = ''
    if 'score' not in st.session_state:
        st.session_state.score = 0.0

def get_prediction() -> None:
    """
    Sets the prediction and confidence score based on user input.
    """
    if st.session_state.input_text == '':
        st.error('You did not enter anything, try again.')
        return
    st.session_state.prediction = 'positive'
    st.session_state.score = 0.8
    # make_prediction(st.session_state.sentence)

def display_input_form() -> None:
    """
    Displays an input form for users to submit text and get a sentiment prediction.
    """
    st.title('Make a Prediction')
    st.write('Use the following text box to make a prediction and see the label and confidence score from the trained model.')
    with st.form('input_form'):
        st.text_input(label='Input a sentence', placeholder='I loved the movie', help='Try inputting a sentence with a strong opinion for better results.', key='input_text')
        st.form_submit_button(label='Submit', on_click=get_prediction, type='primary')
        
def display_prediction_results():
    """
    Displays the prediction results including predicted label and confidence scores.
    """
    if st.session_state.prediction != '':
        with st.container(border=True):
            # Prepare the data for display
            data = {
                'Input Text': [st.session_state.input_text],
                'Prediction': [st.session_state.prediction],
                'Probability (Positive)': [f'{st.session_state.score:.2%}'],
                'Probability (Negative)': [f'{1 - st.session_state.score:.2%}']
            }
            # Convert to DataFrame for easy display in Streamlit
            df = pd.DataFrame(data)
            
            # Display the table in Streamlit
            st.markdown(df.style.hide(axis='index').to_html(), unsafe_allow_html=True)
    
def run_page():
    """
    Initializes the Prediction page and displays the prediction input and results.
    """
    # Initalize the page and session variables
    st.set_page_config(page_title='Make Prediction', layout='wide')
    init_session_state_vars()
    
    # Display input field and results
    display_input_form()
    display_prediction_results()

run_page()