import pandas as pd
import streamlit as st
# from moviesentiments_lr.model.predict import make_prediction

def init_session_state_vars():
    if 'prediction' not in st.session_state:
        st.session_state.prediction = ''
    if 'score' not in st.session_state:
        st.session_state.score = 0.0

def get_prediction():
    if st.session_state.input_text == '':
        st.error('You did not enter anything, try again.')
        return
    st.session_state.prediction = 'positive'
    st.session_state.score = 0.8
    # make_prediction(st.session_state.sentence)

def display_input_form():
    st.title('Make a Prediction')
    st.write('Use the following text box to make a prediction and see the label and confidence score from the trained model.')
    with st.form('input_form'):
        st.text_input(label='Input a sentence', placeholder='I loved the movie', help='Try inputting a sentence with a strong opinion for better results.', key='input_text')
        st.form_submit_button(label='Submit', on_click=get_prediction, type='primary')
        
def display_prediction_results():
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
    # Initalize the page and session variables
    st.set_page_config(page_title='Make Prediction', layout='wide')
    init_session_state_vars()
    
    # Display input field and results
    display_input_form()
    display_prediction_results()

run_page()