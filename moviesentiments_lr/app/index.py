import pandas as pd
import streamlit as st
from card import create_cards
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
    st.header('Make Predictions')
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
                'Confidence (Positive)': [f'{st.session_state.score:.2%}'],
                'Confidence (Negative)': [f'{1 - st.session_state.score:.2%}']
            }
            # Convert to DataFrame for easy display in Streamlit
            df = pd.DataFrame(data)
            
            # Display the table in Streamlit
            st.markdown(df.style.hide(axis='index').to_html(), unsafe_allow_html=True)
    
def display_index_page():
    # Initalize the page and session variables
    st.set_page_config(page_title='Home', layout='wide')
    init_session_state_vars()
    
    # Set page title and information
    st.title('Logistic Regression')
    st.write('Model results after training with a learning rate of **0.1** and L2 regularization parameter of **0.001** over **200** epochs.')
    st.sidebar.success('Select a page above.')
    
    # Define metrics
    test_values = [0.8812, 0.8724, 0.8952, 0.8836] # Accuracy, Precision, Recall, F1
    
    # Display the cards
    st.markdown(create_cards(test_values), unsafe_allow_html=True)
    
    # Display input field and results
    display_input_form()
    display_prediction_results()

display_index_page()