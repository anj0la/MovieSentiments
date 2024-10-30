import streamlit as st
from card import create_cards



def configure_page():
    st.set_page_config(page_title='Home', layout='wide')
    st.title('Dashboard')
    st.sidebar.success('Select a page above.')
    
    # Define metrics
    test_values = [0.8812, 0.8724, 0.8952, 0.8836] # Accuracy, Precision, Recall, F1
    
    # Display the cards
    st.markdown(create_cards(test_values), unsafe_allow_html=True)
    
    
    st.markdown("""
    The current machine learning model used in this application is NAME GOES HERE, an explainable AI that is able to 
    predict whether a input sentence is positive, negative or not applicable, calculate hierarchical results, 
    extract the top phrase causing the prediction, and calculate the scores of each word in the input sentence.
    Along with that, the application support (or will support) multiple machine learning models that can be used to 
    do the same things as the current explainable AI, discussed above.
    #### Current Research
    To reduce the gap of the end users’ understanding about a black box AI model’s behaviour, algorithms must be 
    developed for an explainable AI system that semantically understands human language. 
    This is why an interactive explanation system has been created, so the end users may interact with the model
    and ask follow up questions, in which the explainable AI will make a decision based on the answers to the follow-up
    questions.
    #### Goals
    - Handling different models
    - Make it easier for end users to understand how the AI arrived at its prediction, or more importantly,
      why the AI made a certain prediction
    - Obtain feedback to know where limitations in the explainable AI lie
    - Improve the current AI to better reduce the gap between end users using the AI model and the behaviour of the AI
    """)
    
    
configure_page()