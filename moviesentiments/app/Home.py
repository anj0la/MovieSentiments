"""
File: Home.py

Author: Anjola Aina
Date Modified: October 30th, 2024

This module sets up the homepage for the Streamlit application, providing an overview of the project.
It displays information about the logistic regression model used for sentiment analysis on IMDB movie reviews.

Functions:
    run_app: Initializes the page layout, displays project background information, and renders metrics as cards.

External Dependencies:
    - streamlit: Used to configure and display content in a web application.
    - create_cards (from card): Function to generate HTML-styled cards displaying model performance metrics.
"""
import streamlit as st
from card import create_cards
from moviesentiments.utils.install_nltk_data import download_stopwords

def run_app() -> None:
    """
    Sets up the Streamlit homepage with project background and performance metrics.

    The page provides an introduction to the project, highlighting model specifics such as
    learning rate selection, L2 regularization, and early stopping. It also displays final
    model results with accuracy, precision, recall, and F1 score.

    """
    # Install nltk stopwords
    download_stopwords()
    
    # Initalize the page
    st.set_page_config(page_title='Home', layout='wide')
    st.sidebar.success('Select a page above.')

    # Set page title and information
    st.title('Background')    
    st.markdown("""
        
    This project analyzes IMDB movie reviews using a custom-built Logistic Regression model. It was designed to deepen understanding of foundational neural network concepts by covering key aspects such as loss computation, backpropagation, and weight updates.

    To optimize performance, learning rates of 0.1, 0.01, and 0.001 were tested, and L2 regularization was applied to penalize large weights. Additionally, early stopping was implemented to prevent overfitting by halting training when progress plateaued.

    Explore the tabs to interact with the model and dive deeper into the project details!
    
    """)
    
    st.write('*Final model results after training with a learning rate of **0.1** and L2 regularization parameter of **0.001** over **200** epochs.*')
    
    # Define metrics
    test_values = [0.8812, 0.8724, 0.8952, 0.8836] # Accuracy, Precision, Recall, F1
    
    # Display the cards
    st.markdown(create_cards(test_values), unsafe_allow_html=True)
    
    st.divider()
    st.link_button(label='Source code', url='https://github.com/anj0la/MovieSentiments-LR', help='Click to visit source code on GitHub.', type='primary')
    
run_app()