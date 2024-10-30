import streamlit as st
from card import create_cards

def run_app():
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