import streamlit as st
from card import create_cards

def run_app():
    # Initalize the page
    st.set_page_config(page_title='Home', layout='wide')
    st.sidebar.success('Select a page above.')

    # Set page title and information
    st.title('Background')    
    st.markdown("""
        
        This project is designed to analyze IMDB movie reviews using a Logisitic Regression model built from scratch.
        It was created with the aim of further understanding the basics of a simple neural network, including loss computation, backward propagation, and updating weights.
        
        The chosen learning rate was found by experimenting with rates such as 0.1, 0.01 and 0.001. L2 regularization was added to penalize the weights. Early stopping was added to stop the model if there was no improvement after a certain number of epochs.
        
        Feel free to explore the following tabs to interact with the model, get some insights into the training process, or view how the accuracy, precision, recall and f1 scores changes as number of epochs increases.
        
    """)
    
    # st.header('Final Results')
    st.write('*Final model results after training with a learning rate of **0.1** and L2 regularization parameter of **0.001** over **200** epochs.*')
    
    # Define metrics
    test_values = [0.8812, 0.8724, 0.8952, 0.8836] # Accuracy, Precision, Recall, F1
    
    # Display the cards
    st.markdown(create_cards(test_values), unsafe_allow_html=True)
    
    st.divider()

    
    st.link_button(label='Source code', url='https://github.com/anj0la/MovieSentiments-LR', help='Click to visit source code on GitHub.', type='primary')

run_app()