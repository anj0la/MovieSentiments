import pandas as pd
import streamlit as st
# from moviesentiments_lr.model.predict import make_prediction

def run_page():
    # Initalize the page and session variables
    st.set_page_config(page_title='Training Progress', layout='wide')
    
    # Set page title and information
    st.title('Training Progress')
    st.write('Listed below are the loss and accuracy graphs generated training, to see the progress of the model. The chosen learning rates were **0.1** and **0.01** as the model learned on these rates. Decreasing the learning rate resulted in a lack of training, hence the more aggresive learning rates.')
   
    st.header('Losses over Epochs')
    
    st.header('Accuracy over Epochs')

run_page()