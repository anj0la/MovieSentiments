import pandas as pd
import streamlit as st

def create_loss_columns(epoch1: int, epoch2: int, lr1: float = 0.1, lr2: float = 0.01):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(f'moviesentiments_lr/figures/loss_epoch_{epoch1}_lr_{lr1}.png')

    with col2:
        st.image(f'moviesentiments_lr/figures/loss_epoch_{epoch1}_lr_{lr2}.png')
        
    with col3:
        st.image(f'moviesentiments_lr/figures/loss_epoch_{epoch2}_lr_{lr1}.png')

    with col4:
        st.image(f'moviesentiments_lr/figures/loss_epoch_{epoch2}_lr_{lr2}.png')
        
    st.html(f"<p style='text-align: center; font-style: italic;'>Epochs {epoch1} & {epoch2} - Learning Rates: {lr1} & {lr2}</p>")
    
def create_accuracy_columns(epoch1: int, epoch2: int, lr1: float = 0.1, lr2: float = 0.01):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(f'moviesentiments_lr/figures/accuracy_epoch_{epoch1}_lr_{lr1}.png')

    with col2:
        st.image(f'moviesentiments_lr/figures/accuracy_epoch_{epoch1}_lr_{lr2}.png')
        
    with col3:
        st.image(f'moviesentiments_lr/figures/accuracy_epoch_{epoch2}_lr_{lr1}.png')

    with col4:
        st.image(f'moviesentiments_lr/figures/accuracy_epoch_{epoch2}_lr_{lr2}.png')
        
    st.html(f"<p style='text-align: center; font-style: italic;'>Epochs {epoch1} & {epoch2} - Learning Rates: {lr1} & {lr2}</p>")

def display_graphs():
    st.header('Loss Graphs by Epoch and Learning Rate')
    create_loss_columns(10, 25)

    # Space between rows for visual clarity
    st.write(' ') 
    create_loss_columns(50, 100)

    st.header('Accuracy Graphs by Epoch')
    create_accuracy_columns(10, 25)

    # Space between rows for visual clarity
    st.write(' ') 
    create_accuracy_columns(50, 100)

    
def run_page():
    # Initalize the page and session variables
    st.set_page_config(page_title='Training Progress', layout='wide')
    
    # Set page title and information
    st.title('Training Progress')
    st.write('Listed below are the loss and accuracy graphs generated training, to see the progress of the model. The chosen learning rates were **0.1** and **0.01** as the model learned on these rates. Decreasing the learning rate resulted in a lack of training, hence the more aggresive learning rates.')
       
    display_graphs()

run_page()