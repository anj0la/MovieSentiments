"""
File: 2_Training_Progress.py

Author: Anjola Aina
Date Modified: October 30th, 2024

Displays loss and accuracy graphs across various epochs and learning rates.

Functions:
    create_loss_columns: Displays loss graphs for selected epochs and learning rates.
    create_accuracy_columns: Displays accuracy graphs for selected epochs and learning rates.
    display_graphs: Organizes and displays loss and accuracy graphs on the page.
    run_page: Initializes the Training Progress page and displays training progress information.

External Dependencies:
    - streamlit: Used for displaying content in a web application.
"""
import streamlit as st

def create_loss_columns(epoch1: int, epoch2: int, lr1: float = 0.1, lr2: float = 0.01) -> None:
    """
    Displays loss graphs for two selected epochs and learning rates.

    Args:
        epoch1 (int): First epoch for which to display the loss graph.
        epoch2 (int): Second epoch for which to display the loss graph.
        lr1 (float): First learning rate to use for displaying graphs.
        lr2 (float): Second learning rate to use for displaying graphs.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(f'moviesentiments/figures/loss_epoch_{epoch1}_lr_{lr1}.png')
    with col2:
        st.image(f'moviesentiments/figures/loss_epoch_{epoch1}_lr_{lr2}.png')
    with col3:
        st.image(f'moviesentiments/figures/loss_epoch_{epoch2}_lr_{lr1}.png')
    with col4:
        st.image(f'moviesentiments/figures/loss_epoch_{epoch2}_lr_{lr2}.png')
        
    st.html(f"<p style='text-align: center; font-style: italic;'>Epochs {epoch1} & {epoch2} - Learning Rates: {lr1} & {lr2}</p>")
    
def create_accuracy_columns(epoch1: int, epoch2: int, lr1: float = 0.1, lr2: float = 0.01) -> None:
    """
    Displays accuracy graphs for two selected epochs and learning rates.

    Args:
        epoch1 (int): First epoch for which to display the accuracy graph.
        epoch2 (int): Second epoch for which to display the accuracy graph.
        lr1 (float): First learning rate to use for displaying graphs.
        lr2 (float): Second learning rate to use for displaying graphs.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(f'moviesentiments/figures/accuracy_epoch_{epoch1}_lr_{lr1}.png')
    with col2:
        st.image(f'moviesentiments/figures/accuracy_epoch_{epoch1}_lr_{lr2}.png') 
    with col3:
        st.image(f'moviesentiments/figures/accuracy_epoch_{epoch2}_lr_{lr1}.png')
    with col4:
        st.image(f'moviesentiments/figures/accuracy_epoch_{epoch2}_lr_{lr2}.png')
        
    st.html(f"<p style='text-align: center; font-style: italic;'>Epochs {epoch1} & {epoch2} - Learning Rates: {lr1} & {lr2}</p>")

def display_graphs() -> None:
    """
    Renders loss and accuracy graphs by epoch and learning rate.
    """
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
    """
    Initializes the Training Progress page with loss and accuracy graphs for model evaluation.
    """
    # Initalize the page and session variables
    st.set_page_config(page_title='Training Progress', layout='wide')
    
    # Set page title and information
    st.title('Training Progress')
    st.write('Listed below are the loss and accuracy graphs generated training, to see the progress of the model. The chosen learning rates were **0.1** and **0.01** as the model learned on these rates. Decreasing the learning rate resulted in a lack of training, hence the more aggresive learning rates.')
       
    display_graphs()

# Display the page
run_page()