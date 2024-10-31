"""
File: predict.py

Author: Anjola Aina
Date Modified: October 30th, 2024

This module contains functions used to make predictions with the logistic regression model and is integrated with the Streamlit app.

Functions:
    make_prediction: Preprocesses input text, loads the trained model, and returns prediction with confidence scores.

External Dependencies:
    - joblib: Loads the pre-trained model and vectorizer.
    - numpy, pandas: For data processing.
    - logistic_regression (from model): Provides logistic regression model.
    - clean_review (from utils.preprocess): Cleans input text data.
"""
import joblib
import numpy as np
import pandas as pd
from model.logistic_regression import LogisticRegression
from utils.preprocess import clean_review

def make_prediction(sentence: str) -> tuple[float, str]:
    """
    Makes a prediction with the LogisticRegression model.

    Args:
        sentence (str): The sentence to make a prediction on.
        
    Returns:
        tuple(float, str): A tuple containing the raw logits of the model and the predicted class.
    """
    vectorizer_path = 'moviesentiments/data/model/vectorizer.pkl'
    le_path = 'moviesentiments/data/model/le.pkl'
    
    # Load the trained vectorizer and label encoder    
    vectorizer = joblib.load(vectorizer_path)
    le = joblib.load(le_path)
    
    # Convert sentence to Dataframe for easier processing
    df = pd.DataFrame({'review': [sentence]})
    
    # Transform the review into a suitable input
    cleaned_sentence = clean_review(df)
    vectorized_sentence = vectorizer.transform(cleaned_sentence).toarray()
    
    # Load the trained weights and bias for the model
    model = LogisticRegression()
    model.load_model()
    
    # Make a prediction
    logits = model.predict(vectorized_sentence)
    prediction = np.round(logits)
    prediction = prediction.astype(int).flatten()
    label = le.inverse_transform(prediction)
    
    return logits, label[0]