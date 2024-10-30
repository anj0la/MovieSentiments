"""
File: predict.py

Author: Anjola Aina
Date Modified: October 29th, 2024

Description:

This file is used to make a prediction using the logistic regression model.
TODO: Update documentation.
"""
import joblib
import numpy as np
import pandas as pd
from model.logistic_regression import LogisticRegression
from utils.preprocess import clean_review

def make_prediction(sentence: str) -> None:
    """
    Makes a prediction with the specificed model.

    Args:
        model_name (str): The model used to make a prediction. Defaults to LR.
    """
    vectorizer_path = 'moviesentiments_lr/data/model/vectorizer.pkl'
    le_path = 'moviesentiments_lr/data/model/le.pkl'
    
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
    
    return logits, label