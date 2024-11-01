"""
File: train.py

Author: Anjola Aina
Date Modified: October 30th, 2024

This module defines the training function for the logistic regression model. It includes data preparation, training, and evaluation.

Functions:
    train: Preprocesses data, trains the logistic regression model, and outputs evaluation metrics.

External Dependencies:
    - joblib: Saves model components (vectorizer, label encoder).
    - pandas, sklearn: For data manipulation and model preparation.
    - logistic_regression (from model): Provides logistic regression model.
"""
import joblib
import pandas as pd
import time
from logistic_regression import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train(file_path: str, lr: float = 1e-1, epochs: int = 200, batch_size: int = 64, reg_lambda: float = 1e-3, patience: int = 3, min_delta: int = 10) -> None:
    """
    Trains the logisitic regression model.

    Args:
        lr (float, optional): Learning rate for optimization. Defaults to 1e-1 (0.1).
        epochs (int, optional): Number of training epochs. Defaults to 200.
        batch_size (int, optional): Size of mini-batches for training. Defaults to 64.
        reg_lambda (float, optional): L2 regularization strength. Defaults to 1e-3 (0.001).
        patience (int, optional): Number of epochs without improvement for early stopping. Defaults to 3.
        min_delta (int, optional): Minimum improvement threshold for validation loss. Defaults to 10.

    """
    vectorizer_path = 'moviesentiments/data/model/vectorizer.pkl'
    le_path = 'moviesentiments/data/model/le.pkl'
    df = pd.read_csv(file_path)
    vectorizer = TfidfVectorizer()
    le = LabelEncoder()
    
    # Fit-transform the reviews and sentiments (learns the vocabulary)
    X = vectorizer.fit_transform(df['review'])
    y = le.fit_transform(df['sentiment'].values)
    
    # Save vectorizer and label encoder
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(le, le_path)
        
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Trains (and validates) the model
    classifier = LogisticRegression(lr=lr, epochs=epochs, batch_size=batch_size, reg_lambda=reg_lambda, patience=patience, min_delta=min_delta)
    
    # Track overall training time
    overall_start_time = time.time()

    classifier.fit(X_train, y_train) 
    
    # Calculate total training time
    total_duration = time.time() - overall_start_time
    print(f'Training completed in {total_duration:.2f} seconds')
    
    # Convert to dense array
    X_test = X_test.toarray()
    
    # Evaluate model on test set
    accuracy, precision, recall, f1_score = classifier.evaluate(X_test, y_test)
    
    # Print metrics
    print(f'Test Acc: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')    
    print(f'Recall: {recall * 100:.2f}%')    
    print(f'F1 Score: {f1_score * 100:.2f}%')    
    
# train(file_path='moviesentiments/data/reviews/cleaned_movie_reviews.csv')