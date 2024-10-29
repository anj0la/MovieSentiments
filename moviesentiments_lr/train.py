"""
File: train.py

Author: Anjola Aina
Date Modified: October 29th, 2024

Description:

This file contains the train function which is used to train the custom LogisiticRegression model class.
"""
import joblib
import pandas as pd
import time
from model.logistic_regression import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train(file_path: str, lr: float = 1e-1, epochs: int = 100, batch_size: int = 64, reg_lambda: float = 1e-2, patience: int = 3, min_delta: int = 10) -> None:
    """
    Trains the logisitic regression model.

    Args:
        file_path (str): The path to the cleaned file.
        lr (float, optional): The learning rate. Defaults to 0.01.
        epochs (int, optional): The number of epochs. Defaults to 100.
        batch_size (int, optional): The batch size. Defaults to 64.
        decay_factor (float, optional): The decay factor (how fast the learning rate decreases). Defaults to 1.0.
        lr_step (int, optional): The interval at which the learning rate decreases by the decay factor. Defaults to 10.
        reg_lambda (float, optional): The hyperparameter for L2 regularization. Defaults to 0.01.
        no_progress_epochs (int, optional): The early stopping parameter. Defaults to 10.
    """
    vectorizer_path = 'moviesentiments_lr/data/model/vectorizer.pkl'
    le_path = 'moviesentiments_lr/data/model/le.pkl'
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
    
train(file_path='moviesentiments_lr/data/reviews/cleaned_movie_reviews.csv')