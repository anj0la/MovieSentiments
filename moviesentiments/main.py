"""
File: main.py
Author: Anjola Aina
Date Modified: October 30, 2024

Description:
    Command-line entry point for performing sentiment analysis on user-provided text 
    using a logistic regression model. This script allows users to input text directly 
    in the console and receive a sentiment prediction with probability scores.

Dependencies:
    - argparse: Handles command-line arguments for text input.
    - moviesentiments.model.predict.make_prediction: Imports the function to perform 
      the sentiment analysis on the input text.

Functions:
    main: Sets up the command-line interface, parses text input, and outputs the 
          sentiment prediction and probability scores.

Usage:
    Run this file with text input to obtain a sentiment prediction:
        $ moviesentiments --text This is a fantastic movie!
"""
import argparse
import nltk
from moviesentiments.model.predict import make_prediction
from moviesentiments.utils.install_nltk_data import download_nltk_data

def main():
    parser = argparse.ArgumentParser(description='Make a sentiment prediction from the command line.')

    # Add arguments for the prediction input
    parser.add_argument('--text', type=str, required=True, help='The text to analyze for sentiment.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if required nltk libraries have been downloaded to project
    try:
        nltk.data.find('stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        download_nltk_data()

    # Run prediction and output result
    if args.text != ' ':
        logits, result = make_prediction(args.text)  # Assuming `make_prediction` takes a single text input
        print(f'Predicted sentiment: {result} | Logits: {logits}')
        print(f'Positive probability: {logits:.2f} | Negative probability: {1 - logits:.2f}')
    else:
        print('Please provide text to analyze.')

if __name__ == "__main__":
    main()