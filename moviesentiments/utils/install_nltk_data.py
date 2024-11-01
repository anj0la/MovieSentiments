"""
File: install_nltk_data.py

Author: Anjola Aina
Date Modified: October 31st, 2024

This module ensures that required NLTK data resources are downloaded for 
natural language processing tasks such as lemmatization and stopword removal.
The `download_nltk_data` function will silently download the data if it is not 
already available in the environment.

Dependencies:
    - nltk: Natural Language Toolkit, used for natural language processing tasks.
"""
import nltk

def download_nltk_data() -> None:
    """
    Downloads necessary NLTK data resources for text processing if they are 
    not already available. The function includes:
    
    - 'stopwords' for removing common stopwords in text processing.
    - 'wordnet' for lemmatization with the WordNet Lemmatizer.
    - 'omw-1.4' for additional support in morphological processing.

    Downloads occur silently without output to the console, ensuring 
    they only happen when necessary.
    """
    # Download data with no output if not already downloaded
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
