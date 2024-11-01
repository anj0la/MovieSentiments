import nltk

def download_nltk_data():
    # Download data with no output if not already downloaded
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
