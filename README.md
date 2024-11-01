# MovieSentiments

<div align="center">
    <img src="moviesentiments_logo_image.png" alt="Screenshot of the logo" width="400">
</div>

## A Sentiment Analysis Tool

MovieSentiments is a project designed to analyze IMDB movie reviews using a custom-built Logistic Regression model. Created to deepen understanding of neural network fundamentals, the project focuses on key steps such as loss computation, backpropagation, and weight updates, which are typically handled automatically by libraries like PyTorch, Keras, and TensorFlow. The model has achieved an accuracy score of 88.12% and generalizes well.

The model can be used via two methods: using Streamlit or the console. Streamlit is a simple app framework for web applications, and the application provides four main interfaces: the homepage, the prediction page for user input and predictions, and the training progress and epoch metrics pages for insight into model performance. More details on how to use the two methods can be found below.

The dataset used to train the model can be found here. https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews?resource=download&select=reviews.csv

## Using the Streamlit App

The Streamlit app can be found via the following link: https://moviesentiments.streamlit.app/

To make a prediction, simply navigate to the **Make Prediction** page and input a sentence to see the prediction, and positive / negative class probabilities.
The following demonstration shows how one can navigate to the page and make a prediction for the sentence "I loved the movie". 

[Insert video here]

### Viewing Training Progress and Testing Metrics

Click on the **Training Progress** page to view some of the graphs that were generated during training with the selected learning rates. The learning rates of 0.1 and 0.01 were chosen, as a rate lower than 0.01 failed to train effectively after initial testing. 

To view the testing metrics for the model at the specified epochs with the above learning rates, click on the **Epoch Metrics** page.

## Using the Console

Using the console is relatively simple. Once the project has been built, using the --text field, input your sentence to get a prediction.

[Insert video demo here]

## Building the Project

Before using the console application, the project must first be set up in a virtual environment. Follow the steps below as a guide. This example uses .venv as the virtual environment, but feel free to substitute with your preferred environment name.

### 1. Create a virtual environment in the directory of the project.
```
python3 -m venv .venv
```

### 2. Activate the virtual environment on your operating system.

On macOS/ Linux:

```
source .venv/bin/activate
```

On Windows:

```
.venv\Scripts\activate
```

### 3. Install the dependencies into your virtual environment. 
*Note: The project is built in editable mode so there's no need to build the project after installing all dependencies from requirements.txt.*
```
pip install -r requirements.txt
```
### 4. Run the project and make a prediction.
```
moviesentiments --text "I loved the movie!"
```

## Retraining the Model

To retrain the model, run the train function located in the model folder. You can call this function from main.py before launching the app to generate updated results. The weights and bias are saved in the same folder, so as long as the folder structure remains consistent, you can adjust parameters like the learning rate, regularization strength, and number of epochs, while still maintaining a functional model.