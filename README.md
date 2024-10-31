# MovieSentiments

MovieSentiments is a project designed to analyze IMDB movie reviews using a custom-built Logistic Regression model. Created to deepen understanding of neural network fundamentals, the project focuses on key steps such as loss computation, backpropagation, and weight updates, which are typically handled automatically by libraries like PyTorch, Keras, and TensorFlow. The model has achieved an accuracy score of 88.12% and generalizes well.

The model is deployed via Streamlit, a simple app framework for web applications, with four main interfaces: the homepage, the prediction page for user input and predictions, and the training progress and epoch metrics pages for insight into model performance. More details on each page are provided below.

The dataset used to train the model can be found here. https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews?resource=download&select=reviews.csv

## Home Page

The home page provides an overview of the project, along with the model's final testing metrics. A GitHub link is also available as a button for easy access to the project repository.

## Make Prediction Page

On the prediction page, users can input text for analysis and receive predictions labeled as positive or negative, along with probability scores for each class.

## Training Progress Page

This page offers insight into the training process, including learning rate selection. After initial testing, learning rates of 0.1 and 0.01 were chosen, as a rate lower than 0.01 failed to train effectively. L2 regularization, initially set at 1e-2, was reduced to 1e-3 in the final model to prevent underfitting.

## Epoch Metrics Page

The epoch metrics page displays the testing metrics done on the testing set for the model at the specified epochs with learning rates 0.1 and 0.01 respectively. It is clear that the model performs better when the learning rate is 0.1 as opposed to 0.01, which contributed to the decision to stick with a learning rate of 0.1 for the final model.

## Setup

Setting up the project is relatively simple. If choosing to use the command-line interface option, follow the steps below.

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
```
pip install -r requirements.txt
```

### 5. Install and build the project.

Example: installing the project by its name. 
```
pip install moviesentiments
```

Example: installing the project in editable mode. 
```
pip install -e .
```
### 6. Run the project and make a prediction.
```
moviesentiments --text "I loved the movie!"
```

## Retraining the Model

To retrain the model, run the train function located in the model folder. You can call this function from main.py before launching the app to generate updated results. The weights and bias are saved in the same folder, so as long as the folder structure remains consistent, you can adjust parameters like the learning rate, regularization strength, and number of epochs, while still maintaining a functional model.