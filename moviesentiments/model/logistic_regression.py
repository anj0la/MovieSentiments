"""
File: logistic_regression.py

Author: Anjola Aina
Date Modified: October 30th, 2024

Description:

This file contains the LogisticRegression class which is used to implement a binary classifier with a sigmoid activation function.
"""
import numpy as np
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from moviesentiments.utils.early_stopper import EarlyStopper
from utils.plot_graphs import plot_loss, plot_accuracy

class LogisticRegression:
    """
    A logistic regression model with sigmoid activation and optional L2 regularization.

    This class implements a simple logistic regression model using gradient descent optimization.
    It includes support for mini-batch training, L2 regularization, and early stopping.

    Attributes:
        lr (float): Learning rate for gradient descent.
        epochs (int): Number of training epochs.
        batch_size (int): Size of mini-batches used for training.
        reg_lambda (float): L2 regularization parameter to penalize large weights.
        weights (np.ndarray or None): Model weights, initialized during training.
        bias (float or None): Model bias term, initialized during training.
        early_stopper (EarlyStopper): Instance of EarlyStopper for early stopping based on validation loss.
    """
    def __init__(self, lr: float = 0.1, epochs: int = 10, batch_size: int = 64, reg_lambda: float = 0.0, patience: int = 3, min_delta: int = 10) -> None:
        """
        Initializes the logistic regression model with hyperparameters.

        Args:
            lr (float, optional): Learning rate for optimization. Defaults to 0.1.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Size of mini-batches for training. Defaults to 64.
            reg_lambda (float, optional): L2 regularization strength. Defaults to 0.0.
            patience (int, optional): Number of epochs without improvement for early stopping. Defaults to 3.
            min_delta (int, optional): Minimum improvement threshold for validation loss. Defaults to 10.
        """
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None # Bias
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    
    def _initialize_weights(self, n_features: int) -> None:
        """ 
        Initializes the weights in the logistic regression by assigning fixed values to the weights.
        
        Args:
            n_features(int): The number of features.
        """
        self.weights = np.zeros(n_features)
        self.bias = 0.5
        
    def _forward_pass(self, x: np.ndarray) -> np.ndarray:
        """
        Implements the forward pass for a logistic regression model.

        Args:
            x (np.ndarray): The input.

        Returns:
            np.ndarray: The predicted probability from the activation function.
        """
        z = self.bias + np.dot(x, self.weights)
        return self._sigmoid(z)
        
    def _update_weights(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> None:
        """
        Updates the weights and bias.

        Args:
            X (np.ndarray): The training set.
            y (np.ndarray): The corresponding labels for the training set.
            y_hat (np.ndarray): The predicted labels from the training set.
        """
        m = X.shape[0]
        d_weight = (1 / m) * np.dot(X.T, (y_hat - y))
        d_bias = (1 / m) * np.sum(y_hat - y)
        self.weights -= self.lr * d_weight
        self.bias -= self.lr * d_bias
            
    def _loss_function(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Implements the total cross entropy loss function.

        Args:
            y (np.ndarray): The true labels.
            y_hat (np.ndarray): The predicted labels.

        Returns:
            float: The total loss from the predicted labels.
        """
        # Clip y_hat to avoid log(0)
        y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
        m = y.shape[0]
        
        cross_entropy_loss = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        l2_reg = (self.reg_lambda / (2 * m)) * np.sum(np.square(self.weights))
        return cross_entropy_loss + l2_reg
       
    def _evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
        """
        Evaluates the model on the validation set.

        Args:
            X_val (np.ndarray): The validation set.
            y_val (np.ndarray): The corresponding labels for the validation set.

        Returns:
            tuple(float, float): A tuple containing the loss and accuracy score for the validation set.
        """
        # Forward pass on entire validation set
        y_hat = self._forward_pass(X_val)
        
        # Compute the loss
        loss = self._loss_function(y_val, y_hat)
        
        # Predicted labels
        predicted_labels = [1 if pred >= 0.5 else 0 for pred in y_hat]
        
        # Compute accuracy
        accuracy = accuracy_score(y_true=y_val, y_pred=predicted_labels)
        
        return loss, accuracy
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Implements the sigmoid function, an activiation function that changes the weighted sum z to a probability.

        Args:
            z (np.ndarray): The weighted sum.

        Returns:
            np.ndarray: The predicted probability of z.
        """
        return 1 / (1 + np.exp(-z))
        
    def _save_model(self) -> None:
        """
        Saves the trained weights and bias.
        """
        np.save('moviesentiments/data/model/weights.npy', self.weights)
        np.save('moviesentiments/data/model/bias.npy', np.array(self.bias))
        
    def load_model(self) -> None:
        """
        Loads the trained weights and bias to the model.
        """
        self.weights = np.load('moviesentiments/data/model/weights.npy')
        self.bias = np.load('moviesentiments/data/model/bias.npy')

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the model.

        Args:
            X_train (np.ndarray): The training set.
            y_train (np.ndarray): The corresponding labels for the testing set.
        """
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Convert to dense arrays to work with model
        X_train, X_val = X_train.toarray(), X_val.toarray()
        
        # Initalize weights
        self._initialize_weights(X_train.shape[1])
        
        all_train_losses = []
        all_val_losses = []
        all_train_accuracy = []
        all_val_accuracy = []
        num_batches = X_train.shape[0] // self.batch_size
        
        # Training loop
        for epoch in tqdm(range(self.epochs), desc='Training...'):
            
            # Track epoch time
            epoch_start_time = time.time()
            
            print(f'\nStarting epoch {epoch + 1}...')
            
            # Generate batch indices and shuffle them
            batch_indices = np.arange(num_batches)
            np.random.shuffle(batch_indices)
            
            # Reset loss and labels after every epoch
            epoch_loss = 0
            total_correct = 0
            total_samples = 0
            
            for i in batch_indices:
                # Get the mini-batch
                start_i = i * self.batch_size
                end_i = start_i + self.batch_size
                X_batch = X_train[start_i:end_i]
                y_batch = y_train[start_i:end_i]
                
                # Forward pass
                y_hat = self._forward_pass(X_batch)
                
                # Compute predictions and accuracy for the current batch
                batch_predicted_labels = [1 if pred >= 0.5 else 0 for pred in y_hat]
                total_correct += sum(1 for true, pred in zip(y_batch, batch_predicted_labels) if true == pred)
                total_samples += len(y_batch)
                
                # Compute the loss for current batch
                batch_loss = self._loss_function(y_batch, y_hat)
                epoch_loss += batch_loss
                
                # Update the weights for current batch
                self._update_weights(X_batch, y_batch, y_hat)
                
            # Compute average loss and accuracy        
            avg_loss = epoch_loss / num_batches
            train_accuracy = total_correct / total_samples

            # Get validaation loss and accuracy
            val_loss, val_accuracy = self._evaluate(X_val, y_val)
            
            # Early stopping check
            if self.early_stopper.early_stop(val_loss):             
                break
            
            # Save the weights and bias to be used for predictions
            self._save_model()

            # Append train and val losses and accurary
            all_train_losses.append(avg_loss)
            all_train_accuracy.append(round(train_accuracy, 2))
            all_val_losses.append(val_loss)
            all_val_accuracy.append(round(val_accuracy, 2))
            
            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time
            
            # Print train and val metrics
            print(f'\t Epoch: {epoch + 1} out of {self.epochs}')
            print(f'\t Time Taken: {epoch_duration:.2f} seconds')
            print(f'\t Train Loss: {avg_loss:.3f} | Train Acc: {train_accuracy * 100:.2f}%')
            print(f'\t Valid Loss: {val_loss:.3f} | Valid Acc: {val_accuracy * 100:.2f}%')
            
        # Visualize (and save) plots
        x_axis = list(range(1, self.epochs + 1))
        plot_loss(x_axis=x_axis, train_losses=all_train_losses, val_losses=all_val_losses, figure_path=f'moviesentiments/figures/loss_epoch_{len(x_axis)}_lr_{self.lr}.png')
        plot_accuracy(x_axis=x_axis, train_accuracy=all_train_accuracy,val_accuracy= all_val_accuracy, figure_path=f'moviesentiments/figures/accuracy_epoch_{len(x_axis)}_lr_{self.lr}.png')

    def predict(self, X: np.ndarray) -> float:
        """
        Predicts the probability for a given input X with the use of the sigmoid activation function.
        
        Args:
            X (ndarray): The input to make a prediction on.

        Returns:
            float: The raw logits of the model.
        """
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float, float]:
        """
        Evaluates the trained model on the testing set.

        Args:
            X_test (np.ndarray): The testing set.
            y_test (np.ndarray): The corresponding labels for the testing set.

        Returns:
            tuple(float, float, float, float): A tuple containing the following testing metrics: accuracy, precision, recall and f1.
        """
        y_pred = np.round(self.predict(X_test))
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        return accuracy, precision, recall, f1
