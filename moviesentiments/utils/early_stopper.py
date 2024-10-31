"""
File: early_stopper.py

Author: Anjola Aina
Date Modified: October 30th, 2024

Description:

This file contains the EarlyStopper class which is used for early stopping based on validation loss improvement.
The original source for the code can be found via the following link: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
"""
class EarlyStopper:
    """
    A utility class for early stopping based on validation loss improvement.

    Early stopping halts training if validation loss does not improve over a defined patience period,
    preventing overfitting and saving computational resources.

    Attributes:
        patience (int): Number of consecutive epochs without improvement before stopping.
        min_delta (float): Minimum change in validation loss to qualify as an improvement.
        counter (int): Tracks epochs without improvement.
        min_validation_loss (float): Best observed validation loss during training.
    """
    def __init__(self, patience: int = 1, min_delta: int = 0) -> None:
        """
        Initializes the early stopping mechanism.

        Args:
            patience (int, optional): Number of epochs to wait for improvement. Defaults to 1.
            min_delta (int, optional): Minimum threshold for validation loss improvement. Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        """
        Determines whether to stop training based on validation loss.

        Args:
            validation_loss (float): The current validation loss.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False