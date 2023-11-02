# Training an evaluation code
import numpy as np 
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import time
import wandb
from torch.utils.data import Dataset,DataLoader
import torchvision.models as models
import json
import torchvision

def get_predictions(model, dataloader, device):
    """
    Obtain predictions from a model for the data provided by a dataloader.

    Args:
        model (nn.Module): The trained model to be used for predictions.
        dataloader (DataLoader): DataLoader containing the dataset for which predictions are needed.
        device (torch.device): The device (CPU or CUDA) where the model and data are located.

    Returns:
        list: List containing the predicted classes for the input data.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []  # Initialize an empty list to store predictions

    with torch.no_grad():  # Disable gradient computation
        for data in dataloader:  # Iterate over the data in the dataloader
            inputs, _ = data  # Extract inputs; labels are not needed for prediction
            inputs = inputs.to(device)  # Move inputs to the specified device
            outputs = model(inputs)  # Forward pass to get outputs from the model
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            predictions += predicted.cpu().tolist()  # Store predictions

    return predictions  # Return the list of predictions


def compute_val_loss(dataloader, model, device, criterion):
    """
    Compute the validation loss and accuracy for a given dataset.

    Args:
        dataloader (DataLoader): DataLoader for the validation dataset.
        model (nn.Module): The trained model to be evaluated.
        device (torch.device): The device (CPU or CUDA) where the model and data are located.
        criterion (loss function): The loss function used for calculating validation loss.

    Returns:
        tuple: A tuple containing the accuracy and average loss for the validation dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0  # Initialize total loss
    total_correct = 0  # Initialize total correct predictions
    total_samples = 0  # Initialize total number of samples

    with torch.no_grad():  # Disable gradient computation
        for data in dataloader:  # Iterate over the data in the dataloader
            inputs, labels = data  # Extract inputs and labels
            inputs, labels = inputs.to(device), labels.to(device)  # Move to the specified device
            inputs = inputs.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor)  # Adjust the input dimensions
            outputs = model(inputs)  # Forward pass to get outputs from the model
            loss = criterion(outputs, labels)  # Compute loss
            total_loss += loss.item()  # Accumulate the loss
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            total_correct += (predicted == labels).sum().item()  # Count correct predictions
            total_samples += labels.size(0)  # Count total samples

    avg_loss = total_loss / len(dataloader)  # Calculate average loss
    accuracy = total_correct / total_samples  # Calculate accuracy
    return accuracy, avg_loss  # Return accuracy and average loss


class EarlyStopping:
    def __init__(self, patience=5):
        """
        Initialize the EarlyStopping object.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
        """
        self.patience = patience  # Number of epochs to wait before stopping
        self.counter = 0  # Counter to track the number of epochs without improvement
        self.best_accuracy = None  # Variable to store the best accuracy observed
        self.stop = False  # Flag to indicate whether to stop the training or not

    def __call__(self, accuracy):
        """
        Check if training should be stopped based on the provided accuracy.

        Args:
            accuracy (float): The accuracy obtained in the current epoch.
        """
        if self.best_accuracy is None:  # If it's the first epoch
            self.best_accuracy = accuracy  # Set the current accuracy as the best
        elif accuracy < self.best_accuracy:  # If the accuracy decreased
            self.counter += 1  # Increment the counter
            if self.counter >= self.patience:  # If the counter reached the patience limit
                self.stop = True  # Set the stop flag to True
        else:  # If the accuracy improved
            self.best_accuracy = accuracy  # Update the best accuracy
            self.counter = 0  # Reset the counter


def train_model(model, epochs, train_loader, valid_loader, model_name, lr=0.05,
                patience=5, device='cuda', model_save_path='models'):
    """
    Train the provided model.

    Args:
        model (nn.Module): The neural network model to be trained.
        epochs (int): The number of epochs to train the model.
        train_loader (DataLoader): DataLoader for the training data.
        valid_loader (DataLoader): DataLoader for the validation data.
        lr (float): Learning rate for the optimizer. Default is 0.05.
        patience (int): Number of epochs with no improvement after which training will be stopped. Default is 5.
        device (str): The device to use for training ('cuda' for GPU, 'cpu' for CPU). Default is 'cuda'.

    Returns:
        tuple: Tuple containing lists of training losses and validation losses for each epoch.
    """
    # Print the starting message with information about early stopping and number of steps per epoch
    print(f"Starting training with early stopping patience of {patience}")
    print(f"Each epoch has {len(train_loader)} steps.")

    # Initialize EarlyStopping object
    early_stopping = EarlyStopping(patience=patience)
    vis_iter = 20  # Frequency of visualizing loss
    
    # Set the model to training mode
    model.train()

    # Initialize lists to store losses
    loss_i = []
    loss_val_i = []

    # Initialize optimizer and loss criterion
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Set up Weights & Biases for monitoring
    wandb.watch(model, criterion, log="all", log_freq=vis_iter)

    # Create directory for saving models if it doesn't exist
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    best_accuracy = 0.0  # Initialize the best accuracy

    # Iterate over epochs
    for epoch in range(epochs):
        start_time = time.time()  # Record the start time of the epoch
        print(f'\nEpoch: {epoch+1}...')  # Print the epoch number
        running_loss = 0.0  # Initialize running loss

        # Iterate over batches in the training data
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data  # Get the inputs and labels
            inputs, labels = inputs.to(device), labels.to(device)  # Move to the specified device
            inputs = inputs.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor)  # Adjust the input dimensions

            optim.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)  # Forward pass
            #print(outputs.shape)
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optim.step()  # Update model parameters

            running_loss += loss.item()  # Accumulate loss

            # Log and print the training loss periodically
            if i > 0 and i % vis_iter == 0:
                wandb.log({"train_loss": running_loss / vis_iter, "step": epoch * len(train_loader) + i})
                print(f"Step: {i+1}/{len(train_loader)} Loss: {running_loss / vis_iter}")
                running_loss = 0.0
        
        # Perform validation if a validation loader is provided
        if valid_loader:
            accuracy, val_loss = compute_val_loss(valid_loader, model, device, criterion)
            loss_val_i.append(val_loss)  # Append validation loss
            print(f"Validation Loss: {val_loss:.4f} Accuracy: {accuracy:.4f}")
            wandb.log({"val_loss": val_loss, "val_accuracy": accuracy, "epoch": epoch})

            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(model_save_path, f"{model_name}_best_model.pth"))
                print("Best model saved.")

        loss_i.append(loss.item())  # Append training loss
        print("Epoch duration: {:.2f}s".format(time.time() - start_time))

        # Check for early stopping
        early_stopping(accuracy)
        if early_stopping.stop:
            print(f"\nEarly stopping invoked in epoch {epoch+1}")
            break

     # Save the last model after training completes
    torch.save(model.state_dict(), os.path.join(model_save_path, f"{model_name}_last_model.pth"))
    print("Last model saved.")


    return loss_i, loss_val_i  # Return lists of training and validation losses
