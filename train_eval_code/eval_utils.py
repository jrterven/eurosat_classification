
import os
import torch
import torch.nn as nn
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve

def remove_wandb_hooks(model):
    """
    Remove all forward hooks added by Weights & Biases (wandb) from a PyTorch model.

    This function iterates through all modules in the given model and clears any
    forward hooks that have been attached to the modules. It is particularly useful
    when you want to evaluate a model without the interference of wandb logging.

    Args:
        model: The PyTorch model from which to remove the wandb hooks.
    
    Note:
        This function removes all forward hooks from the model. If there are
        essential hooks that the model requires for its functionality (apart from wandb),
        they will be removed too. Use this function only if you are sure that removing
        all forward hooks won't affect your model's performance.
    """
    for module in model.modules():
        # Check if the module has any forward hooks.
        if hasattr(module, '_forward_hooks'):
            # If so, clear all forward hooks from the module.
            # This includes hooks added by wandb or any other source.
            module._forward_hooks.clear()



def evaluate_test_set(model, test_loader, device, index_to_label):
    """
    Evaluate the given model on the test set.

    Args:
        model: The neural network model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        device: The device (e.g., 'cuda' or 'cpu') on which the model is to be evaluated.

    This function performs a forward pass through the model for each batch in the test_loader,
    collects the predictions, and prints a classification report.
    """
    
    # Set the model to evaluation mode. This will turn off features like dropout.
    model.eval()

    # Lists to store all predictions and true labels.
    all_preds = []
    all_labels = []

    # Disable gradient computation. Gradients are not needed for evaluation and 
    # disabling them reduces memory usage and speeds up computations.
    with torch.no_grad():
        for data in test_loader:
            # Extract inputs (images, etc.) and labels (ground truth) from the data.
            inputs, labels = data

            # Move inputs and labels to the specified device (GPU or CPU).
            inputs, labels = inputs.to(device), labels.to(device)

            # Adjust the input dimensions and convert to float. 
            # PyTorch expects the input dimensions to be [batch_size, channels, height, width].
            inputs = inputs.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor)

            # Forward pass: compute the output of the model given the inputs.
            outputs = model(inputs)

            # Get the index of the highest score in the output tensor for each input. 
            # This index corresponds to the predicted class.
            _, predicted = torch.max(outputs.data, 1)

            # Extend the lists with predictions and true labels.
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print the classification report comparing true and predicted labels.
    # 'index_to_label.values()' should be a list of label names in the same order as the indices.
    print(classification_report(all_labels, all_preds, target_names=index_to_label.values(), digits=3))

    

def precision_recall_analysis(model, test_loader, device, output_path, model_name, index_to_label):
    """
    Evaluates a given model on the test dataset and generates precision-recall curves for each class
    and all classes combined. It also saves these values to a JSON file.

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to be evaluated.

    test_loader : torch.utils.data.DataLoader
        DataLoader containing the test dataset.

    device : torch.device
        The device (CPU or GPU) to run the model on.

    output_path : str
        The path where the output JSON file will be saved.

    model_name : str
        The name of the model, used for naming the output JSON file.

    index_to_label : dict
        A dictionary mapping class indices to their respective labels.

    Returns:
    --------
    None
        This function does not return anything. It saves the precision-recall data to a JSON file
        and displays the precision-recall curves.

    Notes:
    ------
    - This function requires the model to be in evaluation mode.
    - It disables gradient computation during evaluation to save memory.
    - It assumes that the inputs are 4D tensors, with the color channel as the last dimension.

    Raises:
    -------
    FileNotFoundError
        If the specified output path is not found.
    """
    
    # Set the model to evaluation mode
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient computation
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor)
            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels_one_hot = np.eye(len(index_to_label))[all_labels]
    
    precision_recall_data = {}

    # Precision-Recall curve for each class
    for i, label in enumerate(index_to_label.values()):
        precision, recall, _ = precision_recall_curve(all_labels_one_hot[:, i], all_probs[:, i])
        plt.plot(recall, precision, label=f'Class {label}')

    # Precision-Recall curve for all classes
    precision, recall, _ = precision_recall_curve(all_labels_one_hot.ravel(), all_probs.ravel())
    plt.plot(recall, precision, label='All Classes', color='black', lw=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

    # Save precision and recall for all classes
    precision_recall_data['All Classes'] = {'precision': precision.tolist(), 'recall': recall.tolist()}

    # Save the data to a JSON file
    filename = f"{model_name}_precision_recall_values.json"
    output_path = os.path.join(output_path, filename)
    with open(output_path, 'w') as json_file:
        json.dump(precision_recall_data, json_file, indent=4)

