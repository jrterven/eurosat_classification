
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Define a class EurosatDataset that inherits from Dataset.
class EurosatDataset(Dataset):
    # Constructor method for the class.
    def __init__(self, _type, transform, data_path):
        
        # A dictionary mapping dataset types (valid, test, train) to their respective file paths.
        type_to_folder = {
            "valid": f"{data_path}validation.csv",
            "test": f"{data_path}test.csv",
            "train": f"{data_path}train.csv"
        }
        
        # Read the CSV file for the given type (valid, test, or train) using pandas and store the data.
        self.data = pd.read_csv(type_to_folder[_type])

        # Store the base path to the folder containing the data.
        self.folder_base = data_path
        
        self.transform = transform

    # Method to return the length of the dataset.
    def __len__(self):
        # Return the number of rows in the data.
        return len(self.data)

    # Method to get an item at a specific index from the dataset.
    def __getitem__(self, idx):
        # Convert the index to a list if it's a tensor.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract the filename (X) and label (Y) from the data at the given index.
        X, Y = self.data.iloc[idx].Filename, self.data.iloc[idx].Label

        # Convert the image path to a tensor.
        X = self.path_to_tensor(self.folder_base + X)

        # Convert the label to a long tensor.
        Y = torch.tensor(Y, dtype=torch.long)

        # Return the image tensor and its corresponding label.
        return X, Y

    # Method to convert an image path to a tensor.
    def path_to_tensor(self, path):
        # Open the image file.
        img = Image.open(path)

        # Apply the predefined transformations to the image.
        img_transformed = self.transform(img)

        # Permute the dimensions of the tensor for compatibility with PyTorch.
        return img_transformed.permute(1, 2, 0)


def visualize_classes(data_loader, index_to_label):
    # Set Seaborn's aesthetic parameters for a more polished look.
    sns.set(style="whitegrid", context="notebook")

    # Create a 2x5 grid of subplots with a suitable figure size.
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))

    # Adjust the space between the plots for better visibility.
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Dictionary to keep track of whether an image of each class has been found.
    found_classes = {}

    # Iterate over the batches in the data loader.
    for images, labels in data_loader:
        # Iterate over each image and its label in the current batch.
        for img, label in zip(images, labels):
            label_item = label.item()
            # If the image of this class hasn't been found yet, store it.
            if label_item not in found_classes:
                found_classes[label_item] = img

            # If images for all 10 classes have been found, break out of the loop.
            if len(found_classes) == 10:
                break

        # Check again outside the inner loop to break from the outer loop.
        if len(found_classes) == 10:
            break

    # Now that we have one image for each class, display them on the grid.
    for i, (label, img) in enumerate(found_classes.items()):
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        elif img.shape[0] == 1:
            img = img.squeeze(0)
        img = img.numpy()
        axs[i // 5, i % 5].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axs[i // 5, i % 5].set_title(index_to_label[label])
        axs[i // 5, i % 5].axis('off')
        sns.despine(ax=axs[i // 5, i % 5], left=True, bottom=True)  # Remove spines for a cleaner look

    plt.tight_layout()
    plt.show()