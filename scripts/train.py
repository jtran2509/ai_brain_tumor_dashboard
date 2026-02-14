from data_loader import get_brain_loaders # import function from data_loader.py
from models import get_model # Import model from models.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import os
import copy

# 1. Define Config class
class Config:
    batch_size = 32
    epochs = 50
    lr = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 2. Define Early Stopping class
class EarlyStopping():
    """
    Create an early stopping callback that will stop the training
    and restore the best weights when a certain patience interval is hit.
    """

    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ''

    def __call__(self, model, val_loss):
        if self.best_loss == None: # Save the current loss as the "best" and keeps a copy of the model
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta: # if the model get better by significant amount?
            self.best_loss = val_loss # update best loss
            self.counter = 0 # Reset counter to 1
            self.best_model.load_state_dict(model.state_dict()) # Save new best model weights
        elif self.best_loss - val_loss < self.min_delta: # If the model stays the same or got worse
            self.counter +=1
            if self.counter >= self.patience: # Counter reaches patience limit
                self.status = f"Stopped on {self.counter}"
                if self.restore_best_weights: # Restore best weights
                    model.load_state_dict(self.best_model.state_dict())
                return True
        self.status = f'{self.counter}/{self.patience}'
        return False

# 3. Define BrainTumorClassifier (integrate training, Early Stopping, Ploting, predict, etc.)
class BrainTumorClassifier:
    def __init__(self, model, optimizer, criterion, val_transformations, train_dataset, config, output_folder=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.output_folder = 'torch_models/' if output_folder is None else output_folder
        self.history = {}
        self.val_transformations = val_transformations
        self.classes = train_dataset.classes
        self.Config = config

    def train(self, training_loader, eval_loader, num_epochs, output_filename):
        """
        Method to train the model for a given number of epochs using the provided training and evaluation data loaders

        Args:
            training_loader (torch.utils.data.dataloader.DataLoader): PyTorch data loader for training data
            eval_loader (torch.utils.data.dataloader.DataLoader): PyTorch data loader for evaluation data
            num_epochs (int): Number of epochs to train the model.
            output_filename (str): Name of file to save the trained model to.

        Returns:
            None. Trains the model and saves it to the specified output directory
        """

        # Create output directory
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        file_dir = os.path.join(self.output_folder, output_filename)

        # Initialize metrics
        train_accuracies, train_losses = [], []
        val_accuracies, val_losses = [], []

        # Define early stopping variables
        early_stopper = EarlyStopping()
        best_epoch = 0
        best_val_loss = np.inf

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            #Train one epoch
            train_acc, train_loss = self._train_epoch(training_loader)
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)

            #Eval one epoch
            val_acc, val_loss = self._eval_epoch(eval_loader)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)

            print()

            #Update the best model weights
            if val_loss < best_val_loss:
                best_val_loss = val_loss #Update the best val loss
                best_epoch = epoch+1 # Update best epoch for best val loss

            # Check for early stopping
            if early_stopper(self.model, val_loss):
                print(f"Validation loss has not improved for {early_stopper.patience} epochs. Early Stopping...")
                print(f"Reverting back to weights for epoch {best_epoch}.")
                break

            # End of training
            print("Training finished!")

            # Save model
            torch.save(self.model.state_dict(), file_dir)

            # Get history
            self.history = {
                'accuracy': {'train': train_accuracies, 'eval': val_accuracies},
                'loss': {'train': train_losses, 'eval': val_losses}
            }

    def plot_training(self):
        """
        Plots the training and validation accuracy and loss over each epoch. Retrieves the accuracy and loss
        values from the `self.history` dictionary and creates two subplots to display the accuracy and loss
        data separately.
        """

        # Get the accuracies and loss values from self.history
        train_accuracies = self.history['accuracy']['train']
        val_accuracies = self.history['accuracy']['eval']
        train_losses = self.history['loss']['train']
        val_losses = self.history['loss']['eval']

        # Create subplots for accuracies and loss plots
        fig, axs = plt.subplots(2, 1, figsize = (10, 6))
        fig.subplots_adjust(hspace = 0.5)

        # Plot training and validation accuracies
        axs[0].plot(train_accuracies, label = 'Training accuracy')
        axs[0].plot(val_accuracies, label = 'Validation accuracy')
        axs[0].set_title("Accuracy")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend()

        # Plot training and validation losses
        axs[1].plot(train_losses, label = 'Training losses')
        axs[1].plot(val_losses, label = "Validation losses")
        axs[1].set_title("Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        axs[1].legend()

        plt.show()

    def evaluate(self, test_loader):
        """
        Evaluates the model on a given test loader and returns the accuracy and loss metrics.
        Uses the 'eval_epoch' method to calculate the metrics for each batch in the loader and returns the fina
        metrics for the entire dataset.

        Args:
            test_loader (torch.utils.data.dataloader.DataLoader): PyTorch DataLoader for the test dataset

        Returns:
            Tuple of the average accuracy and loss the dataset
        """
        return self._eval_epoch(test_loader, return_metrics= True)

    def predict(self, img):
        """
        Takes an image and returns the predicted class label. Applies the validation transformations to the
        image, performs a forward pass through the model, and returns the predicted class.

        Args:
            img: PIL image object of the input image
        Returns:
            The predicted class label as a string
        """
        self.model.eval()
        img = self.val_transformations(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.Config.device)
        pred = self.model(img)
        pred_idx = torch.argmax(pred, dim=1)
        return self.classes[pred_idx]

    def _train_epoch(self, training_loader):
        """
        Train a single epoch of the model

        Args:
            training_loader (torch.utils.data.DataLoader): The PyTorch data loader object for the training set

        Returns:
            Tuple of floats: The average training accuracy and avergae training loss for the epoch.
        """
        running_loss = 0.00
        running_acc = 0.00
        self.model.train()

        prog_bar = tqdm(enumerate(training_loader), total = len(training_loader))
        for batch_idx, (imgs, labels) in prog_bar:

            # Configure to device
            imgs, labels = imgs.to(self.Config.device), labels.to(self.Config.device)

            # Zero out gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)

            #Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate loss of batch in average
            running_loss += loss.item()

            # Calculate accuracy
            running_acc += (outputs.argmax(-1) == labels).float().mean().item()
            avg_loss = running_loss / (batch_idx + 1)
            avg_acc = running_acc / (batch_idx + 1)

            prog_bar.set_description(f" Batch {batch_idx+1} / {len(training_loader)} - Avg Train Loss: {avg_loss:.4f}, Avg Train Accuracy: {avg_acc:.4f}")
        return avg_acc, avg_loss

    def _eval_epoch(self, eval_loader, return_metrics = True):
        """
        Evaluate the performance of the model on a validation set for a single epoch.

        Args:
            eval_loader (torch.utils.data.dataloader.DataLoader): PyTorch validation loader
            return_metrics (bool): Whether to return the evaluation metrics (accuracy and loss).

        Returns:
            tuple: a tuple containing the evaluation metrics. The first element is the average validation accuracy,
            and the 2nd one  if the average validation loss.
        """

        running_loss = 0.00
        running_acc = 0.00

        self.model.eval()
        prog_bar = tqdm(enumerate(eval_loader), total = len(eval_loader))
        for batch_idx, (imgs, labels) in prog_bar:

            # Configure to device
            imgs, labels = imgs.to(self.Config.device), labels.to(self.Config.device)

            # Forward pass
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)

            # Calculate the loss of batch in average
            running_loss += loss.item()

            # Calculate accuracy
            running_acc += (outputs.argmax(-1) == labels).float().mean().item()

            avg_loss = running_loss / (batch_idx+1)
            avg_acc = running_acc / (batch_idx+1)

            prog_bar.set_description(f"Batch {batch_idx+1}/{len(eval_loader)} - Avg Val Loss: {avg_loss:.4f}, Avg Val Accuracy: {avg_acc:.4f}")

        if return_metrics:
            return avg_acc, avg_loss

# 4. Let's connect everything and run the model!
if __name__=="__main__":
    # 1. Load data
    train_loader, val_loader, test_loader, classes = get_brain_loaders(
        "D:\ai_brain_tumor_dashboard\data", batch_size=Config.batch_size)
    # 2. Init model from model.py
    model = get_model(num_classes=4, device=Config.device)
    # 3. Define loss and Optimizer
    loss_function = torch.nn.CrossEntropyLoss() # Calculate how far off the models' prediction from the real answer
    optimizer = torch.optim.Adam(model.parameters(), lr = Config.lr)
    # 4. Initialize the model
    trainer = BrainTumorClassifier(model=model, optimizer=optimizer,
                                criterion = loss_function,
                                train_dataset=train_loader.dataset, config=Config, output_folder="../models/densenet_v1/")

    # Run the model & START TRAINING!
    trainer.train(
    training_loader = train_loader,
    eval_loader = val_loader,
    num_epochs = Config.epochs,
    output_filename = "best_model.pth"
)




