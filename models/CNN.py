import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
import torch.nn.functional as F 

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=3)  # Assuming 10 features
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 6, 128)  # Adjust input size based on the flattened output
        self.fc2 = nn.Linear(128, 3)  # Assuming 3 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNModel():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.x_train = torch.tensor(np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1), dtype=torch.float32).permute(0, 2, 1)
        self.y_train = torch.tensor(np.array(y_train), dtype=torch.long)
        self.x_test = torch.tensor(np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1), dtype=torch.float32).permute(0, 2, 1)
        self.y_test = torch.tensor(np.array(y_test), dtype=torch.long)
        
        # Create DataLoader
        self.train_dataset = TensorDataset( self.x_train, self.y_train)
        self.test_dataset = TensorDataset(self.x_test, self.y_test)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        # Initialize model, loss function and optimizer
        self.model = CNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    # Training function
    def train_model(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Testing function
    def test_model(self):
        self.model.eval()
        total = 0
        correct = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        print(f'Accuracy: {accuracy:.4f}')
        print(pd.crosstab(all_labels, all_preds, rownames=['Actual'], colnames=['Predicted']))
        print(metrics.classification_report(all_labels, all_preds))

