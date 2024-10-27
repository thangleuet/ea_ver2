import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Step 1: CNN Model Definition
class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)

        conv1_output_size = input_size - 2
        pool1_output_size = conv1_output_size // 2
        conv2_output_size = pool1_output_size - 2
   
        self.fc1 = nn.Linear(32 * conv2_output_size, 128)  # You may need to calculate this size based on pooling
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        # x should have shape (batch_size, 1, num_features) for Conv1d
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)  # Max pooling layer
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
class CNNModel():
    def __init__(self, X_train, y_train, X_test, y_test, list_features):
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        input_size = X_train.shape[1]  # Number of input features
        num_classes = len(np.unique(y_train))  # Number of output classes
        self.model = CNN(input_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self, num_epochs):
        train_losses, test_accuracies = [], []

        for epoch in range(num_epochs):
            self.model.train()
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            train_losses.append(loss.item())

            # Evaluate the model
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(self.X_test_tensor)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == self.y_test_tensor).float().mean().item()
                test_accuracies.append(accuracy)

            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')
            self.model.train()
       
        return train_losses, test_accuracies

    def test_model(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test_tensor)
            loss = self.criterion(outputs, self.y_test_tensor)
        print(f'Test Loss: {loss.item():.4f}')
        accuracy = (outputs.argmax(1) == self.y_test_tensor).float().mean().item()
        print(f'Test Accuracy: {accuracy:.4f}')
        print(pd.crosstab(self.y_test_tensor, outputs.argmax(1), rownames=['Actual'], colnames=['Predicted']))
        print(metrics.classification_report(self.y_test_tensor, outputs.argmax(1)))


    
       