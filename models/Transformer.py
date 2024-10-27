import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Transformer Model Definition
class Transformer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_size, 64)  # Embedding layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4), num_layers=2
        )
        self.fc = nn.Linear(64, num_classes)  # Final classification layer

    def forward(self, x):
        # x should have shape (batch_size, seq_len, input_size)
        x = self.embedding(x)  # Apply embedding layer
        x = x.permute(1, 0, 2)  # Change shape to (seq_len, batch_size, embedding_dim)
        x = self.transformer_encoder(x)  # Pass through transformer
        x = x.mean(dim=0)  # Take the mean over the sequence length
        x = self.fc(x)  # Pass to the final classification layer
        return x

class TransformerModel:
    def __init__(self, X_train, y_train, X_test, y_test, list_features):

        window_size = 12

        X_train = self.get_sequence_data(X_train, window_size)
        X_test = self.get_sequence_data(X_test, window_size)
        y_train = y_train[window_size:]
        y_test = y_test[window_size:]

        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        input_size = X_train.shape[-1]  # Number of input features
        num_classes = len(np.unique(y_train))  # Number of output classes
        self.model = Transformer(input_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_sequence_data(self, data, window_size=12):
        """Generates sliding window sequences from input data."""
        sequences = [data[i - window_size:i] for i in range(window_size, len(data))]
        return np.array(sequences)
        

    def train_model(self, num_epochs):
        train_losses, test_accuracies = [], []

        for epoch in range(num_epochs):
            self.model.train()
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)  # Forward pass
                loss = self.criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights

            train_losses.append(loss.item())

            # Evaluate the model
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(self.X_test_tensor)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == self.y_test_tensor).float().mean().item()
                test_accuracies.append(accuracy)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')
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
