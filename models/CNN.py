import os
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F  # Import thêm F cho loss thủ công
from operator import itemgetter

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 3 * 3, 64)  # 32 kênh * 3 * 3 kích thước ảnh
        self.fc2 = nn.Linear(64, 3)  # 3 lớp đầu ra (0, 1, 2)

    def forward(self, x):
        # Tầng tích chập + ReLU + BatchNorm + MaxPool
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        
        # Làm phẳng tensor để đưa vào tầng fully connected
        x = torch.flatten(x, 1)
        
        # Tầng fully connected + ReLU
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Tầng đầu ra (không cần activation)
        return x


class CNNModel():
    def __init__(self, X_train, y_train, X_test, y_test, list_features):

        self.preprocess_data(X_train, y_train, X_test, y_test,  list_features)
      
        # Initialize model, loss function and optimizer
        self.model = CNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Define output path
        folder_path = 'weights'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.output_path = os.path.join(folder_path, 'model_cnn.pt')

    def get_sample_weights(self, y):
        """
        calculate the sample weights based on class weights. Used for models with
        imbalanced data and one hot encoding prediction.

        params:
            y: class labels as integers
        """

        y = y.astype(int)  # compute_class_weight needs int labels
        class_weights = compute_class_weight('balanced' , np.unique(y), y)
        
        print("real class weights are {}".format(class_weights), np.unique(y))
        print("value_counts", np.unique(y, return_counts=True))
        sample_weights = y.copy().astype(float)
        for i in np.unique(y):
            sample_weights[sample_weights == i] = class_weights[i]  # if i == 2 else 0.8 * class_weights[i]
            # sample_weights = np.where(sample_weights == i, class_weights[int(i)], y_)

        return sample_weights
    
    def reshape_as_image(self, x, img_width, img_height):
        x_temp = np.zeros((len(x), img_height, img_width))
        for i in range(x.shape[0]):
            # print(type(x), type(x_temp), x.shape)
            x_temp[i] = np.reshape(x[i], (img_height, img_width))

        return x_temp

    def preprocess_data(self, X_train, y_train, X_test, y_test, list_features):
        number_feature = 9
        select_k_best = SelectKBest(k=number_feature)
        select_k_best.fit(X_train, y_train)
        selected_features = itemgetter(*select_k_best.get_support(indices=True))(list_features)

        feature_index = []
        for i in selected_features:
            feature_index.append(list_features.index(i))

        x_train = X_train[:, feature_index]
        x_test = X_test[:, feature_index]

        dim_image = int(np.sqrt(number_feature))
        x_train = self.reshape_as_image(x_train, dim_image, dim_image)
        x_test = self.reshape_as_image(x_test, dim_image, dim_image)

        # adding a 1-dim for channels (3)
        x_train = np.stack((x_train,) * 3, axis=-1)
        x_test = np.stack((x_test,) * 3, axis=-1)

        # convert to tensor
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.x_test = torch.tensor(x_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)


    # Training function
    def train_model(self, num_epochs):
        self.model.train()
        train_losses, test_accuracies = [], []
        best_accuracy = 0

        # Tính sample weights cho dữ liệu huấn luyện
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.x_train)
            loss = self.criterion(outputs, self.y_train)
            loss.backward()
            self.optimizer.step()
            
            train_losses.append(loss.item())
         
            # Evaluate the model
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(self.x_test)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == self.y_test).float().mean().item()
                test_accuracies.append(accuracy)

                # save model if it has the best accuracy
                torch.save(self.model.state_dict(), self.output_path)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')
            self.model.train()

        return train_losses, test_accuracies

    # Testing function
    def test_model(self):
        self.model.eval()
        self.model.load_state_dict(torch.load(self.output_path))
        total = 0
        correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            outputs = self.model(self.x_test)
            _, predicted = torch.max(outputs.data, 1)

            # Collect all predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(self.y_test.cpu().numpy())

            # Calculate total and correct predictions
            total += self.y_test.size(0)
            correct += (predicted == self.y_test).sum().item()

        # Compute accuracy
        accuracy = correct / total
        print(f'Accuracy: {accuracy:.4f}')

        # Create a crosstab of actual vs predicted values
        crosstab_result = pd.crosstab(
            pd.Series(all_labels, name='Actual'),
            pd.Series(all_preds, name='Predicted')
        )
        print(crosstab_result)

        # Print detailed classification report
        print(metrics.classification_report(all_labels, all_preds))
            

