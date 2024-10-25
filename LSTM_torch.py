import os
import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing, metrics
from torch.utils.data import DataLoader, TensorDataset
import pandas_ta as ta
from sklearn.utils.class_weight import compute_class_weight

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set parameters
n_steps = 50      # 50 time steps
n_inputs = 11     # Size of input vector
n_neurons = 256   # Number of neurons in LSTM
n_outputs = 3     # Number of output classes

learning_rate = 0.0001
batch_size = 128
n_epochs = 300

# Load training data
csv_files_train = [f for f in os.listdir('data') if f.endswith('.csv')]
df_raw_train = pd.concat([pd.read_csv(os.path.join('./data', f)) for f in csv_files_train], axis=0)
train = df_raw_train[100:]
train = train.loc[~train.index.duplicated(keep='first')].reset_index(drop=True)

# Load test data
csv_files_test = [f for f in os.listdir('test') if f.endswith('.csv')]
df_raw_test = pd.concat([pd.read_csv(os.path.join('./test', f)) for f in csv_files_test], axis=0)
test = df_raw_test[100:]
test = test.loc[~test.index.duplicated(keep='first')].reset_index(drop=True)

def get_sample_weights(y):
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

# Function to create features and labels (same logic as original)
def create_feature_label(df_ori):
    df = df_ori.copy()
    df_length = len(df)

    # Initialize new columns
    df['mix_mv_avg'] = np.nan
    df['mv_avg_diff'] = np.nan
    df['avg_quantity'] = np.nan
    df['quantity_price'] = np.nan
    df['price_diff'] = np.nan
    df['5_price_diff'] = np.nan
    df['ct_rising'] = np.nan
    df['label'] = np.nan
    df['trend_name'] = 2  # Default to 2 (no clear trend)
    df['durration_trend'] = 0
    df['number_pullback'] = 0
    df['kill_zone'] = 0
    first_index = df_ori.index[0]

    o =df_ori['Open'].values
    c =df_ori['Close'].values
    h =df_ori['High'].values
    l = df_ori['Low'].values
    v = df_ori['volume'].astype(float).values

    for i in df_ori.index:
        # Kill zone
        date_time = pd.to_datetime(df.Date.loc[i])
        hour = date_time.hour
        if hour >= 0 and hour <= 5:
            df.at[i, 'kill_zone'] = 1
        elif hour >= 7 and hour <= 10:
            df.at[i, 'kill_zone'] = 2
        elif hour >= 12 and hour <= 15:
            df.at[i, 'kill_zone'] = 3
        elif hour >= 18 and hour <= 20:
            df.at[i, 'kill_zone'] = 4
        else:
            df.at[i, 'kill_zone'] = 0

        mv_avg_3 = df.Close.loc[max(first_index, i - 2):i + 1].mean()
        mv_avg_6 = df.Close.loc[max(first_index, i - 5):i + 1].mean()
        mv_avg_25 = df.Close.loc[max(first_index, i - 24):i + 1].mean()
        mv_avg_34 = df.Close.loc[max(first_index, i - 33):i + 1].mean()

        df.at[i, 'mix_mv_avg'] = np.mean([mv_avg_3, mv_avg_6, mv_avg_25, mv_avg_34])
        df.at[i, 'mv_avg_diff'] = mv_avg_3 - mv_avg_6

        avg_quantity = df.volume.loc[max(first_index, i - 4):i + 1].mean()
        df.at[i, 'avg_quantity'] = avg_quantity
        df.at[i, 'quantity_price'] = df.volume.loc[i] / df.Close.loc[i]

        df.at[i, 'price_diff'] = df.Close.loc[i] - df.Close.loc[max(first_index, i - 1)]
        df.at[i, '5_price_diff'] = df.Close.loc[i] - df.Close.loc[max(first_index, i - 4)]

        rising_count = df.Close.loc[max(first_index, i - 9):i + 1].diff().gt(0).sum()
        df.at[i, 'ct_rising'] = rising_count

        if i + 13 < df_length and i - first_index > 12:
            window_begin = i-12
            window_end = i+13
            high_window = df.High.loc[i - 12:i + 13]
            low_window = df.Low.loc[i - 12:i + 13]

            max_high = high_window.max()  # Max High in next 24 days
            min_low = low_window.min()    # Min Low in next 24 days
            close_price = df.Close.loc[i]

            max_high_index = high_window.idxmax()
            min_low_index = low_window.idxmin()

            window_middle = int((window_begin + window_end) / 2)
            min_after = df.Close.loc[max_high_index : window_end].min()
            max_after = df.Close.loc[min_low_index : window_end].max()

            if max_high_index == window_middle and max_high - min_after > 10:
                df.at[i, 'label'] = 0
            elif min_low_index == window_middle and max_after - min_low < 10:
                df.at[i, 'label'] = 1
            else:
                df.at[i, 'label'] = 2
            

            # if max_high_index == i:
            #     df.at[i, 'label'] = 0
            # elif min_low_index == i:
            #     df.at[i, 'label'] = 1

            # if (max_high - close_price > 15) and (close_price - min_low < 5):
            #     df.at[i, 'label'] = 1
            # elif (max_high - close_price < 5) and (close_price - min_low > 15):
            #     df.at[i, 'label'] = 0
            # if (max_high - close_price > 15) and (close_price - min_low > 15):
            #     if max_high_index < min_low_index:
            #         min_low_small =  df.Low.loc[i + 1:max_high_index].min()
            #         if close_price - min_low_small < 5:
            #             df.at[i, 'label'] = 1
            #         else:
            #             df.at[i, 'label'] = 2
            #     else:
            #         max_high_small = df.Low.loc[i + 1:min_low_index].max()
            #         if max_high_small - close_price < 5:
            #             df.at[i, 'label'] = 0
            #         else:
            #             df.at[i, 'label'] = 2
            # elif (max_high - close_price > 15) and (close_price - min_low <= 15):
            #     min_low_small =  df.Low.loc[i + 1:max_high_index].min()
            #     if close_price - min_low_small < 5:
            #         df.at[i, 'label'] = 1
            #     else:
            #         df.at[i, 'label'] = 2
            # elif (max_high - close_price <= 15) and (close_price - min_low > 15):
            #     max_high_small = df.Low.loc[i + 1:min_low_index].max()
            #     if max_high_small - close_price < 5:
            #         df.at[i, 'label'] = 0
            #     else:
            #         df.at[i, 'label'] = 2
            # else:
            #     df.at[i, 'label'] = 2
        else:
            df.at[i, 'label'] = 2

        technical_info = df['technical_info'].loc[i]
        trend_data = ast.literal_eval(technical_info).get('current', [])
        if trend_data:
            trend_name = 0 if 'down' in trend_data[0]['trend_name'] else 1
            durration_trend = trend_data[0].get('duration_trend', 0)
            number_pullback = trend_data[0].get('number_pullback', 0)
        else:
            trend_name = 2
            durration_trend = 0
            number_pullback = 0

        df.at[i, 'trend_name'] = trend_name
        df.at[i, 'durration_trend'] = durration_trend
        df.at[i, 'number_pullback'] = number_pullback

    return df

def reshape(df, window_size=50, n_inputs=11):
    df_as_array = np.array(df)
    temp = np.array([np.arange(i - window_size, i) for i in range(window_size, df.shape[0])])
    new_df = df_as_array[temp[0:len(temp)]]
    return new_df    

def inference(model, df_row, scaler, window_size=50):
    """
    Perform inference on a single input row from the DataFrame.

    Parameters:
    - model: Trained LSTM model.
    - df_row: A DataFrame row with required columns as input.
    - scaler: Scaler object used during training.
    - window_size: Number of time steps (should match training).

    Returns:
    - predicted_label: Predicted class label.
    """
    # Extract the relevant features from the row and reshape them
    features = df_row[col_for_x].values.reshape(1, -1)
    
    # Scale the features using the fitted scaler
    features_scaled = scaler.transform(features)

    # Reshape to match the LSTM input shape (1, n_steps, n_inputs)
    features_reshaped = features_scaled.reshape(1, window_size, n_inputs)

    # Convert to a PyTorch tensor and send to the same device as the model
    input_tensor = torch.tensor(features_reshaped, dtype=torch.float32).to(device)

    # Set the model to evaluation mode and make a prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()  # Get the predicted class

    return predicted_class

# Add features and labels for train and test
train_2 = create_feature_label(train)
test_2 = create_feature_label(test)

# Define features
col_for_x = ['mix_mv_avg', '5_price_diff', 'mv_avg_diff', 'avg_quantity', 
             'quantity_price', 'ct_rising', 'Close', 'trend_name', 
             'durration_trend', 'number_pullback', 'kill_zone']

X_train = train_2[col_for_x]
y_train = train_2['label']
X_test = test_2[col_for_x]
y_test = test_2['label']

# Normalize values for train and test
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = y_train[n_steps:]
y_test = y_test[n_steps:]
# One-hot encode labels
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values

x_train_all = reshape(X_train_scaled)

x_test_all = reshape(X_test_scaled)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train_all, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

x_test_tensor = torch.tensor(x_test_all, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(n_inputs, n_neurons, batch_first=True)
        self.fc = nn.Linear(n_neurons, n_outputs)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

model = LSTMModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_loss = float('inf')
folder_path = 'weights'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
MODEL_PATH =   os.path.join(folder_path, 'model_lstm.pt')

# Training loop
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, torch.argmax(y_batch, dim=1))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), MODEL_PATH)
        

# Evaluate the model
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Initialize lists to store results
y_test_pred_classes = []
y_test_true_classes = []
confidences = []

with torch.no_grad():
    for i in range(len(x_test_tensor)):
        # Get model predictions and confidence
        output = model(x_test_tensor[i].unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

        # Store predictions, true labels, and confidences
        y_test_pred_classes.append(predicted_class)
        y_test_true_classes.append(torch.argmax(y_test_tensor[i]).item())
        confidences.append(confidence)

# Convert lists to NumPy arrays for easier manipulation
y_test_pred_classes = np.array(y_test_pred_classes)
y_test_true_classes = np.array(y_test_true_classes)
confidences = np.array(confidences)

# Filter predictions with confidence > 0.8
mask = confidences > 0
y_pred_filtered = y_test_pred_classes[mask]
y_true_filtered = y_test_true_classes[mask]

# Calculate accuracy with filtered predictions
filtered_accuracy = metrics.accuracy_score(y_true_filtered, y_pred_filtered)
print(f'Test Accuracy (confidence > 0.8): {filtered_accuracy:.4f}')

# Create crosstab for filtered predictions
crosstab_result = pd.crosstab(
    y_true_filtered, y_pred_filtered, 
    rownames=['True Label'], colnames=['Predicted Label']
)

print("Crosstab of True vs Predicted Labels (confidence > 0.8):")
print(crosstab_result)
