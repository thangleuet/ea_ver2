import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.preprocessing import StandardScaler
import os
import ast

# Parameters


# Helper: Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(TransformerEncoder, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)  # Residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)    # Residual connection
        return x

# Full Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, n_steps, n_inputs):

        self.n_steps = n_steps        # Sequence length
        self.n_inputs = n_inputs       # Input features
        self.n_outputs = 3       # Output classes
        self.d_model = 128        # Embedding/Model dimension
        self.n_heads = 8         # Number of attention heads
        self.d_ff = 256          # Feedforward layer size
        self.n_layers = 4        # Number of Transformer layers
        self.batch_size = 64

        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(self.n_inputs, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(self.d_model, self.n_heads, self.d_ff) for _ in range(self.n_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.d_model, self.n_outputs)

    def forward(self, x):
        x = self.input_proj(x)  # (batch_size, n_steps, d_model)
        x = self.pos_encoder(x.permute(1, 0, 2))  # (n_steps, batch_size, d_model)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.global_pool(x.permute(1, 2, 0)).squeeze(-1)  # (batch_size, d_model)
        output = self.fc(x)
        return output

# Instantiate the model
n_steps = 48
n_inputs = 11
model = TransformerModel(n_steps, n_inputs)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Helper: Training loop
def train_model(model, X_train, y_train, X_val, y_val, model_save_path):
    best_val_loss = np.inf
    n_epochs = 300
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_accuracy =  (val_outputs.argmax(1) == y_val).float().mean().item()  

            # Save the model if the validation accuracy improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"New best model saved with Val Accuracy: {val_accuracy:.4f}")

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}")
            # torch.save(model.state_dict(), f"weights/model_transformer_epoch_{epoch}.pt")

def get_feature(index, df):
    first_index = df.index[0]
    df_length = len(df)

    date_time = pd.to_datetime(df.Date.loc[index])
    hour = date_time.hour
    if hour >= 0 and hour <= 5:
        df.at[index, 'kill_zone'] = 1
    elif hour >= 7 and hour <= 10:
        df.at[index, 'kill_zone'] = 2
    elif hour >= 12 and hour <= 15:
        df.at[index, 'kill_zone'] = 3
    elif hour >= 18 and hour <= 20:
        df.at[index, 'kill_zone'] = 4
    else:
        df.at[index, 'kill_zone'] = 0

    # Calculate moving averages and other features
    mv_avg_3 = df.Close.loc[max(first_index, index - 2):index + 1].mean()
    mv_avg_7 = df.Close.loc[max(first_index, index - 6):index + 1].mean()
    mv_avg_25 = df.Close.loc[max(first_index, index - 24):index + 1].mean()
    mv_avg_34 = df.Close.loc[max(first_index, index - 33):index + 1].mean()

    df.at[index, 'mix_mv_avg'] = np.mean([mv_avg_3, mv_avg_7, mv_avg_25, mv_avg_34])
    df.at[index, 'mv_avg_diff'] = mv_avg_3 - mv_avg_7

    avg_quantity = df.volume.loc[max(first_index, index - 4):index + 1].mean()
    df.at[index, 'avg_quantity'] = avg_quantity
    df.at[index, 'quantity_price'] = df.volume.loc[index] / df.Close.loc[index]

    df.at[index, 'price_diff'] = df.Close.loc[index] - df.Close.loc[max(first_index, index - 1)]
    df.at[index, '5_price_diff'] = df.Close.loc[index] - df.Close.loc[max(first_index, index - 4)]

    rising_count = (df.Close.loc[max(first_index, index - 9):index + 1].diff().gt(0).sum())
    df.at[index, 'ct_rising'] = rising_count

    if index + 6 < df_length:
        high_window = df.High.loc[index - 5:index + 6]
        low_window = df.Low.loc[index - 5:index + 6]

        max_high = high_window.max()  # Max High in next 24 days
        min_low = low_window.min()    # Min Low in next 24 days
        close_price = df.Close.loc[index]

        max_high_index = high_window.idxmax()
        min_low_index = low_window.idxmin()

        if max_high_index == index:
            df.at[index, 'label'] = 0
        elif min_low_index == index:
            df.at[index, 'label'] = 1

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
        else:
            df.at[index, 'label'] = 2
    else:
        df.at[index, 'label'] = 2


    # Extract technical information
    technical_info = df['technical_info'].loc[index]
    trend_data = ast.literal_eval(technical_info).get('current', [])
    if trend_data:
        trend_name = 0 if 'down' in trend_data[0]['trend_name'] else 1
        durration_trend = trend_data[0].get('duration_trend', 0)
        number_pullback = trend_data[0].get('number_pullback', 0)
    else:
        trend_name = 2
        durration_trend = 0
        number_pullback = 0

    df.at[index, 'trend_name'] = trend_name
    df.at[index, 'durration_trend'] = durration_trend
    df.at[index, 'number_pullback'] = number_pullback
    return df

# Function to create features and labels
def get_data_label(df_ori):
    df = df_ori.copy()

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

    # Iterate through each row
    for i in df_ori.index:
        df = get_feature(i, df)
    return df
# Define a function for reshaping
def reshape(df, window_size=48):
    df_as_array = np.array(df)
    temp = np.array([np.arange(i - window_size, i) for i in range(window_size, df.shape[0])])
    new_df = df_as_array[temp[0:len(temp)]]
    return new_df  

def inference(model, df, index_current, confidence_threshold=0.9):
    scaler = joblib.load('scaler.pkl')
    df = get_feature(index_current, df)
    feature = ['mix_mv_avg', '5_price_diff', 'mv_avg_diff', 'avg_quantity', 
             'quantity_price', 'ct_rising', 'Close', 'trend_name', 
             'durration_trend', 'number_pullback', 'kill_zone']
    data_current = df[feature]
    data_current_scale =  pd.DataFrame(scaler.transform(data_current), columns=feature)
    data_current_scale.columns = feature

    data_current_reshaped = reshape(data_current_scale)
    data_current_reshaped = data_current_reshaped[-1:]

    data_current_tensor = torch.tensor(data_current_reshaped, dtype=torch.float32)
    outputs = model(data_current_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)


def analyze_model(n_steps, n_inputs):
    # Load data
    csv_files_infer = [f for f in os.listdir('test') if f.endswith('.csv')]
    df_raw_infer = pd.concat([pd.read_csv(os.path.join('./test', f)) for f in csv_files_infer], axis=0)
    df_infer = df_raw_infer[100:]
    df_infer = df_infer.loc[~df_infer.index.duplicated(keep='first')].reset_index(drop=True)

    # Load model
    model = TransformerModel(n_steps, n_inputs)
    model_save_path = os.path.join("weights", 'model_transformer.pt')
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    for index in df_infer.index:
        inference(model, df_infer, index)


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

# Add features and labels for train and test
train_2 = get_data_label(train)
test_2 = get_data_label(test)

# Define features
col_for_x = ['mix_mv_avg', '5_price_diff', 'mv_avg_diff', 'avg_quantity', 
             'quantity_price', 'ct_rising', 'Close', 'trend_name', 
             'durration_trend', 'number_pullback', 'kill_zone']

X_train = train_2[col_for_x]
y_train = train_2['label']
X_test = test_2[col_for_x]
y_test = test_2['label']

# Normalize values for train and test
scaler = StandardScaler().fit(X_train)
joblib.dump(scaler, 'scaler_transformer.pkl')

# 2. Chuẩn hóa X_train
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=col_for_x)
X_train_scaled.columns = col_for_x

# 3. Chuẩn hóa X_test bằng cùng scaler
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=col_for_x)
X_test_scaled.columns = col_for_x

# One-hot encode labels
y_train = y_train[n_steps:]
y_test = y_test[n_steps:]
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values  

# Apply the reshape function 
X_train_reshaped = reshape(X_train_scaled)
X_test_reshaped = reshape(X_test_scaled)

# Prepare data (assuming X_train, X_test, y_train, y_test are loaded and reshaped)
X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)

X_val_tensor = X_train_tensor[int(0.8 * len(X_train_tensor)):]
y_val_tensor = y_train_tensor[int(0.8 * len(y_train_tensor)):]

X_train_tensor = X_train_tensor[:int(0.8 * len(X_train_tensor))]
y_train_tensor = y_train_tensor[:int(0.8 * len(y_train_tensor))]

X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)
y_test_tensor = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

# Train the model
folder_model = 'weights'
if not os.path.exists(folder_model):
    os.makedirs(folder_model)
model_save_path = os.path.join(folder_model, 'model_transformer.pt')
# train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, model_save_path)

# Evaluate on test data
model.load_state_dict(torch.load(model_save_path))
model.eval()

with torch.no_grad():
    # Get model outputs
    test_outputs = model(X_test_tensor)
    
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(test_outputs, dim=1)
    
    # Get the predicted labels and their corresponding confidence
    confidences, test_preds = probabilities.max(dim=1)
    
    # Filter predictions with confidence > 0.9
    confident_mask = confidences > 0.5
    confident_preds = test_preds[confident_mask]
    confident_labels = y_test_tensor[confident_mask]
    
    # Calculate accuracy only for confident predictions
    test_accuracy = (confident_preds == confident_labels).float().mean().item()

    print(f'Test Accuracy (Confidence > 0.9): {test_accuracy:.4f}')

    # Crosstab of predictions vs true labels (for confident predictions)
    crosstab_result = pd.crosstab(
        confident_labels.numpy(), confident_preds.numpy(),
        rownames=['True Label'], colnames=['Predicted Label']
    )
    print("Crosstab of True vs Predicted Labels (Confidence > 0.9):")
    print(crosstab_result)
