import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
import os
import ast

# Parameters
n_steps = 50        # Sequence length
n_inputs = 11       # Input features
n_outputs = 3       # Output classes
d_model = 64        # Embedding/Model dimension
n_heads = 8         # Number of attention heads
d_ff = 256          # Feedforward layer size
n_layers = 4        # Number of Transformer layers
learning_rate = 0.0001
n_epochs = 100
batch_size = 32

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
    def __init__(self, n_inputs, n_steps, d_model, n_heads, d_ff, n_layers, n_outputs):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(n_inputs, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        x = self.input_proj(x)  # (batch_size, n_steps, d_model)
        x = self.pos_encoder(x.permute(1, 0, 2))  # (n_steps, batch_size, d_model)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.global_pool(x.permute(1, 2, 0)).squeeze(-1)  # (batch_size, d_model)
        output = self.fc(x)
        return output

# Instantiate the model
model = TransformerModel(n_inputs, n_steps, d_model, n_heads, d_ff, n_layers, n_outputs)
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Helper: Training loop
def train_model(model, X_train, y_train, X_val, y_val, n_epochs, batch_size):
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
            val_accuracy = (val_outputs.argmax(1) == y_val.argmax(1)).float().mean().item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}")
            
# Load training data
csv_files_train = [f for f in os.listdir('data') if f.endswith('.csv')]
df_raw_train = pd.concat([pd.read_csv(os.path.join('./data', f)) for f in csv_files_train], axis=0)
train = df_raw_train[100:]

# Load test data
csv_files_test = [f for f in os.listdir('test') if f.endswith('.csv')]
df_raw_test = pd.concat([pd.read_csv(os.path.join('./test', f)) for f in csv_files_test], axis=0)
test = df_raw_test[100:]

# Function to create features and labels
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

    # Iterate through each row
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

        # Calculate moving averages and other features
        mv_avg_3 = df.Close.loc[max(first_index, i - 2):i + 1].mean()
        mv_avg_7 = df.Close.loc[max(first_index, i - 6):i + 1].mean()
        mv_avg_25 = df.Close.loc[max(first_index, i - 24):i + 1].mean()
        mv_avg_34 = df.Close.loc[max(first_index, i - 33):i + 1].mean()

        df.at[i, 'mix_mv_avg'] = np.mean([mv_avg_3, mv_avg_7, mv_avg_25, mv_avg_34])
        df.at[i, 'mv_avg_diff'] = mv_avg_3 - mv_avg_7

        avg_quantity = df.volume.loc[max(first_index, i - 4):i + 1].mean()
        df.at[i, 'avg_quantity'] = avg_quantity
        df.at[i, 'quantity_price'] = df.volume.loc[i] / df.Close.loc[i]

        df.at[i, 'price_diff'] = df.Close.loc[i] - df.Close.loc[max(first_index, i - 1)]
        df.at[i, '5_price_diff'] = df.Close.loc[i] - df.Close.loc[max(first_index, i - 4)]

        rising_count = (df.Close.loc[max(first_index, i - 9):i + 1].diff().gt(0).sum())
        df.at[i, 'ct_rising'] = rising_count

        if i + 24 < df_length:
            max_high = df.High.loc[i + 1:i + 25].max()
            min_low = df.Low.loc[i + 1:i + 25].min()
            close_price = df.Close.loc[i]

            if (max_high - close_price > 15) and (close_price - min_low < 5):
                df.at[i, 'label'] = 1
            elif (max_high - close_price < 5) and (close_price - min_low > 15):
                df.at[i, 'label'] = 0
            else:
                df.at[i, 'label'] = 2
        else:
            df.at[i, 'label'] = 2

        # Extract technical information
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
X_train_scaled = pd.DataFrame(preprocessing.scale(X_train))
X_train_scaled.columns = col_for_x

X_test_scaled = pd.DataFrame(preprocessing.scale(X_test))
X_test_scaled.columns = col_for_x

# One-hot encode labels
y_train = y_train[:-50]
y_test = y_test[:-50]
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values

# Define a function for reshaping
def reshape(df, window_size=50, n_inputs=11):
    df_as_array = np.array(df)
    temp = np.array([np.arange(i - window_size, i) for i in range(window_size, df.shape[0])])
    new_df = df_as_array[temp[0:len(temp)]]
    new_df2 = new_df.reshape(len(temp), n_inputs * window_size)
    return new_df2    

# Apply the reshape function 
X_train_reshaped = reshape(X_train_scaled)
X_train_reshaped = X_train_reshaped.reshape((X_train_reshaped.shape[0], n_steps, n_inputs))  # Reshape for LSTM
X_test_reshaped = reshape(X_test_scaled)
X_test_reshaped = X_test_reshaped.reshape((X_test_reshaped.shape[0], n_steps, n_inputs))


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
train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, n_epochs, batch_size)

# Evaluate on test data
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = test_outputs.argmax(1)
    test_accuracy = (test_preds == y_test_tensor).float().mean().item()

    print(f'Test Accuracy: {test_accuracy:.4f}')

    # Crosstab of predictions vs true labels
    crosstab_result = pd.crosstab(y_test_tensor.numpy(), test_preds.numpy(),
                                  rownames=['True Label'], colnames=['Predicted Label'])
    print("Crosstab of True vs Predicted Labels:")
    print(crosstab_result)
