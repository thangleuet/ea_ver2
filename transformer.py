import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn import metrics, preprocessing
import ast

# Ensure TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

# Set parameters
n_steps = 50       # Sequence length
n_inputs = 11      # Number of input features
n_outputs = 3      # Number of output classes
d_model = 64       # Model dimension for Transformer
n_heads = 8        # Number of attention heads
d_ff = 256         # Dimension of the feedforward network
n_layers = 4       # Number of Transformer layers
learning_rate = 0.0001
n_epochs = 100

# Helper: Positional Encoding Function
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    # Apply sin to even indices, cos to odd indices
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return tf.constant(pos_encoding, dtype=tf.float32)

# Transformer Encoder Layer
def transformer_encoder_layer(inputs, num_heads, d_model, d_ff):
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    out1 = tf.keras.layers.LayerNormalization()(inputs + attn_output)  # Residual connection

    ff_output = tf.keras.layers.Dense(d_ff, activation='relu')(out1)
    ff_output = tf.keras.layers.Dense(d_model)(ff_output)
    out2 = tf.keras.layers.LayerNormalization()(out1 + ff_output)  # Residual connection

    return out2

# Transformer Model Definition
def build_transformer_model(n_steps, n_inputs, n_outputs, d_model, n_heads, d_ff, n_layers):
    inputs = tf.keras.Input(shape=(n_steps, n_inputs))

    # Add positional encoding
    pos_encoding = positional_encoding(n_steps, d_model)
    inputs_encoded = inputs + pos_encoding

    # Pass through multiple Transformer layers
    x = inputs_encoded
    for _ in range(n_layers):
        x = transformer_encoder_layer(x, n_heads, d_model, d_ff)

    # Global average pooling and output layer
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(n_outputs, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Instantiate the Transformer model
model = build_transformer_model(n_steps, n_inputs, n_outputs, d_model, n_heads, d_ff, n_layers)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

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

x_train_all = X_train_reshaped.reshape((X_train_reshaped.shape[0], n_steps, n_inputs))  # Reshape for LSTM

# Split the train data into train and validation
x_train_input = x_train_all[0:int(x_train_all.shape[0] * 0.8), :]
x_val_input = x_train_all[int(x_train_all.shape[0] * 0.8):, :]
y_train_input = y_train[0:int(y_train.shape[0] * 0.8), :]
y_val_input = y_train[int(y_train.shape[0] * 0.8):, :]

X_test_reshaped = reshape(X_test_scaled)
x_test = X_test_reshaped.reshape((X_test_reshaped.shape[0], n_steps, n_inputs))


# Training loop (assuming X_train_reshaped and y_train are prepared)
x_train_input = X_train_reshaped.reshape((-1, n_steps, n_inputs))
x_val_input = x_train_input[int(0.8 * len(x_train_input)):]
y_train_input = y_train[:int(0.8 * len(y_train))]
y_val_input = y_train[int(0.8 * len(y_train)):]

history = model.fit(x_train_input, y_train_input, epochs=n_epochs, batch_size=32,
                    validation_data=(x_val_input, y_val_input))

# Evaluate on the test set
x_test = X_test_reshaped.reshape((-1, n_steps, n_inputs))
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict and generate crosstab
y_test_pred = model.predict(x_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

crosstab_result = pd.crosstab(y_test_true_classes, y_test_pred_classes,
                              rownames=['True Label'], colnames=['Predicted Label'])

print("Crosstab of True vs Predicted Labels:")
print(crosstab_result)
