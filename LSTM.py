import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

import os
import numpy as np
import pandas as pd
import ast

from sklearn import preprocessing
import tensorflow as tf
import time
from sklearn import metrics

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set parameters
n_steps = 50     # 50 time steps
n_inputs = 11    # Size of the input vector
n_neurons = 256  # Number of neurons in LSTM
n_outputs = 3    # Number of output classes

learning_rate = 0.0001
batch_size = 128  # Adjust batch size as necessary
n_epochs = 200

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

# Reset the default graph
tf.reset_default_graph()

# Define placeholders for inputs and outputs
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="X")
    y = tf.placeholder(tf.int32, [None, n_outputs], name="y")

# Define the LSTM model
with tf.name_scope("RNN"):
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)

# Define the output layer
with tf.name_scope("output"):
    logits = tf.layers.dense(states.h, n_outputs, name="logits")  # Use states.h for output
    Y_prob = tf.nn.softmax(logits, name="Y_prob")

# Define the loss and optimizer
with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('loss', loss)

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

# Define accuracy metrics
with tf.name_scope("accuracy"):
    correct = tf.equal(tf.argmax(Y_prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    acc_summary = tf.summary.scalar('accuracy', accuracy)

# Define a summary writer
logdir = "./tf_logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

summary_writer = tf.summary.FileWriter(logdir)

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        X_batch, y_batch = x_train_input, y_train_input
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 10 == 0:
            train_loss, train_accuracy = sess.run([loss, accuracy], feed_dict={X: x_val_input, y: y_val_input})
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # Evaluate on the validation set
    val_loss, val_accuracy = sess.run([loss, accuracy], feed_dict={X: x_val_input, y: y_val_input})
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Test the model
    y_test_pred = sess.run(Y_prob, feed_dict={X: x_test})
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    y_test_true_classes = np.argmax(y_test.reshape(-1, n_outputs), axis=1)

    # Calculate accuracy for test set
    test_accuracy = metrics.accuracy_score(y_test_true_classes, y_test_pred_classes)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    crosstab_result = pd.crosstab(y_test_true_classes, y_test_pred_classes, 
                               rownames=['True Label'], colnames=['Predicted Label'])

    print("Crosstab of True vs Predicted Labels:")
    print(crosstab_result)