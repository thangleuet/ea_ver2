import warnings

from annotation.data_loader import DataCustom
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

import os
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn import metrics

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set parameters
n_steps = 50     # 50 time steps
n_neurons = 256  # Number of neurons in LSTM
n_outputs = 3    # Number of output classes

learning_rate = 0.0001
batch_size = 128  # Adjust batch size as necessary
n_epochs = 200

# Load training data
data_train = DataCustom('data', n_steps)

# Load test data
data_test = DataCustom('test', n_steps)

# Add features and labels for train and test
X_train, y_train = data_train.x_data, data_train.y_data
X_test, y_test = data_test.x_data, data_test.y_data

n_inputs = len(data_train.col_for_x)

# Apply the reshape function 
x_train_all = X_train.reshape((X_train.shape[0], n_steps, n_inputs))  # Reshape for LSTM
y_train = pd.get_dummies(y_train).values # transform the label into one hot encoding
y_test = pd.get_dummies(y_test).values

# Split the train data into train and validation
x_train_input = x_train_all[0:int(x_train_all.shape[0] * 0.8), :]
x_val_input = x_train_all[int(x_train_all.shape[0] * 0.8):, :]
y_train_input = y_train[0:int(y_train.shape[0] * 0.8), :]
y_val_input = y_train[int(y_train.shape[0] * 0.8):, :]

x_test = X_test.reshape((X_test.shape[0], n_steps, n_inputs))

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
saver = tf.train.Saver()
model_path = "best_model_lstm.ckpt"
# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    best_val_accuracy = 0.0 
    for epoch in range(n_epochs):
        X_batch, y_batch = x_train_input, y_train_input
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        train_loss, train_accuracy = sess.run([loss, accuracy], feed_dict={X: x_val_input, y: y_val_input})
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        if train_accuracy > best_val_accuracy:
            best_val_accuracy = train_accuracy
            saver.save(sess, model_path)
            print(f"Model saved at epoch {epoch} with Val Accuracy: {best_val_accuracy:.4f}")

    # Evaluate on the validation set
    val_loss, val_accuracy = sess.run([loss, accuracy], feed_dict={X: x_val_input, y: y_val_input})
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Test the model
    with tf.Session() as sess:
        # Restore the saved model
        saver.restore(sess, model_path)
        print("Model restored for testing.")

        # Test the model
        y_test_pred = sess.run(Y_prob, feed_dict={X: x_test})
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        y_test_true_classes = np.argmax(y_test, axis=1)

        # Calculate accuracy for the test set
        test_accuracy = metrics.accuracy_score(y_test_true_classes, y_test_pred_classes)
        print(f'Test Accuracy: {test_accuracy:.4f}')

        # Print crosstab of predictions
        crosstab_result = pd.crosstab(y_test_true_classes, y_test_pred_classes, 
                                    rownames=['True Label'], colnames=['Predicted Label'])
        print("Crosstab of True vs Predicted Labels:")
        print(crosstab_result)