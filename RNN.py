

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
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

n_steps = 50     # 50 time steps, each step corresponds to 1*8 (a row) of a image
n_inputs = 12    # the size of the input vector
n_neurons = 256  # recurrent neurons/The number of units in the RNN cell #150
n_outputs = 3   # number of neurons/units of the fully connected layer

learning_rate = 0.0001
batch_size = 128 #tried 128, worse than 50
n_epochs = 100

iteration = 0

csv_files_train = [f for f in os.listdir('data') if f.endswith('.csv')]
df_raw_train = pd.concat([pd.read_csv(os.path.join('./data', f)) for f in csv_files_train], axis=0)
train = df_raw_train[100:]

csv_files_test = [f for f in os.listdir('test') if f.endswith('.csv')]
df_raw_test = pd.concat([pd.read_csv(os.path.join('./test', f)) for f in csv_files_test], axis=0)
test = df_raw_test[100:]

window_size = 50

def calculate_rsi(close_prices, window=14):
    # Calculate price differences
    delta = close_prices.diff()

    # Separate gains and losses
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # Calculate rolling averages of gains and losses
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    # Calculate RS (Relative Strength) and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# define a function that creates features
def create_feature_label(df_ori):
    df = df_ori.copy()
    df_length = len(df)

    # Initialize new columns with NaN or default values
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

    # Iterate through each row by index
    for i in df_ori.index:

        # kill zone
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

        # Calculate rolling values safely for the current row's window
        mv_avg_3 = df.Close.loc[max(first_index, i - 2):i + 1].mean()
        mv_avg_7 = df.Close.loc[max(first_index, i - 6):i + 1].mean()
        mv_avg_25 = df.Close.loc[max(first_index, i - 24):i + 1].mean()
        mv_avg_34 = df.Close.loc[max(first_index, i - 33):i + 1].mean()

        # Assign moving average calculations
        df.at[i, 'mix_mv_avg'] = np.mean([mv_avg_3, mv_avg_7, mv_avg_25, mv_avg_34])
        df.at[i, 'mv_avg_diff'] = mv_avg_3 - mv_avg_7

        # Calculate quantity features
        avg_quantity = df.volume.loc[max(first_index, i - 4):i + 1].mean()
        df.at[i, 'avg_quantity'] = avg_quantity
        df.at[i, 'quantity_price'] = df.volume.loc[i] / df.Close.loc[i]

        # Calculate price differences
        df.at[i, 'price_diff'] = df.Close.loc[i] - df.Close.loc[max(first_index, i - 1)]
        df.at[i, '5_price_diff'] = df.Close.loc[i] - df.Close.loc[max(first_index, i - 4)]

        # Count rising days
        rising_count = (df.Close.loc[max(first_index, i - 9):i + 1].diff().gt(0).sum())
        df.at[i, 'ct_rising'] = rising_count

        close_prices = df.Close.loc[max(first_index, i - 13):i + 1]
        rsi_value = calculate_rsi(close_prices).iloc[-1]  # Latest RSI value
        df.at[i, 'RSI'] = rsi_value

        if i + 24 < df_length:  # Đảm bảo không vượt quá giới hạn DataFrame
            high_window = df.High.loc[i + 1:i + 25]
            low_window = df.Low.loc[i + 1:i + 25]

            max_high = high_window.max()  # Max High in next 24 days
            min_low = low_window.min()    # Min Low in next 24 days

            # Get the index of max high and min low
            max_high_index = high_window.idxmax()
            min_low_index = low_window.idxmin()

            close_price = df.Close.loc[i]                # Giá đóng cửa hiện tại

            # Gán nhãn dựa vào điều kiện
            if ((max_high - close_price)/close_price > 0.0079) and ((close_price - min_low)/close_price < 0.0025):
                df.at[i, 'label'] = 1
            elif ((max_high - close_price)/close_price < 0.0025) and ((close_price - min_low)/close_price > 0.0079):
                df.at[i, 'label'] = 0
            elif (max_high - close_price > 15) and (close_price - min_low > 15):
                if max_high_index < min_low_index:
                    min_low_small =  df.Low.loc[i + 1:max_high_index].min()
                    if close_price - min_low_small < 5:
                        df.at[i, 'label'] = 1
                    else:
                        df.at[i, 'label'] = 2
                else:
                    max_high_small = df.Low.loc[i + 1:min_low_index].max()
                    if max_high_small - close_price < 5:
                        df.at[i, 'label'] = 0
                    else:
                        df.at[i, 'label'] = 2
            else:
                df.at[i, 'label'] = 2

            # if (max_high - close_price > 15) and (close_price - min_low < 15):
            #     df.at[i, 'label'] = 1
            # elif (max_high - close_price < 15) and (close_price - min_low > 15):
            #     df.at[i, 'label'] = 0
            # elif (max_high - close_price > 15) and (close_price - min_low > 15):
            #     if max_high_index > min_low_index:
            #         df.at[i, 'label'] = 1
            #     else:
            #         df.at[i, 'label'] = 0
            # else:
            #     df.at[i, 'label'] = 2
        else:
            df.at[i, 'label'] = 2  # Nếu không đủ dữ liệu, gán giá trị mặc định là 2
            
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

        # Assign extracted trend information
        df.at[i, 'trend_name'] = trend_name
        df.at[i, 'durration_trend'] = durration_trend
        df.at[i, 'number_pullback'] = number_pullback

    return df

# add features and labels for train and test
train_2 = create_feature_label(train)
test_2 = create_feature_label(test)

# features we need
col_for_x = ['mix_mv_avg','5_price_diff','mv_avg_diff', 'avg_quantity','quantity_price', 'ct_rising', 'Close', 'trend_name', 'durration_trend', 'number_pullback', 'kill_zone', 'RSI']

X_train = train_2[col_for_x]
y_train = train_2['label']
X_test = test_2[col_for_x]
y_test = test_2['label']

# normalize values for train and test
X_train_scaled = pd.DataFrame(preprocessing.scale(X_train))
X_train_scaled.columns = col_for_x

X_test_scaled = pd.DataFrame(preprocessing.scale(X_test))
X_test_scaled.columns = col_for_x

y_train.columns = 'label'
y_test.columns = 'label'


# define a function for reshaping
def reshape(df,window_size, n_inputs=12):
    df_as_array=np.array(df)
    temp = np.array([np.arange(i-window_size,i) for i in range(window_size,df.shape[0])])
    new_df = df_as_array[temp[0:len(temp)]]
    new_df2 = new_df.reshape(len(temp),n_inputs*window_size)
    return new_df2    

# apply the reshape function 
X_train_reshaped = reshape(X_train_scaled, n_steps)

x_train_all = X_train_reshaped.reshape((X_train_reshaped.shape[0],n_steps,n_inputs)) # reshape for RNN
y_train = y_train[:-n_steps]
y_train_all = np.array(pd.get_dummies(y_train)) # transform the label into one hot encoding

# split the train data into train and validation not random
x_train_input = x_train_all[0:int(x_train_all.shape[0]*0.8), :]
x_val_input = x_train_all[int(x_train_all.shape[0]*0.8):, :]
y_train_input = y_train_all[0:int(y_train_all.shape[0]*0.8), :]
y_val_input = y_train_all[int(y_train_all.shape[0]*0.8):, :]


X_test_reshaped = reshape(X_test_scaled, n_steps)
x_test = X_test_reshaped.reshape((X_test_reshaped.shape[0],n_steps,n_inputs))
y_test = y_test[:-n_steps]

tf.reset_default_graph()

keep_prob = tf.placeholder_with_default(1.0, shape=())

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="X")
    y = tf.placeholder(tf.int32, [None, n_outputs], name="y")

with tf.name_scope("RNN"):
    gru_cells = tf.contrib.rnn.GRUCell(num_units=n_neurons) #
    outputs, states = tf.nn.dynamic_rnn(gru_cells, X, dtype=tf.float32)

with tf.name_scope("output"):
    logits = tf.layers.dense(states, n_outputs, name="logits")
    Y_prob = tf.nn.softmax(logits, name="Y_prob")

# Define the optimizer; taking as input (learning_rate) and (loss)
with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate) # GradientDescentOptimizer #MomentumOptimizer , momentum=0.9
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correctPrediction = tf.equal(tf.argmax(Y_prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def batch_func(X, y, batch_size):
    batches = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i: i+batch_size]
        y_batch = y[i: i+batch_size]
        mini_batch = (X_batch, y_batch)
        batches.append(mini_batch)
    return batches

# -------------------------- train model --------------------------------------

start = time.time()
best_acc = 0
with tf.Session() as sess:  # train model (train set and validation set)
    init.run()
    for epoch in range(n_epochs):

        for a_batch in batch_func(x_train_input, y_train_input, batch_size):
            iteration += 1
            (X_batch, y_batch) = a_batch
            sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
       
        # train accuracy and loss
        train_loss = sess.run(loss, feed_dict = {X:X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})

        # validation set loss and accuracy
        val_loss = sess.run(loss, feed_dict = {X:x_val_input, y:y_val_input})
        acc_val = accuracy.eval(feed_dict = {X:x_val_input, y:y_val_input})

        # print result for each epoch
        print("epoch {}: ".format(epoch+1),
              "Train loss: {}".format(train_loss),
              "Train accuracy: {}".format(acc_train),
              "Validation loss: {}".format(val_loss),
              "Validation accuracy: {}".format(acc_val))
        if acc_val >= best_acc:
            best_acc = acc_val
            print("Save model")
            save_path = saver.save(sess, "./rnn_model.ckpt")

print('Took: %f seconds' %(time.time() - start))

# ----------------------- make predictions ------------------------------------

# use model to predict on test set
with tf.Session() as sess:
    saver.restore(sess, "./rnn_model.ckpt") # load model parameters from disk
    Z = Y_prob.eval(feed_dict = {X: x_test})
    y_pred = np.argmax(Z, axis = 1)

# print results
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
print(metrics.classification_report(y_test,y_pred))
print("Accuracy: " + str(metrics.accuracy_score(y_test,y_pred)))
print("Cohen's Kappa: " + str(metrics.cohen_kappa_score(y_test,y_pred)))