

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

import os
import numpy as np
import pandas as pd
import ast


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import preprocessing

csv_files_train = [f for f in os.listdir('data') if f.endswith('.csv')]
df_raw_train = pd.concat([pd.read_csv(os.path.join('./data', f)) for f in csv_files_train], axis=0)
train = df_raw_train[100:]

csv_files_test = [f for f in os.listdir('test') if f.endswith('.csv')]
df_raw_test = pd.concat([pd.read_csv(os.path.join('./test', f)) for f in csv_files_test], axis=0)
test = df_raw_test[100:]

window_size = 50

# define a function that creates features
def create_feature_label(df_ori):
    df = df_ori.copy()

    # Calculate moving averages
    for period in [3, 7, 25, 34]:
        df[f"mv_avg_{period}"] = df.Close.rolling(period, min_periods=1).mean()

    df["mix_mv_avg"] = df[["mv_avg_3", "mv_avg_7", "mv_avg_25", "mv_avg_34"]].mean(axis=1)
    df["mv_avg_diff"] = df.mv_avg_3 - df.mv_avg_7

    df['avg_quantity'] = df.volume.rolling(5, min_periods=1).mean()
    df['quantity_price'] = df.volume / df.Close

    df['price_diff'] = df.Close.diff().bfill()
    df['5_price_diff'] = df.Close.diff(4).bfill()

    df['ct_rising'] = df.price_diff.gt(0).rolling(10, min_periods=1).sum()
    # df['ct_rising'] = df.price_diff.shift(-1).gt(0).rolling(10, min_periods=1).sum()


    # Define label calculation
    high_low_windows = df[['High', 'Low']].rolling(24, min_periods=1)
    max_high = high_low_windows['High'].max()
    min_low = high_low_windows['Low'].min()
    close_price = df['Close']

    df['label'] = np.where(
        (max_high - close_price > 15) & (close_price - min_low < 5), 1,
        np.where((max_high - close_price < 5) & (close_price - min_low > 15), 0, 2)
    )
    df['label'] = df['label'].shift(-24).fillna(2).astype(int)

    # Extract technical information safely
    trend_data = df['technical_info'].apply(lambda x: ast.literal_eval(x).get('current', []))

    df['trend_name'] = trend_data.apply(lambda trends: 0 if trends and 'down' in trends[0]['trend_name']
                                        else 1 if trends and 'up' in trends[0]['trend_name']
                                        else 2)
    df['durration_trend'] = trend_data.apply(lambda trends: trends[0].get('duration_trend', 0) if trends else 0)
    df['number_pullback'] = trend_data.apply(lambda trends: trends[0].get('number_pullback', 0) if trends else 0)

    return df

# add features and labels for train and test
train_2 = create_feature_label(train)
test_2 = create_feature_label(test)

# features we need
#col_for_x = ['price', 'mv_avg_3','mv_avg_6','mv_avg_12','mv_avg_24','mix_mv_avg','5_price_diff']
col_for_x = ['mix_mv_avg','5_price_diff','mv_avg_diff', 'avg_quantity','quantity_price', 'ct_rising', 'Close', 'trend_name', 'durration_trend', 'number_pullback']
#col_for_x = ['mix_mv_avg','5_price_diff','mv_avg_diff', 'avg_quantity','quantity_price','ct_rising','aux_flag']

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
def reshape(df,window_size=50):
    df_as_array=np.array(df)
    temp = np.array([np.arange(i-window_size,i) for i in range(window_size,df.shape[0])])
    new_df = df_as_array[temp[0:len(temp)]]
    new_df2 = new_df.reshape(len(temp),10*window_size)
    return new_df2    

# apply the reshape function 
X_train_reshaped = reshape(X_train_scaled)
X_test_reshaped = reshape(X_test_scaled)

# update the y_train and y_test labels (we don't need the last 50)
y_train = y_train[:-50]
y_test = y_test[:-50]

a = pd.DataFrame(X_train_reshaped)

# ### Random Forest
# build a RF
rf_clf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 123)
rf_clf.fit(X_train_reshaped, y_train)

# Predicting the Test set results
y_pred = rf_clf.predict(X_test_reshaped)

print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
print(metrics.classification_report(y_test,y_pred))
print("Accuracy: " + str(metrics.accuracy_score(y_test,y_pred)))
print("Cohen's Kappa: " + str(metrics.cohen_kappa_score(y_test,y_pred)))