import os
import pandas as pd
import numpy as np
from models.CNN import CNNModel
from models.random_forest import RandomForest

csv_files_train = [f for f in os.listdir('./data') if f.endswith('.csv')]
df_raw_train = pd.concat([pd.read_csv(os.path.join('./data', f)) for f in csv_files_train], axis=0)
df_raw_train = df_raw_train[100:]
df_raw_train = df_raw_train.loc[~df_raw_train.index.duplicated(keep='first')].reset_index(drop=True)

csv_files_test = [f for f in os.listdir('./test') if f.endswith('.csv')]
df_raw_test = pd.concat([pd.read_csv(os.path.join('./test', f)) for f in csv_files_test], axis=0)
df_raw_test = df_raw_test[100:]
df_raw_test = df_raw_test.loc[~df_raw_test.index.duplicated(keep='first')].reset_index(drop=True)


def prepare_data(df_raw):
    data_x = []
    data_y = []
    df = df_raw.copy()
    df_length = len(df)
    first_index = df.index[0]


    for index in df_raw.index:
        feature = []
        technical_info = eval(df_raw.loc[index]['technical_info'])

        # current_trend
        list_trend = technical_info['current']
        if len(list_trend) > 0:
            current_trend = list_trend[-1]
            if 'down' in current_trend['trend_name']:
                trend_name = 0
            elif 'up' in current_trend['trend_name']:
                trend_name = 1
            else:
                trend_name = 2
            durration_trend = current_trend['duration_trend']
            number_pullback = current_trend['number_pullback']

        # resistances
        list_resistances = technical_info['resistances_list']
        if len(list_resistances) > 0:
            resistance = list_resistances[0]
            y_resistance = max(resistance[1], resistance[2])
            number_touch_resistance = resistance[3]
            count_candle_touch_resistance = sum(resistance[4])
        else:
            y_resistance = df_raw.loc[index]["High"]
            number_touch_resistance = 1
            count_candle_touch_resistance = 1

        # supports
        list_supports = technical_info['supports_list']
        if len(list_supports) > 0:
            support = list_supports[0]
            y_support = min(support[1], support[2])
            number_touch_support = support[3]
            count_candle_touch_support = sum(support[4])
        else:
            y_support = df_raw.loc[index]["Low"]
            number_touch_support = 1
            count_candle_touch_support = 1

        # ema_enveloper
        list_ema_enveloper = technical_info['ema_enveloper']
        if len(list_ema_enveloper) > 0:
            ema_enveloper = list_ema_enveloper[-1]
            type_ema_enveloper = 0 if 'bear' in ema_enveloper[2] else 1

        feature = [trend_name, durration_trend, number_pullback, y_resistance, 
                    number_touch_resistance, count_candle_touch_resistance, y_support, number_touch_support, 
                    count_candle_touch_support, type_ema_enveloper]
    
        if index + 24 < df_length:
            window_begin = index
            window_end = index + 24
            high_window = df.High.loc[window_begin : window_end]
            low_window = df.Low.loc[window_begin : window_end]

            max_high = high_window.max()  # Max High in next 24 days
            min_low = low_window.min()    # Min Low in next 24 days
            current_price = df.Close.loc[index]

            max_high_index = high_window.idxmax()
            min_low_index = low_window.idxmin()

            # if (max_high - current_price > 15) and (current_price - min_low < 5):
            #     df.at[index, 'label'] = 1
            # elif (max_high - current_price < 5) and (current_price - min_low > 15):
            #     df.at[index, 'label'] = 0
            if (max_high - current_price > 15) and (current_price - min_low > 15):
                if max_high_index < min_low_index:
                    min_low_small =  df.Low.loc[index + 1:max_high_index].min()
                    if current_price - min_low_small < 5:
                        df.at[index, 'label'] = 1
                    else:
                        df.at[index, 'label'] = 2
                else:
                    max_high_small = df.Low.loc[index + 1:min_low_index].max()
                    if max_high_small - current_price < 5:
                        df.at[index, 'label'] = 0
                    else:
                        df.at[index, 'label'] = 2
            elif (max_high - current_price > 15) and (current_price - min_low <= 15):
                min_low_small =  df.Low.loc[index + 1:max_high_index].min()
                if current_price - min_low_small < 5:
                    df.at[index, 'label'] = 1
                else:
                    df.at[index, 'label'] = 2
            elif (max_high - current_price <= 15) and (current_price - min_low > 15):
                max_high_small = df.Low.loc[index + 1:min_low_index].max()
                if max_high_small - current_price < 5:
                    df.at[index, 'label'] = 0
                else:
                    df.at[index, 'label'] = 2
            else:
                df.at[index, 'label'] = 2
        else:
            df.at[index, 'label'] = 2

        data_x.append(feature)
        data_y.append(df.at[index, 'label'])

    return data_x, data_y

X_train, y_train = prepare_data(df_raw_train)
X_test, y_test = prepare_data(df_raw_test)

unique_train, counts_train = np.unique(y_train, return_counts=True)
train_class_distribution = dict(zip(unique_train, counts_train))

unique_test, counts_test = np.unique(y_test, return_counts=True)
test_class_distribution = dict(zip(unique_test, counts_test))

print("Số lượng class trong tập huấn luyện:", train_class_distribution)
print("Số lượng class trong tập kiểm tra:", test_class_distribution)

# Chuyển đổi dữ liệu thành định dạng numpy array (nếu chưa)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Random Forest
model_random_forest = RandomForest(X_train, y_train, X_test, y_test)
model_random_forest.train()
model_random_forest.test()

# CNN
model_cnn = CNNModel(X_train, y_train, X_test, y_test)
model_cnn.train_model(num_epochs=50)
model_cnn.test_model()


