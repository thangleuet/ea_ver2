import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# from models.CNN import CNNModel
from models.Transformer import TransformerModel
from models.random_forest import RandomForest
from models.CNN_1D import CNNModel


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

        close_price = df_raw.loc[index]['Close']
        technical_info = eval(df_raw.loc[index]['technical_info'])
        
        td_seq = technical_info['td_sequential']
        td_seq_number = int(td_seq[0])
        td_seq_trend = 0 if 'down' in td_seq else 1

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
            duration_trend = current_trend['duration_trend']
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

        feature = [trend_name, duration_trend, number_pullback, type_ema_enveloper, td_seq_number, td_seq_trend, close_price, y_resistance, number_touch_resistance, count_candle_touch_resistance, y_support, number_touch_support, count_candle_touch_support]
        list_feature = ['trend_name', 'duration_trend', 'number_pullback', 'type_ema_enveloper', 'td_seq_number', 'td_seq_trend', 'close_price', 'y_resistance', 'number_touch_resistance', 'count_candle_touch_resistance', 'y_support', 'number_touch_support', 'count_candle_touch_support']
        if index + 5 < df_length:
            window_begin = index
            window_end = index + 5

            high_window = df.High.loc[window_begin : window_end]
            low_window = df.Low.loc[window_begin : window_end]

            max_high = high_window.max()  # Max High in next 24 days
            min_low = low_window.min()    # Min Low in next 24 days
            current_price = df.Close.loc[index]

            max_high_index = high_window.idxmax()
            min_low_index = low_window.idxmin()

            max_after = df.High.loc[min_low_index + 1:window_end].max()
            min_after = df.Low.loc[max_high_index + 1:window_end].min()

            # if (max_high - current_price > 7) and (current_price - min_low > 7):
            #     if max_high_index < min_low_index:
            #         min_low_small =  df.Low.loc[index + 1:max_high_index].min()
            #         if current_price - min_low_small < 3:
            #             df.at[index, 'label'] = 1
            #         else:
            #             df.at[index, 'label'] = 2
            #     else:
            #         max_high_small = df.Low.loc[index + 1:min_low_index].max()
            #         if max_high_small - current_price < 3:
            #             df.at[index, 'label'] = 0
            #         else:
            #             df.at[index, 'label'] = 2
            # elif (max_high - current_price > 7) and (current_price - min_low <= 7):
            #     min_low_small =  df.Low.loc[index + 1:max_high_index].min()
            #     if current_price - min_low_small < 3:
            #         df.at[index, 'label'] = 1
            #     else:
            #         df.at[index, 'label'] = 2
            # elif (max_high - current_price <= 7) and (current_price - min_low > 7):
            #     max_high_small = df.Low.loc[index + 1:min_low_index].max()
            #     if max_high_small - current_price < 3:
            #         df.at[index, 'label'] = 0
            #     else:
            #         df.at[index, 'label'] = 2
            # else:
            #     df.at[index, 'label'] = 2

            if max_high_index == index and max_high - min_after > 5:
                df.at[index, 'label'] = 1
            elif min_low_index == index and max_after - min_low > 5:
                df.at[index, 'label'] = 0
            else:
                df.at[index, 'label'] = 2

        else:
            df.at[index, 'label'] = 2

        data_x.append(feature)
        data_y.append(df.at[index, 'label'])

    return data_x, data_y, list_feature

def plot_and_save_metrics(train_losses, test_accuracies, name_model):
    folder_path = 'output'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    output_path = os.path.join(folder_path, f'{name_model}.png')
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    plt.suptitle(name_model)

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, 'g', label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Metrics plot saved at {output_path}")


X_train, y_train, list_feature = prepare_data(df_raw_train)
X_test, y_test, list_feature = prepare_data(df_raw_test)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
model_random_forest = RandomForest(X_train, y_train, X_test, y_test, list_feature)
model_random_forest.train()
model_random_forest.test()

print("=============================================================")

# CNN
model_cnn = CNNModel(X_train, y_train, X_test, y_test, list_feature)
train_losses, test_accuracies = model_cnn.train_model(num_epochs=100)
plot_and_save_metrics(train_losses, test_accuracies, 'cnn')
model_cnn.test_model()

print("=============================================================")

# Transformer
model_transformer = TransformerModel(X_train, y_train, X_test, y_test, list_feature)
train_losses, test_accuracies =model_transformer.train_model(num_epochs=100)
plot_and_save_metrics(train_losses, test_accuracies, 'transformer')
model_transformer.test_model()


