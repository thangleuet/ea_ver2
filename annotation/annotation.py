import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

csv_files_train = [f for f in os.listdir('data') if f.endswith('.csv')]
df_raw_train = pd.concat([pd.read_csv(os.path.join('./data', f)) for f in csv_files_train], axis=0)
df_raw_train = df_raw_train[100:]

csv_files_test = [f for f in os.listdir('test') if f.endswith('.csv')]
df_raw_test = pd.concat([pd.read_csv(os.path.join('./test', f)) for f in csv_files_test], axis=0)
df_raw_test = df_raw_test[100:]

def prepare_data(df_raw):
    data_x = []
    data_y = []

    for index in df_raw.index:
        if index < len(df_raw) - 37:
            feature = []
            technical_info = eval(df_raw.iloc[index]['technical_info'])

            # current_trend
            list_trend = technical_info['current']
            if len(list_trend) > 0:
                current_trend = list_trend[0]
                if 'down' in current_trend['trend_name']:
                    trend_name = 0
                elif 'up' in current_trend['trend_name']:
                    trend_name = 1
                else:
                    trend_name = 2
                durration_trend = current_trend['duration_trend']
                slope_trend = current_trend['slope_trend']
                number_pullback = current_trend['number_pullback']

            # resistances
            list_resistances = technical_info['resistances_list']
            if len(list_resistances) > 0:
                resistance = list_resistances[0]
                y_resistance = max(resistance[1], resistance[2])
                number_touch_resistance = resistance[3]
                count_candle_touch_resistance = sum(resistance[4])
            else:
                y_resistance = df_raw.iloc[index]["Close"]
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
                y_support = df_raw.iloc[index]["Close"]
                number_touch_support = 1
                count_candle_touch_support = 1

            # ema_enveloper
            list_ema_enveloper = technical_info['ema_enveloper']
            if len(list_ema_enveloper) > 0:
                ema_enveloper = list_ema_enveloper[0]
                type_ema_enveloper = 0 if 'bear' in ema_enveloper[2] else 1

            feature = [trend_name, durration_trend, number_pullback, y_resistance, 
                        number_touch_resistance, count_candle_touch_resistance, y_support, number_touch_support, 
                        count_candle_touch_support, type_ema_enveloper]
        
            h = []
            l = []
            for i in range(index, index+24):
                h.append(df_raw.iloc[i]["High"])
                l.append(df_raw.iloc[i]["Low"])
            if max(h) - df_raw.iloc[index]["Close"] > 15 and df_raw.iloc[index]["Close"] - min(l) < 5:
                y = 1
            elif max(h) - df_raw.iloc[index]["Close"] < 5 and df_raw.iloc[index]["Close"] - min(l) > 15:
                y = 0
            else:
                y = 2

            data_x.append(feature)
            data_y.append(y)

    return data_x, data_y

X_train, y_train = prepare_data(df_raw_train)
X_test, y_test = prepare_data(df_raw_test)

# Thống kê số lượng class trong tập huấn luyện
unique_train, counts_train = np.unique(y_train, return_counts=True)
train_class_distribution = dict(zip(unique_train, counts_train))

# Thống kê số lượng class trong tập kiểm tra
unique_test, counts_test = np.unique(y_test, return_counts=True)
test_class_distribution = dict(zip(unique_test, counts_test))

print("Số lượng class trong tập huấn luyện:", train_class_distribution)
print("Số lượng class trong tập kiểm tra:", test_class_distribution)

# Chuyển đổi dữ liệu thành định dạng numpy array (nếu chưa)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Fit mô hình Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=123)
rf_clf.fit(X_train, y_train)

# Dự đoán tập kiểm tra
y_pred = rf_clf.predict(X_test)

# In ra bảng crosstab giữa y_test và y_pred
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))

# In ra báo cáo phân loại (classification report)
print(metrics.classification_report(y_test, y_pred))

# In ra độ chính xác của mô hình
print("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))

