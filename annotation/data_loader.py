
import numpy as np
import pandas as pd
import os
import ast
from sklearn import preprocessing

class DataCustom():
    def __init__(self, folder_path, n_steps):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        self.df_raw = pd.concat([pd.read_csv(os.path.join(folder_path, f)) for f in csv_files], axis=0)
        self.df_raw = self.df_raw[100:]  # Remove first 100 rows as they are used for feature extraction

        self.col_for_x = ['mix_mv_avg', '5_price_diff', 'mv_avg_diff', 'avg_quantity', 
             'quantity_price', 'ct_rising', 'Close', 'trend_name', 
             'durration_trend', 'number_pullback', 'kill_zone']

        self.x_data, self.y_data = self.load_data()
        self.x_đata = self.normalize_data()
        self.x_data = self.reshape(window_size=n_steps)


    def load_data(self):
        # Implement your data loading logic here
        df = self.df_raw.copy()
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
        first_index = self.df_raw.index[0]

        # Iterate through each row
        for i in self.df_raw.index:
            
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
                if (max_high - close_price > 15) and (close_price - min_low < 5):
                    df.at[i, 'label'] = 1
                elif (max_high - close_price < 5) and (close_price - min_low > 15):
                    df.at[i, 'label'] = 0
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
                    df.at[i, 'label'] = 2

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

            df.at[i, 'trend_name'] = trend_name
            df.at[i, 'durration_trend'] = durration_trend
            df.at[i, 'number_pullback'] = number_pullback

        x = df[self.col_for_x]
        y = df['label']
        return x, y

    def normalize_data(self):
        x_norm = pd.DataFrame(preprocessing.scale(self.x_data))
        x_norm.columns = self.col_for_x
        return x_norm
    
    # Define a function for reshaping
    def reshape(self, window_size=50):
        n_inputs = len(self.col_for_x)
        df_as_array = np.array(self.x_data)
        temp = np.array([np.arange(i - window_size, i) for i in range(window_size, self.x_data.shape[0])])
        new_df = df_as_array[temp[0:len(temp)]]
        new_df2 = new_df.reshape(len(temp), n_inputs * window_size)

        # y
        self.y_data = self.y_data[:-window_size]

        return new_df2    