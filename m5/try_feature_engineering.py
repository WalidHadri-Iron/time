import pandas as pd
import numpy as np
import os
import datetime as dt
from sklearn import preprocessing
#Reading
train = pd.read_csv(os.path.join(path, 'sales_train_evaluation.csv'))
calendar = pd.read_csv(os.path.join(path, 'calendar.csv'))
sell_prices = pd.read_csv(os.path.join(path, 'sell_prices.csv'))
pd.set_option('mode.chained_assignment', None)

#Reduce Data to 200 days
df_train = train.iloc[: , :206]
df_calendar=calendar.iloc[:200]
indices = df_calendar["wm_yr_wk"].astype(np.int64)
df_sell_prices = sell_prices[sell_prices['wm_yr_wk'].isin(indices)]

# Creating features from calendar (date features)
# d_n ---> n
df_calendar['d'] = df_calendar['d'].str[2:].astype(np.int16)
# transforming the column to datetime type
df_calendar['datetime'] = pd.to_datetime(df_calendar['date'])
# Real order of the day in the week
df_calendar['day_of_week'] = df_calendar['datetime'].dt.dayofweek.astype(np.int64)
# Type chancing to int
df_calendar['month'] = df_calendar['datetime'].dt.month.astype(np.int64)
df_calendar['wm_yr_wk'] = df_calendar['wm_yr_wk'].astype(np.int64)

# Label encoding
enc_dict = {}
for col in ['item_id', 'store_id', 'dept_id']:
    enc = preprocessing.LabelEncoder()
    original_labels = df_train[col].values
    if col in df_sell_prices.columns:
        original_labels = np.concatenate([original_labels, df_sell_prices[col].values])
    enc.fit(original_labels)
    df_train[col] = enc.transform(train[col]).astype(np.int16)
    if col in df_sell_prices.columns:
        df_sell_prices[col] = enc.transform(df_sell_prices[col]).astype(np.int16)
    enc_dict[col] = enc
df_train.drop(['id', 'cat_id'], inplace=True, axis=1)

df_train = pd.melt(df_train, id_vars=['item_id', 'dept_id', 'store_id', 'state_id'],var_name='d')
df_train['value'] = df_train['value'].astype(np.float64)
df_train['d'] = df_train['d'].str[2:].astype(np.int64)
df_train.set_index(['item_id', 'dept_id', 'store_id', 'd'], inplace=True, drop=True)
df_train['dept_id_col'] = df_train.index.get_level_values('dept_id').astype(np.int64)
df_train['store_id_col'] = df_train.index.get_level_values('store_id').astype(np.int64)

#Merge df_calendar with df_train
df_calendar.set_index('d', inplace=True, drop=True)
df_train = df_train.join(df_calendar[['day_of_week',  'month', 'wm_yr_wk', 'snap_WI', 'snap_CA', 'snap_TX']], how='left')
df_train['snap'] = 0
df_train['snap'] = df_train['snap'].astype(np.int8)
for state_id in ['CA', 'TX', 'WI']:
    df_train['snap'] += (df_train['snap_' + state_id] == 1) & (df_train['state_id'] == state_id)
df_train.drop(['state_id', 'snap_WI', 'snap_CA', 'snap_TX'], inplace=True, axis=1)
df_train.sort_index(inplace=True)
