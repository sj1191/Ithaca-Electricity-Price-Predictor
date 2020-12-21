# -*- coding: utf-8 -*-
"PriceModel.ipynb"

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import datetime
import numpy as np
from datetime import timedelta
from pandas.tseries.offsets import *
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2005-07-22', end='2020-07-22')
print (len(holidays))

from matplotlib import pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

url = "/content/drive/My Drive/rtd.csv"
df_price = pd.read_csv(url)

url = "/content/drive/My Drive/rtd2.csv"
df_price2 = pd.read_csv(url)

df_price2.head()

df_price = pd.concat([df_price,df_price2])

df_price['RTD End Time Stamp'] = df_price['RTD End Time Stamp'].apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y %H:%M'))

df_price['RTD End Time Stamp'] = pd.to_datetime(df_price['RTD End Time Stamp'])

df_price.tail()

df_price.describe()

df_tm = df_price

df_price['year'] = df_price['RTD End Time Stamp'].apply(lambda x: x.year)
df_price['month'] = df_price['RTD End Time Stamp'].apply(lambda x: x.month)
df_price['day'] = df_price['RTD End Time Stamp'].apply(lambda x: x.day)

df_price.tail()

l = ['year','month','day']
for i in l:
  df_price_time = df_price[list((i,'RTD Zonal LBMP'))]
  df_price_time = df_price_time.groupby(i, as_index=False)['RTD Zonal LBMP'].mean()
  print(df_price_time)
  df_price_time.plot(x = i, y = 'RTD Zonal LBMP')
  plt.show()

df_tmp = df_price

df_price.rename(columns={"RTD End Time Stamp": "localdate"},inplace = True)

df_price['localdate'] = df_price['localdate'].apply(lambda x: x.date())

cols = ['year','month',	'day','localdate']
df_price = df_price.groupby(cols, as_index=False)[list (('RTD Zonal LBMP','RTD Zonal Losses','RTD Zonal Congestion'))].mean()

df_price.isnull().values.any()

df_price

df_price['IsHoliday'] = [1 if x in holidays or x.weekday() == 6 else 0 for x in df_price['localdate']]

df_price['Season'] = df_price['month'].apply(lambda x:(x%12 + 3)//3)

df_price

df_price.to_csv('price5_20.csv')
!cp price5_20.csv "drive/My Drive/"

url = "/content/drive/My Drive/dam.csv"
df_dam = pd.read_csv(url)

df_dam

df_dam = df_dam.loc[df_dam['Zone Name'] == "CENTRL"]

df_dam

df_dam['Eastern Date Hour'] = df_dam['Eastern Date Hour'].apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y %H:%M'))

df_dam

df_dam.rename(columns={"Eastern Date Hour": "localdate"},inplace = True)

df_dam['localdate'] = df_dam['localdate'].apply(lambda x: x.date())

cols = ['localdate']
df_dam = df_dam.groupby(cols, as_index=False)[list (('DAM Zonal LBMP','DAM Zonal Losses','DAM Zonal Congestion'))].mean()

df_dam['localdate'] = df_dam['localdate'].apply(lambda x: x + timedelta(days=1))

df_dam.head()

df_final = pd.merge(df_price,df_dam,on='localdate',how='inner')

df_final

df_final.to_csv('price_final.csv')
!cp price_final.csv "drive/My Drive/"

url = "/content/drive/My Drive/weather.csv"
df_weather = pd.read_csv(url)

df_weather.head()

df_weather = df_weather[list (('DATE','PRCP','SNOW','SNWD','TMAX','TMIN'))]
df_weather['localdate'] = df_weather['DATE'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())

del df_weather['DATE']

df = pd.merge(df_final,df_weather,on='localdate',how='inner')
df

df.isnull().values.any()
df = df.interpolate(method ='linear', limit_direction ='forward') 
df.isnull().values.any()

del df['localdate']
df.to_csv('price_model_data.csv')
!cp price_model_data.csv "drive/My Drive/"

df_tmpp = df

df = df_tmpp

"""# Linear Regression"""

import numpy as np
from sklearn.linear_model import LinearRegression as ln

X = df.drop('RTD Zonal LBMP',axis=1)
y = df['RTD Zonal LBMP']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

reg = ln().fit(X_train, y_train)

reg.score(X_train, y_train)

pred = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test.values, pred))
print(rmse)

from sklearn.metrics import r2_score
r2_score(y_test, pred)

"""## XGBOOST"""

from numpy import loadtxt
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree
from xgboost import plot_importance

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

param = {
    'max_depth': 3,
    'eta' : .1,
    'gamma' : 0,
    'min samples split' : 2,
    'min samples leaf': 1,
    'objective': "reg:squarederror",
    'subsample': 1
}
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
params = {
    'max_depth': [6,8,10],
    'n_estimators': [50, 100, 200, 500, 1000],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
}
print("Parameter optimization")
xgb_model = xgb.XGBRegressor()
clf = GridSearchCV(xgb_model,param_grid=params, verbose=1, cv=kfold, n_jobs=-1)
clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_params_)

clf.best_estimator_

df.rename(columns = lambda x: x.replace(' ', '_'), inplace=True)

ax = plot_importance(clf.best_estimator_, height=1)
fig = ax.figure
fig.set_size_inches(10, 30)
plt.show()

model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=6, min_child_weight=1, missing=None, n_estimators=200,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test.values, predictions))

rmse

"""# Neural Network"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from matplotlib import pyplot

df_tmp = df

X = df.drop('RTD_Zonal_LBMP',axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
y = df['RTD_Zonal_LBMP']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = Sequential()
model.add(Dense(32, input_dim=15,kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs= 2000, verbose=0)
# evaluate the model
train_acc = model.evaluate(X_train, y_train, verbose=0)
test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

pred = model.predict(X_test)

from sklearn.metrics import r2_score
print(np.sqrt(model.evaluate(X_test,y_test)))
r2_score(y_test, pred)

"""# LSTM"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

df.head()

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),(df['RTD Zonal LBMP']))
plt.xticks(range(0,df.shape[0],500),df['localdate'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('RTD ZONAL LBMP',fontsize=18)
plt.show()

del df['localdate']

df

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 16, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

df.columns

scaler = MinMaxScaler(feature_range = (0,1))
df[df.columns] = scaler.fit_transform(df[df.columns])

df

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 1

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train, train['RTD Zonal LBMP'], time_steps)
X_test, y_test = create_dataset(test, test['RTD Zonal LBMP'], time_steps)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

lstm = tf.keras.Sequential()
lstm.add(tf.keras.layers.LSTM(32,return_sequences= True,input_shape = (X_train.shape[1],X_train.shape[2])))
lstm.add(keras.layers.Dense(units=1))
lstm.compile(
  loss='mean_squared_error',
  optimizer=keras.optimizers.Adam(0.001)
)

history = lstm.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.1,
    verbose=1,
    shuffle=False
)

plt.plot(history.history['loss'],label = 'train')
plt.plot(history.history['val_loss'],label = 'test')
plt.legend()
plt.show()

y_pred = lstm.predict(X_test)

train_predict = lstm.predict(X_train).squeeze()
test_predict = lstm.predict(X_test).squeeze()

print('Train Mean Absolute Error:', mean_absolute_error(y_train.squeeze(), train_predict))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(y_train.squeeze(), train_predict)))
print('Test Mean Absolute Error:', mean_absolute_error(y_test.squeeze(), test_predict))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test.squeeze(), test_predict)))

aa=[x for x in range(len(y_pred))]
plt.figure(figsize=(18,9))
plt.plot(aa, y_test.squeeze(), marker='.', label="actual")
plt.plot(aa, test_predict, 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('RTD ZONAL LBMP', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

dataset = df.values
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)
    
look_back = 30
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

lstm = tf.keras.Sequential()
lstm.add(tf.keras.layers.LSTM(32,return_sequences= True,input_shape = (X_train.shape[1],X_train.shape[2])))
lstm.add(keras.layers.Dense(units=1))
lstm.compile(
  loss='mean_squared_error',
  optimizer=keras.optimizers.Adam(0.001)
)

history = model.fit(X_train, Y_train, epochs=30, batch_size=, validation_data=(X_test, Y_test), 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])
print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])
print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();
