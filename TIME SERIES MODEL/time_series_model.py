"""
Original file is located at
    https://colab.research.google.com/drive/1m8ZhZfxoBCPwPBi4cV-Mb9KF-H96v8xe
"""

from google.colab import drive
drive.mount('/content/drive')

from google.colab import drive
drive.mount('/content/drive')



import numpy as np
import pandas as pd

train=pd.read_csv('/content/drive/MyDrive/datasets/train.csv')

test=pd.read_csv('/content/drive/MyDrive/datasets/test.csv')

meal=pd.read_csv('/content/drive/MyDrive/datasets/meal_info.csv')

train=train.merge(meal, on='meal_id')

train.head()

train['meal_id'].value_counts()

center_id = 55
meal_id = 1885

train_df = train[train['center_id']==center_id]
train_df = train_df[train_df['meal_id']==meal_id]

train_df

#preprocessing task for time series analysis
def pretime():

  df=train_df
  period = len(train_df)
  train_df['Date'] = pd.date_range('2015-01-08', periods=period, freq='W')
  train_df['Day'] = train_df['Date'].dt.day
  train_df['Month'] = train_df['Date'].dt.month
  train_df['Year'] = train_df['Date'].dt.year
  train_df['Quarter'] = train_df['Date'].dt.quarter

pretime()

"""**xb regressor**"""

def prexbreg(train_df):
  xb_data = train_df.drop(columns=['id','center_id','meal_id','category','cuisine'])
  xb_data = xb_data.set_index(['Date'])
  return xb_data

xb_data=prexbreg(train_df)

xb_data

"""**light model for xb regressor**"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
def lighttrain(train_df):

  from xgboost import XGBRegressor
  period=len(train_df)
  x_train = xb_data.drop(columns='num_orders')
  y_train = xb_data['num_orders']
  y_train = np.log1p(y_train)
  split_size = period-15
  X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=15, shuffle=False)
  model_0 = XGBRegressor(
    learning_rate=0.2,
    n_estimators=500,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.6,
    gamma=2.0
  )

  model_0.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], early_stopping_rounds=100, verbose=100)
  best_iteration = model_0.get_booster().best_iteration

  xgb_model = XGBRegressor(
    learning_rate=0.2,
    n_estimators=best_iteration,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.6,
    gamma=2.0
  )

  xgb_model.fit(X_train, Y_train)
  xgb_preds = xgb_model.predict(X_test)
  xgb_preds = np.exp(xgb_preds)
  xgb_preds = pd.DataFrame(xgb_preds)
  xgb_preds.index = Y_test.index
  Y_train = np.exp(Y_train)
  Y_test = np.exp(Y_test)

  mse = mean_squared_error(Y_test, xgb_preds, squared=False)
  print("Light XGB MSE=", mse)

lighttrain(train_df)

"""**medium model of xb**"""

xb_data.head()

def mediumtrain(train_df):
  from xgboost import XGBRegressor
  period=len(train_df)
  x_train = xb_data.drop(columns='num_orders')
  y_train = xb_data['num_orders']
  y_train = np.log1p(y_train)
  split_size = period-15
  X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=15, shuffle=False)
  model_1 = XGBRegressor(
      learning_rate=0.1,
      n_estimators=1000,
      max_depth=3,
      subsample=0.5,
      colsample_bytree=0.8,
      gamma=1.0
  )
  model_1.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], early_stopping_rounds=500, verbose=100)
  a = (model_1.get_booster().best_iteration)
  xgb_model_1 = XGBRegressor(
  learning_rate = 0.1,
  n_estimators = a,
  max_depth = 3,
  subsample = 0.5,
  colsample_bytree = 0.8,
  gamma = 1.0)

  xgb_model_1.fit(X_train, Y_train)
  xgb_preds_1 = xgb_model_1.predict(X_test)
  xgb_preds_1 = np.exp(xgb_preds_1)
  xgb_preds_1 = pd.DataFrame(xgb_preds_1)
  xgb_preds_1.index = Y_test.index
  Y_train = np.exp(Y_train)
  Y_test = np.exp(Y_test)
  from sklearn.metrics import mean_squared_error
  print("Medium XGB MSE=", mean_squared_error(Y_test, xgb_preds_1, squared=False))

mediumtrain(train_df)

"""**heavy model for xb**"""

def heavytrain(train_df):
  period=len(train_df)
  x_train = xb_data.drop(columns='num_orders')
  y_train = xb_data['num_orders']
  y_train = np.log1p(y_train)
  split_size = period-15
  X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=15, shuffle=False)
  from xgboost import XGBRegressor
  model_2 = XGBRegressor(
  learning_rate = 0.01,
  eval_metric ='rmse',
  n_estimators = 50000,
  max_depth = 5,
  subsample = 0.8,
  colsample_bytree = 1,
  gamma = 0.5
  )
  #model.fit(X_train, y_train)
  model_2.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], early_stopping_rounds=500, verbose=100)

  # Retrieve best iteration
  a = (model_2.get_booster().best_iteration)

  # Building and training XGBoost regression model
  xgb_model = XGBRegressor(
  learning_rate = 0.01,
  n_estimators = a,
  max_depth = 5,
  subsample = 0.8,
  colsample_bytree = 1,
  gamma = 0.5)
  xgb_model.fit(X_train, Y_train)
  xgb_preds = xgb_model.predict(X_test)
  xgb_preds = np.exp(xgb_preds)
  xgb_preds = pd.DataFrame(xgb_preds)
  xgb_preds.index = Y_test.index
  Y_train = np.exp(Y_train)
  Y_test = np.exp(Y_test)

  from sklearn.metrics import mean_squared_error
  print("MSE=", mean_squared_error(Y_test, xgb_preds, squared=False))

heavytrain(train_df)

"""**FB PROPHET**"""

#preprocessing for fb prophet
period=len(train_df)
split_size = period-15
def preprophet(train_df):
  period = len(train_df)
  train_df['Date'] = pd.date_range('2015-01-08', periods=period, freq='W')
  xb_data = train_df.drop(columns=['id','center_id','meal_id','category','cuisine'])
  xb_data = xb_data.set_index(['Date'])
  prophet_data = train_df[['Date','num_orders']]
  prophet_data.index = xb_data.index
  prophet_data = prophet_data.iloc[:split_size,:]
  prophet_data =prophet_data.rename(columns={'Date':'ds',
  'num_orders':'y'})
  print(prophet_data.head())
  return prophet_data

prophet_data=preprophet(train_df)

prophet_data.shape

"""**heavy tunning of FBPROPHET**"""

period=len(train_df)
split_size=period-7
def preprophet(train_df):
    global prophet_data
    train_df['Date'] = pd.date_range('2015-01-08', periods=period, freq='W')
    prophet_data = train_df[['Date', 'num_orders']]
    xb_data = train_df.drop(columns=['id','center_id','meal_id','category','cuisine'])
    xb_data = xb_data.set_index(['Date'])
    prophet_data.index = xb_data.index

    prophet_data = prophet_data.iloc[:split_size, :]
    prophet_data = prophet_data.rename(columns={'Date': 'ds', 'num_orders': 'y'})


preprophet(train_df)
print(prophet_data.head(10))
def fblight(train_df):
    period=len(train_df)
    xb_data = train_df.drop(columns=['id','center_id','meal_id','category','cuisine'])
    xb_data = xb_data.set_index(['Date'])
    x_train = xb_data.drop(columns='num_orders')
    y_train = xb_data['num_orders']
    y_train = np.log1p(y_train)
    from sklearn.model_selection import train_test_split
    split_size = period-7
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=7, shuffle=False)
    from prophet import Prophet

    m = Prophet(
        growth='linear',
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        # Adjust other parameters as needed
    )

    # Let Prophet automatically detect seasonality
    m.add_seasonality(name='weekly', period=7, fourier_order=15)
    m.add_seasonality(name='yearly', period=365.25, fourier_order=10)

    # Add holidays if needed
    m.add_country_holidays(country_name='US')
    print(prophet_data)
    m.fit(prophet_data)

    future = m.make_future_dataframe(periods=7, freq='D')  # Adjust frequency if needed
    forecast = m.predict(future)
    prophet_preds = forecast['yhat'].iloc[split_size:]
    prophet_preds.index = Y_test.index

    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(Y_test, prophet_preds, squared=False))

fblight(train_df)

"""**medium tunning of FBProphet**"""

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from prophet import Prophet

def fblight(params,train_df):
    period=len(train_df)
    x_train = xb_data.drop(columns='num_orders')
    y_train = xb_data['num_orders']
    y_train = np.log1p(y_train)
    split_size = period-15
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=15, shuffle=False)
    m = Prophet(
        growth='linear',
        seasonality_mode=params['seasonality_mode'],
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        holidays_prior_scale=params['holidays_prior_scale'],
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=False
    ).add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=30
    ).add_seasonality(
        name='weekly',
        period=7,
        fourier_order=55
    ).add_seasonality(
        name='yearly',
        period=365.25,
        fourier_order=20
    )

    m.fit(prophet_data)
    future = m.make_future_dataframe(periods=15, freq='W')
    forecast = m.predict(future)
    prophet_preds = forecast['yhat'].iloc[split_size:]
    prophet_preds.index = Y_test.index

    mse = mean_squared_error(Y_test, prophet_preds, squared=False)

    print(f'Mean Squared Error for params {params}: {mse}')

    return params, mse

# Define the parameter grid
params_grid = {'seasonality_mode': ['multiplicative', 'additive'],
               'changepoint_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
               'holidays_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
               'seasonality_prior_scale': [10, 20, 30, 40, 50],
               'n_changepoints': [100, 150, 200]}

grid = ParameterGrid(params_grid)

best_params = None
best_mse = float('inf')

# Iterate through parameter combinations
for params in grid:
    _, mse = fblight(params,train_df)  #calling functions
    if mse < best_mse:
        best_mse = mse
        best_params = params

print(f'Best Parameters: {best_params}')
print(f'Best Mean Squared Error: {best_mse}')

"""**light tunning of Fbprophet**"""

def fblig(train_df):
  period=len(train_df)
  x_train = xb_data.drop(columns='num_orders')
  y_train = xb_data['num_orders']
  y_train = np.log1p(y_train)
  split_size = period-15
  X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=15, shuffle=False)
  from prophet import Prophet
  m = Prophet(
    growth='linear',
    #interval_width=0.80,
    seasonality_mode= 'multiplicative',
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False
    ).add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=50,#25
        prior_scale=20
    ).add_seasonality(
        name='daily',
        period=1,
        fourier_order=70,#25
        prior_scale=20
    ).add_seasonality(
        name='weekly',
        period=7,
        fourier_order=50,
        prior_scale=60
    ).add_seasonality(
        name='yearly',
        period=365.25,
        fourier_order= 30)
  m.fit(prophet_data)
  future = m.make_future_dataframe(periods=15, freq='W')
  forecast = m.predict(future)
  forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
  prophet_preds = forecast['yhat'].iloc[split_size:]
  prophet_preds.index = Y_test.index
  from sklearn.metrics import mean_squared_error
  print(mean_squared_error(Y_test, prophet_preds, squared=False))
fblig(train_df)

"""**arima**"""

#installing pmdarima
!pip install pmdarima

import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def prearima(train_df):
    period = len(train_df)
    data = train_df.copy()
    data['Date'] = pd.date_range('2015-01-08', periods=period, freq='W')
    df_ari = data[['Date', 'num_orders']]
    df_ari = df_ari.set_index(['Date'])

    df_log = np.log(df_ari)
    df_double_log = np.log(df_log)
    ts_log_diff = df_log - df_log.shift()
    ts_log_diff.dropna(inplace=True)
    return df_ari, df_double_log

def trainarim(df_ari, ts_log_diff):
    ts_values = ts_log_diff.values
    X = ts_values
    size = int(len(X) * 0.667)
    train, test = X[0:size], X[size:len(X)]

    # Auto ARIMA
    arima_model = auto_arima(train, suppress_warnings=True, seasonal=True)
    arima_model.summary()

    # ARIMA
    history = [x for x in train]
    predictions = list()

    for t in range(len(test)):
        try:
            model = ARIMA(history, order=(0, 1, 1))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]

            # Inverse transformations
            yhat_inverse = np.exp(np.exp(yhat))  # Removed np.cumsum(history)[-1]

            predictions.append(yhat_inverse)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat_inverse, np.exp(np.exp(obs))))
        except (ValueError, np.linalg.LinAlgError):
            pass

    # Calculate Mean Squared Error
    rmse = mean_squared_error(np.exp(np.exp(test)), predictions)
    print('Mean Squared Error (MSE): %.2f' % mse)

# Assuming train_df is your initial DataFrame
df_ari, ts_log_diff = prearima(train_df)
trainarim(df_ari, ts_log_diff)





"""**for future**"""

import seaborn as sns
data = train[train['center_id']==55]
data = data[data['meal_id']==1885]
period = len(data)
data['Date'] = pd.date_range('2015-01-08', periods=period, freq='W')

df_ari = data[['Date','num_orders']]
df_ari = df_ari.set_index(['Date'])
df_ari.head()

df_ari

from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
def test(data):
  rolmean = data.rolling(window=2).mean()
  rolstd = data.rolling(window=2).std()
  plt.figure(figsize=(25,5))
  plt.plot(data, color='blue', label='original cases')
  plt.plot(rolmean, color='red', label='rolling mean')
  plt.plot(rolstd, color='black', label='rolling standard deviation')
  plt.legend(loc='best')
  plt.show()
  dftest = adfuller(data['num_orders'], autolag = 't-stat')
  dfoutput = pd.Series(dftest[0:4], index=['test statitics','p_value','lags used','number of observations'])
  for key,value in dftest[4].items():
    dfoutput['critcal value (%s)'%key] = value
  print(dfoutput)
test(df_ari)

import numpy as np
df_log = np.log(df_ari)
plt.plot(df_log)

test(df_log)

df_double_log=np.log(df_log)

df_double_log

test(df_double_log)

df_log

test_df_log=df_log.rolling(window=7).mean()
test_df_log.head(15)

moving_avg = df_log.rolling(window=12).mean()


ts_log_moving_avg_diff = df_log - moving_avg


ts_log_moving_avg_diff.dropna(inplace=True)
ts_log_moving_avg_diff.head(5)

test(ts_log_moving_avg_diff)

expwighted_avg = df_log.ewm(span=12).mean()
ts_log_ewma_diff = df_log - expwighted_avg
test(ts_log_ewma_diff)

ts_log_diff = df_log - df_log.shift()

ts_log_diff.dropna(inplace=True)
test(ts_log_diff)



ts_values=df_log.values

X = ts_values
size = int(len(X) * 0.667)
train, test = X[0:size], X[size:len(X)]

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from numpy.linalg import LinAlgError
import warnings
warnings.filterwarnings("ignore")

#installing pmdarima
!pip install pmdarima

#using auto arima to find best value for hyper parameter p ,d ,q
from pmdarima.arima import auto_arima

arima_model = auto_arima(train, start_p=1, start_q=1, d=1, max_p=4, max_q=4, start_P=1,
                         D=None, start_Q=1, max_P=4, max_D=1, max_Q=4, max_order=5, m=1,
                         seasonal=True, stationary=False, information_criterion='aic',
                         alpha=0.05, test='kpss', seasonal_test='ocsb', stepwise=True,
                         n_jobs=1, start_params=None, trend=None, method='lbfgs',
                         maxiter=50, offset_test_args=None, seasonal_test_args=None,
                         suppress_warnings=True, error_action='trace', trace=False,
                         random=False, random_state=None, n_fits=10,
                         return_valid_fits=False, out_of_sample_size=0,
                         scoring='mse', scoring_args=None, with_intercept='auto',
                         sarimax_kwargs=None)

arima_model.summary()

"""

1.   p-0
2.   d-1
3.   q-1


best hyper parameter getting from using auto-arima
"""

ts_values=df_log.values

X = ts_values
size = int(len(X) * 0.667)
train, test = X[0:size], X[size:len(X)]

#training will be 66%, test will be 33% as per our model
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from numpy.linalg import LinAlgError
import warnings
warnings.filterwarnings("ignore")

history = [x for x in train]
predictions = list()
#test.reset_index()
for t in range(len(test)):
    try:
        model = ARIMA(history, order=(0,1,1))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    except (ValueError, LinAlgError):
        pass
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
rmse = mean_squared_error(test, predictions)**0.5
print('Test MSE: %.3f' % rmse)


from math import sqrt
rms = sqrt(mean_squared_error(test, predictions))

from math import sqrt
rms = mean_squared_error(np.exp(test), np.exp(predictions))
print('Mean Squarred Error: %.2f'% rms)







test(df_log_minus)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
import warnings

# Function to test stationarity and perform differencing
def test_stationarity(data):
    rolmean = data.rolling(window=2).mean()
    rolstd = data.rolling(window=2).std()
    plt.figure(figsize=(25, 5))
    plt.plot(data, color='blue', label='original cases')
    plt.plot(rolmean, color='red', label='rolling mean')
    plt.plot(rolstd, color='black', label='rolling standard deviation')
    plt.legend(loc='best')
    plt.show()
    dftest = adfuller(data['num_orders'], autolag='t-stat')
    dfoutput = pd.Series(dftest[0:4], index=['test statistics', 'p_value', 'lags used', 'number of observations'])
    for key, value in dftest[4].items():
        dfoutput['critical value (%s)' % key] = value
    print(dfoutput)

data = df_ari.copy()

# Test stationarity
test_stationarity(data)

# Log transformation
df_log = np.log(df_ari)
plt.plot(df_log)

# Differencing
movingaverage = df_log.rolling(window=4).mean()
df_log_minus = df_log - movingaverage
df_log_minus.dropna(inplace=True)

# Test stationarity after differencing
test_stationarity(df_log_minus)

# Split the data into training and testing sets
ts_values = df_log.values
size = int(len(ts_values) * 0.667)
train, test = ts_values[0:size], ts_values[size:len(ts_values)]

# Perform a random search for ARIMA parameters
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

param_combinations = list(itertools.product(p_values, d_values, q_values))

best_rmse = float('inf')
best_params = None

for params in param_combinations:
    try:
        model = ARIMA(train, order=params)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test))
        rmse = sqrt(mean_squared_error(test, predictions))

        # Update best parameters if current combination gives a lower RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

        print(f'Params: {params}, RMSE: {rmse}')

    except (ValueError, LinAlgError):
        pass

print(f'Best Parameters: {best_params}, Best RMSE: {best_rmse}')

# Train the final model with the best parameters
final_model = ARIMA(ts_values, order=best_params)
final_model_fit = final_model.fit()

# Make predictions on the entire dataset
all_predictions = final_model_fit.forecast(steps=len(ts_values))

# Visualize predictions
plt.figure(figsize=(15, 5))
plt.plot(df_log, label='Original')
plt.plot(all_predictions, color='red', label='Predictions')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV

# Function to test stationarity and perform differencing
def test_stationarity(data):
    rolmean = data.rolling(window=2).mean()
    rolstd = data.rolling(window=2).std()
    plt.figure(figsize=(25, 5))
    plt.plot(data, color='blue', label='original cases')
    plt.plot(rolmean, color='red', label='rolling mean')
    plt.plot(rolstd, color='black', label='rolling standard deviation')
    plt.legend(loc='best')
    plt.show()
    dftest = adfuller(data['num_orders'], autolag='t-stat')
    dfoutput = pd.Series(dftest[0:4], index=['test statistics', 'p_value', 'lags used', 'number of observations'])
    for key, value in dftest[4].items():
        dfoutput['critical value (%s)' % key] = value
    print(dfoutput)

data = df_ari.copy()

# Test stationarity
test_stationarity(data)

# Log transformation
df_log = np.log(df_ari)
plt.plot(df_log)

# Differencing
movingaverage = df_log.rolling(window=4).mean()
df_log_minus = df_log - movingaverage
df_log_minus.dropna(inplace=True)

# Test stationarity after differencing
test_stationarity(df_log_minus)

# Split the data into training and testing sets
ts_values = df_log.values
size = int(len(ts_values) * 0.667)
train, test = ts_values[0:size], ts_values[size:len(ts_values)]

# Define the parameter grid for the grid search
param_grid = {
    'order': [(p, d, q) for p in range(3) for d in range(2) for q in range(3)]
}

# Create ARIMA model
arima = ARIMA(train, order=(1, 1, 1))  # Use initial values for 'order'

# Perform GridSearchCV
grid_search = GridSearchCV(arima, param_grid, scoring='neg_mean_squared_error', cv=3)
grid_search.fit(train)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# Train the final model with the best parameters
final_model = ARIMA(ts_values, order=best_params['order'])
final_model_fit = final_model.fit()

# Make predictions on the entire dataset
all_predictions = final_model_fit.forecast(steps=len(ts_values))

# Visualize predictions
plt.figure(figsize=(15, 5))
plt.plot(df_log, label='Original')
plt.plot(all_predictions, color='red', label='Predictions')
plt.legend()
plt.show()

from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(df_log_minus, nlags=50)
lag_pacf = pacf(df_log_minus, nlags=20, method='ols')
plt.figure(figsize=(10,8))
#plot acf gives p values
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_log_minus)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_log_minus)), linestyle='--', color='gray')
plt.title('ACF')
plt.legend(loc='best')

#plot pacf gives q value
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_log_minus)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_log_minus)), linestyle='--', color='gray')
plt.title('PACF')
plt.legend(loc='best')

df_log.head()

ts_values=df_log.values
print(train)
X = ts_values
size = int(len(X) * 0.667)
train, test = X[0:size], X[size:len(X)]

"""**without tunning arima**"""

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from numpy.linalg import LinAlgError
import warnings
warnings.filterwarnings("ignore")

history = [x for x in train]
predictions = list()
#test.reset_index()
for t in range(len(test)):
    try:
        model = ARIMA(history, order=(1,0,1))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    except (ValueError, LinAlgError):
        pass
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
rmse = mean_squared_error(test, predictions)**0.5
print('Test MSE: %.3f' % rmse)


from math import sqrt
rms = sqrt(mean_squared_error(test, predictions))

from math import sqrt
rms = sqrt(mean_squared_error(np.exp(test), np.exp(predictions)))
print('root Mean Squarred Error: %.2f'% rms)

"""**dhanayawad code naa chune ke liye**

**light model with tunning arima**
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

def arima_grid_search(train, test):
    best_mse = np.inf
    best_order = None

    p_values = range(5)
    d_values = range(3)
    q_values = range(5)

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)

                try:
                    model = ARIMA(train, order=order)
                    model_fit = model.fit()
                    predictions = model_fit.forecast(steps=len(test))
                    mse = mean_squared_error(test, predictions)

                    if mse < best_mse:
                        best_mse = mse
                        best_order = order

                except (ValueError, LinAlgError):
                    pass

    return best_order

# Assuming 'train' and 'test' are your time series data
best_order = arima_grid_search(train, test)
print("Best ARIMA Order:", best_order)

# Now, use the best order to fit the ARIMA model and make predictions
history = train.tolist()
predictions = list()

for t in range(len(test)):
    try:
        model = ARIMA(history, order=best_order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    except (ValueError, LinAlgError):
        pass

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

from math import sqrt
rms = sqrt(mean_squared_error(np.exp(test), np.exp(predictions)))
print('root Mean Squarred Error: %.2f'% rms)

"""**heavy tunning on arima**"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from numpy.linalg import LinAlgError
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
from itertools import product
warnings.filterwarnings("ignore")

def arima_grid_search(train, test, p_values, d_values, q_values):
    best_mse = np.inf
    best_order = None

    for p, d, q in product(p_values, d_values, q_values):
        try:
            model = ARIMA(train, order=(p, d, q))
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            mse = mean_squared_error(test, predictions)

            if mse < best_mse:
                best_mse = mse
                best_order = (p, d, q)

        except (ValueError, LinAlgError):
            pass

    return best_order

# Define the range of values for p, d, and q
p_values = range(5)
d_values = range(2)
q_values = range(5)

# Assuming 'train' and 'test' are your time series data
best_order = arima_grid_search(train, test, p_values, d_values, q_values)
print("Best ARIMA Order:", best_order)

# Now, use the best order to fit the ARIMA model and make predictions
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    try:
        model = ARIMA(history, order=best_order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    except (ValueError, LinAlgError):
        pass

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

from math import sqrt
rms = sqrt(mean_squared_error(np.exp(test), np.exp(predictions)))
print('root Mean Squarred Error: %.2f'% rms)

"""**without tuning regressor models**"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def preprocess_data(data):
    x_train = data.drop(columns='num_orders')
    y_train = data['num_orders']

    y_train = np.log1p(y_train)

    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=15, shuffle=False)

    return X_train, X_test, Y_train, Y_test

def train_models(X_train, X_test, Y_train, Y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'AdaBoost': AdaBoostRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Lasso': Lasso(),
        'Extra Trees': ExtraTreesRegressor()
    }

    mse_results = {}

    for model_name, model in models.items():
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)

        inverse_predictions = np.expm1(predictions)
        inverse_predictions = pd.DataFrame(inverse_predictions)
        inverse_predictions.index = Y_test.index

        mse = mean_squared_error(np.exp(Y_test), inverse_predictions)
        mse_results[model_name] = mse

        print(model_name, "--->", mse)

    mse_df = pd.DataFrame(list(mse_results.items()), columns=['Model', 'MSE'])
    mse_df.sort_values(by='MSE', inplace=True)

    return mse_df

# Assuming xb_data is your initial data frame
X_train, X_test, Y_train, Y_test = preprocess_data(xb_data)
mse_results_df = train_models(X_train, X_test, Y_train, Y_test)

# Display the results
print(mse_results_df)

"""**medium tuning regression model**"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def preprocess_data(data):
    x_train = data.drop(columns=['num_orders','Day','Month','Year','Quarter'])
    y_train = data['num_orders']
    y_train = np.log1p(y_train)

    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=15, shuffle=False)

    return X_train, X_test, Y_train, Y_test

def train_models_with_random_search(X_train, X_test, Y_train, Y_test, model, param_dist):
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, scoring='neg_mean_squared_error', cv=5)
    random_search.fit(X_train, Y_train)

    best_model = random_search.best_estimator_

    predictions = best_model.predict(X_test)

    inverse_predictions = np.expm1(predictions)
    inverse_predictions = pd.DataFrame(inverse_predictions)
    inverse_predictions.index = Y_test.index

    mse = mean_squared_error(np.exp(Y_test), inverse_predictions)

    print("Best Model:", best_model)
    print("MSE:", mse)

    return best_model

def train_models(X_train, X_test, Y_train, Y_test):
    models = {
        'Linear Regression': (LinearRegression(), {}),
        'Decision Tree': (DecisionTreeRegressor(), {}),
        'Random Forest': (RandomForestRegressor(), {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
        'AdaBoost': (AdaBoostRegressor(), {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1.0]}),
        'Gradient Boosting': (GradientBoostingRegressor(), {'n_estimators': [50, 100, 150], 'max_depth': [3, 4, 5], 'min_samples_split': [2, 5, 10]}),
        'Lasso': (Lasso(), {'alpha': [0.001, 0.01, 0.1, 1.0]}),
        'Extra Trees': (ExtraTreesRegressor(), {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]})
    }

    mse_results = {}

    for model_name, (model, param_dist) in models.items():
        print("Training", model_name)
        if param_dist:
            best_model = train_models_with_random_search(X_train, X_test, Y_train, Y_test, model, param_dist)
        else:
            model.fit(X_train, Y_train)
            best_model = model

        predictions = best_model.predict(X_test)

        inverse_predictions = np.expm1(predictions)
        inverse_predictions = pd.DataFrame(inverse_predictions)
        inverse_predictions.index = Y_test.index

        mse = mean_squared_error(np.exp(Y_test), inverse_predictions)
        mse_results[model_name] = mse

        print(model_name, "--->", mse)

    mse_df = pd.DataFrame(list(mse_results.items()), columns=['Model', 'MSE'])
    mse_df.sort_values(by='MSE', inplace=True)

    return mse_df

# Assuming xb_data is your initial data frame
X_train, X_test, Y_train, Y_test = preprocess_data(xb_data)
mse_results_df = train_models(X_train, X_test, Y_train, Y_test)

# Display the results
print(mse_results_df)

data.shape



"""**light tunning regression model**"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def preprocess_data(data):
    x_train = data.drop(columns=['num_orders','Day','Month','Year','Quarter'])
    y_train = data['num_orders']
    y_train = np.log1p(y_train)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    X_train, X_test, Y_train, Y_test = train_test_split(x_train_scaled, y_train, test_size=15, shuffle=False)

    return X_train, X_test, Y_train, Y_test, scaler

def train_models_with_random_search(X_train, X_test, Y_train, Y_test, model, param_dist):
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, scoring='neg_mean_squared_error', cv=5)
    random_search.fit(X_train, Y_train)

    best_model = random_search.best_estimator_

    predictions = best_model.predict(X_test)

    inverse_predictions = np.expm1(predictions)

    mse = mean_squared_error(np.exp(Y_test), inverse_predictions)

    print("Best Model:", best_model)
    print("MSE:", mse)

    return best_model

def train_models(X_train, X_test, Y_train, Y_test, scaler):
    models = {
        'Linear Regression': (LinearRegression(), {}),
        'Decision Tree': (DecisionTreeRegressor(), {}),
        'Random Forest': (RandomForestRegressor(), {'n_estimators': [10, 20, 30], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [2, 3, 7], 'min_samples_leaf': [1, 2, 4], 'max_features': ['auto', 'sqrt', 'log2', None]}),
        'AdaBoost': (AdaBoostRegressor(), {'n_estimators': [5, 10, 15], 'learning_rate': [0.01, 0.1, 1.0], 'loss': ['linear', 'square', 'exponential']}),
        'Gradient Boosting': (GradientBoostingRegressor(), {'n_estimators': [5, 10, 15], 'max_depth': [3, 4, 5], 'min_samples_split': [2, 5, 7], 'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.8, 0.9, 1.0]}),
        'Lasso': (Lasso(), {'alpha': [0.001, 0.01, 0.1, 1.0]}),
        'Extra Trees': (ExtraTreesRegressor(), {'n_estimators': [10, 20, 30], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [2, 5, 7], 'min_samples_leaf': [1, 2, 3], 'max_features': ['auto', 'sqrt', 'log2', None]})
    }

    mse_results = {}

    for model_name, (model, param_dist) in models.items():
        print("Training", model_name)
        if param_dist:
            best_model = train_models_with_random_search(X_train, X_test, Y_train, Y_test, model, param_dist)
        else:
            model.fit(X_train, Y_train)
            best_model = model

        predictions = best_model.predict(X_test)

        inverse_predictions = np.expm1(predictions)
        mse = mean_squared_error(np.exp(Y_test), inverse_predictions)
        mse_results[model_name] = mse

        print(model_name, "--->", mse)

    mse_df = pd.DataFrame(list(mse_results.items()), columns=['Model', 'MSE'])
    mse_df.sort_values(by='MSE', inplace=True)

    return mse_df

# Assuming xb_data is your initial data frame
X_train, X_test, Y_train, Y_test, scaler = preprocess_data(xb_data)
mse_results_df = train_models(X_train, X_test, Y_train, Y_test, scaler)

# Display the results
print(mse_results_df)

"""**heavy tunning regression model**"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def preprocess_data(data):
    x_train = data.drop(columns=['num_orders','Day','Month','Year','Quarter'])
    y_train = data['num_orders']
    y_train = np.log1p(y_train)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    X_train, X_test, Y_train, Y_test = train_test_split(x_train_scaled, y_train, test_size=15, shuffle=False)

    return X_train, X_test, Y_train, Y_test, scaler

def train_models_with_grid_search(X_train, X_test, Y_train, Y_test, model, param_grid):
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, Y_train)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    inverse_predictions = np.expm1(predictions)

    mse = mean_squared_error(np.exp(Y_test), inverse_predictions)

    print("Best Model:", best_model)
    print("MSE:", mse)

    return best_model

def train_models(X_train, X_test, Y_train, Y_test, scaler):
    models = {
        'Linear Regression': (LinearRegression(), {}),
        'Decision Tree': (DecisionTreeRegressor(), {}),
        'Random Forest': (RandomForestRegressor(), {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
        'AdaBoost': (AdaBoostRegressor(), {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1.0]}),
        'Gradient Boosting': (GradientBoostingRegressor(), {'n_estimators': [50, 100, 150], 'max_depth': [3, 4, 5], 'min_samples_split': [2, 5, 10]}),
        'Lasso': (Lasso(), {'alpha': [0.001, 0.01, 0.1, 1.0]}),
        'Extra Trees': (ExtraTreesRegressor(), {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]})
    }

    mse_results = {}

    for model_name, (model, param_grid) in models.items():
        print("Training", model_name)
        if param_grid:
            best_model = train_models_with_grid_search(X_train, X_test, Y_train, Y_test, model, param_grid)
        else:
            model.fit(X_train, Y_train)
            best_model = model

        predictions = best_model.predict(X_test)

        inverse_predictions = np.expm1(predictions)
        mse = mean_squared_error(np.exp(Y_test), inverse_predictions)
        mse_results[model_name] = mse

        print(model_name, "--->", mse)

    mse_df = pd.DataFrame(list(mse_results.items()), columns=['Model', 'MSE'])
    mse_df.sort_values(by='MSE', inplace=True)

    return mse_df

# Assuming xb_data is your initial data frame
X_train, X_test, Y_train, Y_test, scaler = preprocess_data(xb_data)
mse_results_df = train_models(X_train, X_test, Y_train, Y_test, scaler)

# Display the results
print(mse_results_df)



