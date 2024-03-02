import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from prophet import Prophet
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib
import json
import pickle

def xgbtrain(tuning):
    # Define the XGBoost model
    xgb_model = XGBRegressor()

    if tuning == 'No Tuning':
        model=xgb_model
    elif tuning == 'Grid Search':
        # Hyperparameter tuning using Grid Search
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5]
        }

        model = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')
    
    elif tuning == 'Random Search':
        param_dist = {
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5]
        }
        model = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=10, cv=3,
                                        scoring='neg_mean_absolute_error', random_state=42)

    return model

def random_forest_train(tuning):
    # Define the Random Forest model
    rf_model = RandomForestRegressor()
    if tuning == 'No Tuning':
        # Train the model without hyperparameter tuning
        model = rf_model.fit(X_train, Y_train)
    
    elif tuning == 'Grid Search':
        # Hyperparameter tuning using Grid Search
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20,30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        model = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=3, scoring='neg_mean_absolute_error')
    
    elif tuning == 'Random Search':
        # Hyperparameter tuning using Random Search
        param_dist_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        model = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist_rf, n_iter=10, cv=3,
                                            scoring='neg_mean_absolute_error', random_state=42)
    return model

def decision_tree_train(tuning):
    
    dec_tree_model = DecisionTreeRegressor()

    if tuning == 'No Tuning':
        # Train the model without hyperparameter tuning
        model = dec_tree_model
    
    elif tuning == 'Grid Search':
        param_grid = {
            'max_depth': [None, 5, 10, 15,20,30],
            'min_samples_split': [2, 5, 10,15,20],
            'min_samples_leaf': [1, 2, 4,5]
        }
        model = GridSearchCV(estimator=dec_tree_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')

    elif tuning == 'Random Search':
        # Hyperparameter tuning using Random Search
        param_dist = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        model = RandomizedSearchCV(estimator=dec_tree_model, param_distributions=param_dist, n_iter=10, cv=3,
                                        scoring='neg_mean_absolute_error', random_state=42)

    return model

def gbtrain(tuning):
    gb_model = GradientBoostingRegressor()

    if tuning == 'No Tuning':
        model = gb_model

    elif tuning == 'Grid Search':
        # Hyperparameter tuning using Grid Search
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2,0.3],
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5,6]
        }
        model = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')
    elif tuning == 'Random Search':    
        # Hyperparameter tuning using Random Search
        param_dist = {
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5]
        }

        model = RandomizedSearchCV(estimator=gb_model, param_distributions=param_dist, n_iter=10, cv=3,
                                            scoring='neg_mean_absolute_error', random_state=42)
        return model
    
def extra_tree_train(tuning):
    extra_tree_model = ExtraTreesRegressor()
    if tuning == 'No Tuning':
        model = extra_tree_model
    elif tuning == 'Grid Search':
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = GridSearchCV(estimator=extra_tree_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')
    elif tuning == 'Random Search':
        param_dist = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomizedSearchCV(estimator=extra_tree_model, param_distributions=param_dist, n_iter=10, cv=3,
                                            scoring='neg_mean_absolute_error', random_state=42)
    return model

def adaboost_train(tuning):
    adaboost_model = AdaBoostRegressor()
    if tuning == 'No Tuning':
        model = adaboost_model
    
    elif tuning == 'Grid Search':
        # Hyperparameter tuning using Grid Search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        }

        model = GridSearchCV(estimator=adaboost_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')

    elif tuning == 'Random Search':
        # Hyperparameter tuning using Random Search
        param_dist = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        }

        model = RandomizedSearchCV(estimator=adaboost_model, param_distributions=param_dist, n_iter=10, cv=3,
                                                scoring='neg_mean_absolute_error', random_state=42)

    return model


models = {
        "Random Forest": random_forest_train,
        "XGBoost": xgbtrain,
        "Gradient Bootsting": gbtrain,
        "Decision Tree": decision_tree_train,
        "Extra Trees Regressor": extra_tree_train,
        "AdaBoost Regressor": adaboost_train,
    }

def fblight(train_df):
                period = len(train_df)
                x_train = xb_data.drop(columns='num_orders')
                y_train = xb_data['num_orders']
                y_train = np.log1p(y_train)

                m = Prophet(
                    growth='linear',
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.05,
                )

                m.add_seasonality(name='weekly', period=7, fourier_order=15)
                m.add_seasonality(name='yearly', period=365.25, fourier_order=10)
                m.add_country_holidays(country_name='US')

                m.fit(train_df_processed)

                with open('D:/Study/INTERNSHIP/FINALDUPLICATE/data/models/model.pkl', 'wb') as file:
                    pickle.dump(m, file)
                    print("Prophet model saved successfully.")

                return m

def train_model(model_id,tuning):
    
    if model_id in models:
        if mlflow.active_run():
            mlflow.end_run()
        print(f"Training Model {models[model_id]}.")

        with mlflow.start_run():
            # Executes the corresponding model function
            model = models[model_id](tuning)
            model.fit(X_train, Y_train)

            y_pred = model.predict(X_test)

            signature = infer_signature(X_test, y_pred)
            mse = mean_squared_error(Y_test, y_pred)
            print(f"Mean Squared Error: {mse}")
            mlflow.log_metric("mse", mse)
            if tuning == 'No Tuning':
                mlflow.log_params(model.get_params())
                mlflow.sklearn.log_model(
                model, 
                models[model_id].__name__,
                signature=signature, 
                registered_model_name=models[model_id].__name__)

            else:
                mlflow.log_params(model.best_params_)
                mlflow.sklearn.log_model(
                model.best_estimator_, 
                models[model_id].__name__,
                signature=signature, 
                registered_model_name=models[model_id].__name__)

        #Save the model as pkl file
        with open('D:/Study/INTERNSHIP/FINALDUPLICATE/data/models/model.pkl','wb') as f:
            pickle.dump(model,f)
        
        with open(r'D:\Study\INTERNSHIP\FINALDUPLICATE\scaler.pkl','rb') as f:
            scaler=pickle.load(f)
        li=[]
        for i in range(max_week,max_week+6):
            li.append([i, center_id, meal_id, category, cuisine, checkout, base])
        scaler.transform(li)
        
        res=model.predict(li)
        print(res)
        res=[str(i) for i in res]
        with open(r"D:\Study\INTERNSHIP\FINALDUPLICATE\data\result\result.txt","w") as f:
            for i in res:
                f.write(i)
                f.write("\n")

    else:
        print(f"Invalid model_id: {model_id}")


"""
model_id = dvc.api.params.get('model_id')
tuning = dvc.api.params.get('tuning')
center_id = dvc.api.params.get('center_id')
meal_id = dvc.api.params.get('meal_id')
"""

model_id=""
tuning=""
center_id=""
meal_id=""

with open('params.json','r') as f:
    di=json.load(f)
    model_type=di['model_type']
    model_id=di['model_id']
    tuning=di['tuning']
    center_id=int(di['center_id'])
    meal_id=int(di['meal_id'])
    max_week=int(di['max'])
    category=int(di['category'])
    cuisine=int(di['cuisine'])
    checkout=float(di['checkout'])
    base=float(di['base'])

df = pd.read_csv("D:/Study/INTERNSHIP/FINALDUPLICATE/data/processed/preprocessed_data.csv")
X = df.drop(columns=['num_orders'], axis=1)
Y = df['num_orders']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

if center_id!=0 and meal_id!=0:
    df=df[df['center_id']==center_id]
    df=df[df['meal_id']==meal_id]

if model_type=="Time Series":
    train=df
    train_df = train[(train['center_id'] == center_id) & (train['meal_id'] == meal_id)]
    period = len(train_df)
    train_df['Date'] = pd.date_range('2015-01-08', periods=period, freq='W')
    filename = f'model.pkl'
    xb_data = train_df.drop(columns=['center_id', 'meal_id', 'category', 'cuisine'])
    xb_data = xb_data.set_index(['Date'])
    def preprophet(train_df):
                split_size = period - 15
                prophet_data = train_df[['Date', 'num_orders']]
                prophet_data.index = xb_data.index
                prophet_data = prophet_data.iloc[:split_size, :]
                prophet_data = prophet_data.rename(columns={'Date': 'ds', 'num_orders': 'y'})
                return prophet_data
    train_df_processed = preprophet(train_df)
    prophet_model = fblight(train_df)

    future_period = 5
    last_training_date = prophet_model.history['ds'].max()
    # Create a future DataFrame for the next 5 days with WEEK frequency
    future = pd.date_range(start=last_training_date + pd.DateOffset(1), periods=future_period, freq='W')
    future_df = pd.DataFrame({'ds': future})

    # Make sure 'ds' column is in datetime format
    future_df['ds'] = pd.to_datetime(future_df['ds'])

    res=prophet_model.predict(future_df)['yhat']

    res=[str(i) for i in res]
    with open(r"D:\Study\INTERNSHIP\FINALDUPLICATE\data\result\result.txt","w") as f:
        for i in res:
            f.write(i)
            f.write("\n")

# elif model_type=="LSTM":
#     def preprolstm(train_df,center_id,meal_id):
#         train_df=train[train["center_id"]==center_id]
#         train_df=train_df[train["meal_id"]==meal_id]
#         period = len(train_df)
#         train_df['Date'] = pd.date_range('2015-01-08', periods=period, freq='W')
#         train_df = train_df.set_index(['Date'])
#         lstm_data = train_df.drop(columns=['id','center_id','meal_id','category','cuisine','week'])
#         test_split=round(len(lstm_data)*0.20)
#         df_for_training=lstm_data[:-test_split]
#         df_for_testing=lstm_data[-test_split:]
#         scaler = MinMaxScaler(feature_range=(0,1))
#         df_for_training_scaled = scaler.fit_transform(df_for_training)
#         df_for_testing_scaled=scaler.transform(df_for_testing)
#         def createXY(dataset,n_past):
#             dataX = []
#             dataY = []
#             for i in range(n_past, len(dataset)):
#                     dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
#                     dataY.append(dataset[i,0])
#             return np.array(dataX),np.array(dataY)
#         trainX,trainY=createXY(df_for_training_scaled,20)
#         testX,testY=createXY(df_for_testing_scaled,20)
#         return testX,testY,trainX,trainY
    
#     testX,testY,trainX,trainY=preprolstm(df,center_id,meal_id)

#     def trainlstm(testX,testY,trainX,trainY):
#         from keras.wrappers.scikit_learn import KerasRegressor
#         #applying model
#         def build_model(optimizer):
#             grid_model = Sequential()

#             grid_model.add(LSTM(50,return_sequences=True,input_shape=(20,5)))
#             grid_model.add(LSTM(50))
#             grid_model.add(Dropout(0.2))
#             grid_model.add(Dense(1))
#             grid_model.compile(loss = 'mse',optimizer = optimizer)
#             return grid_model
#         grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
#         parameters = {'batch_size' : [2,4],
#                     'epochs' : [8,10],
#                     'optimizer' : ['adam','Adadelta'] }
#         grid_search  = GridSearchCV(estimator = grid_model,
#                                     param_grid = parameters,
#                                     cv = 2)
#         grid_search = grid_search.fit(trainX,trainY)
#         grid_search.best_params_
#         lstm_model=grid_search.best_estimator_.model
#         return lstm_model

#     lstm_model=trainlstm(testX,testY,trainX,trainY)
    
else:
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    train_model(model_id,tuning)