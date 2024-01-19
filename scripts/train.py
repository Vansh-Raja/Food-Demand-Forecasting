### Temporary Code till def RandomForestRegressor()

# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
import mlflow
import mlflow.sklearn
import sys
import joblib

df = pd.read_csv("D:/Study/INTERNSHIP/ETL1/data/processed/preprocessed_data.csv")

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

#Splitting and Feature Selection
X=df.drop(columns=['num_orders'],axis=1)
Y=df['num_orders']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=2)
from sklearn.feature_selection import SelectKBest, f_classif
k_best = SelectKBest(f_classif, k=6)
X_train_filtered = k_best.fit_transform(X_train, Y_train)
X_test_filtered = k_best.transform(X_test)

# Instanciating the models:
rf_model = RandomForestRegressor()
xgb_model = XGBRegressor()
gb_regressor = GradientBoostingRegressor()
model = Lasso()
dt_regressor = DecisionTreeRegressor()
et_model = ExtraTreesRegressor()
adaboost = AdaBoostRegressor()

def RandomForestRegressor():
    
    print("Training Model RandomForestRegressor.")

    param_grid = {
        'n_estimators': [int(x) for x in range(100, 1001, 100)],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [int(x) for x in range(10, 110, 10)],
        'min_samples_split': [2, 5, 10  ],
        'min_samples_leaf': [1, 2, 4]
    }

    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=5,  # Number of iterations
        cv=5,  # Cross-validation folds
        verbose=2,
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )
    return random_search

    
def XGBoostRegressor():
          
    param_dist2 = {
    'n_estimators': [int(x) for x in range(100, 1001, 100)],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    }
    
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist2,
        n_iter=10,  # Adjust the number of iterations as needed
        scoring='neg_mean_squared_error',
        cv=5,  # Number of cross-validation folds
        verbose=1,
        n_jobs=-1,  # Use all available CPU cores
        random_state=42
    )
    return random_search
   
     
def GradientBoostingRegressor():
    # Gradient Boosting Regressor
    print("Gradient Boosting Regressor")

    
def Lasso():
       
    from scipy.stats import uniform
    param_grid = {'alpha': uniform(0, 1)}
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=100, cv=5, random_state=42)
    return random_search


def DecisionTreeRegressor():
    # Decision Tree Regressor
    print("Decision Tree Regressor")
    

def ExtraTreesRegressor():
    from scipy.stats import randint
    param_dist = {
        'n_estimators': randint(10, 200),
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': randint(1, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(estimator=et_model, param_distributions=param_dist, n_iter=100, cv=5, verbose=1, n_jobs=-1)
    
    return random_search

    
def AdaBoostRegressor():
         
    from scipy.stats import randint
    
    param_dist = {
    'n_estimators': randint(50, 200),
    'learning_rate': [0.01, 0.1, 0.5, 1],
    }


models = {
        1: RandomForestRegressor,
        2: XGBoostRegressor,
        3: GradientBoostingRegressor,
        4: Lasso,
        5: DecisionTreeRegressor,
        6: ExtraTreesRegressor,
        7: AdaBoostRegressor,
    }


def train_model(model_id):
    
    if model_id in models:
        if mlflow.active_run():
            mlflow.end_run()
        print(f"Training Model {models[model_id]}.")
        with mlflow.start_run():
            # Executes the corresponding model function
            model = models[model_id]()
            model.fit(X_train, Y_train)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(Y_test, y_pred)
            print(f"Mean Squared Error: {mse}")
            
            mlflow.log_params(model.best_params_)
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(model.best_estimator_, models[model_id].__name__)

            #Save the model as pkl file
            model_path = f"D:/Study/INTERNSHIP/ETL1/data/models/model.pkl"
            joblib.dump(model.best_estimator_, model_path)

    else:
        print(f"Invalid model_id: {model_id}")

model_id = sys.argv[1]
train_model(int(model_id))