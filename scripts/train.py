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
gb_regressor = GradientBoostingRegressor()
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor    

# Importing the dataset
print("Importing the dataset.. ")
df = pd.read_csv("bigdata.csv")

# One Hot Encoding
categorical_cols = ['center_type', 'category', 'cuisine']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in categorical_cols:
    df[column] = label_encoder.fit_transform(df[column])
    
# Outlier Removal
import seaborn as sns
percentile25 = df['num_orders'].quantile(0.25)
percentile75 = df['num_orders'].quantile(0.75)
iqr=percentile75-percentile25
upper_limit = percentile75 + 1.5*iqr
lower_limit = percentile25 - 1.5*iqr
tf=df[df['num_orders'] > upper_limit]
new_df=df[df['num_orders'] < upper_limit]
percentile25 = new_df['base_price'].quantile(0.25)
percentile75 = new_df['base_price'].quantile(0.75)
iqr=percentile75-percentile25
print(iqr)
upper_limit = percentile75 + 1.5*iqr
lower_limit = percentile25 - 1.5*iqr
print("upper limit ",upper_limit)
print("lower limit ",lower_limit)
tf=new_df[new_df['base_price'] > upper_limit]
tf.shape
new_df=new_df[new_df['base_price'] < upper_limit]
new_df.shape
sns.boxplot(new_df['num_orders'])
percentile25 = new_df['num_orders'].quantile(0.25)
percentile75 = new_df['num_orders'].quantile(0.75)
iqr=percentile75-percentile25
print(iqr)
upper_limit = percentile75 + 1.5*iqr
lower_limit = percentile25 - 1.5*iqr
print("upper limit ",upper_limit)
print("lower limit ",lower_limit)
tf=new_df[new_df['num_orders'] > upper_limit]
tf.shape
new_df=new_df[new_df['num_orders'] < upper_limit]
new_df.shape

#Splitting and Feature Selection
X=new_df.drop(columns=['num_orders'],axis=1)
Y=new_df['num_orders']
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
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=10,  # Number of iterations
        cv=5,  # Cross-validation folds
        verbose=2,
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    print("Fitting the model..")
    random_search.fit(X_train, Y_train)
    print("Model fitted..")

    y_pred = random_search.predict(X_test)

    mse = mean_squared_error(Y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
def XGBoostRegressor():
    # XGBoost Regressor
    print("XGBoost Regressor")
    
def GradientBoostingRegressor():
    # Gradient Boosting Regressor
    print("Gradient Boosting Regressor")
    
def Lasso():
    # Lasso
    print("Lasso")

def DecisionTreeRegressor():
    # Decision Tree Regressor
    print("Decision Tree Regressor")
    
def ExtraTreesRegressor():
    # Extra Trees Regressor
    print("Extra Trees Regressor")
    
def AdaBoostRegressor():
    # Ada Boost Regressor
    print("Ada Boost Regressor")

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
        # Executes the corresponding model function
        result = models[model_id]()

        print(f"Training Model {models[model_id]}.")
    else:
        print(f"Invalid model_id: {model_id}")
    
train_model(1)
    
    