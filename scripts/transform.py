
import numpy as np
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
#from gain import GAIN
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_iterative_imputer
# from missingpy import MissForest
from scipy.stats.mstats import winsorize
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, MaxAbsScaler, Normalizer, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder, HelmertEncoder, SumEncoder, BackwardDifferenceEncoder, LeaveOneOutEncoder, JamesSteinEncoder, BinaryEncoder
import pickle
import sys
import dvc.api
import json

"""
model_type = dvc.api.params.get('model_type')
missing_option = dvc.api.params.get('null_values')
categorical_option = dvc.api.params.get('encoding')
scaling_option = dvc.api.params.get('scaling')
"""

with open('D:/Study/INTERNSHIP/FINALDUPLICATE/params.json','r') as f:
    di=json.load(f)
    model_type=di['model_type']
    missing_option=di['null_values']
    categorical_option=di['encoding']
    scaling_option=di['scaling']
    center_id = di['center_id']
    meal_id = di['meal_id']
    
def get_categorical_columns(dataset):
    # Assuming categorical columns have 'object' data type, you can adjust the condition based on your dataset
    print(dataset.info())
    return list(dataset.select_dtypes(include=['object']).columns)

def transform(dataset, missing_option, categorical_option, scaling_option):
    print(missing_option, categorical_option, scaling_option)
    # Handling missing values
    '''
    print("\nHandling Missing Values:")
    print("1. Mean Imputation")
    print("2. Median Imputation")
    print("3. Custom Value Imputation")
    print("4. Most Frequent Imputation")
    print("5. KNN Imputation")
    print("6. Linear Regression Imputation")
    print("7. Iterative Imputation")
    print("8. Multiple Imputation by Chained Equations (MICE)")
    print("9 Autoencoder Imputation")
    '''
    missing_columns = dataset.columns[dataset.isnull().any()].tolist()
    
    if missing_columns:
        
        if missing_option == "Mean":
            imputer = SimpleImputer(strategy='mean')
        elif missing_option == "Median":
            imputer = SimpleImputer(strategy='median')
        elif missing_option == "Mode":
            imputer = SimpleImputer(strategy='most_frequent')
        elif missing_option == "Linear Regression Imputation":
            imputer = IterativeImputer(max_iter=10, random_state=0)
        
        dataset[missing_columns] = imputer.fit_transform(dataset[missing_columns])
    
    # Encoding categorical data
    '''print("\nEncoding Categorical Data:")
    print("1. One-Hot Encoding")
    print("2. Label Encoding")
    print("3. Target Encoding")
    print("4. Helmert Coding")
    print("5. Sum Coding")
    print("6. Backward Difference Coding")
    print("7. Leave-One-Out Encoding")
    print("8. James-Stein Encoder")
    print("9. Binary Encoding")'''
    
    categorical_columns = get_categorical_columns(dataset)
    if categorical_option=='None':
        print("None")

    elif categorical_columns and categorical_option == 'One Hot Encoding':
        encoder = OneHotEncoder()
        print(dataset.iloc[0])
        dataset = pd.DataFrame(encoder.fit_transform(dataset[categorical_columns]).toarray(), columns=encoder.get_feature_names_out(categorical_columns))
        
    elif categorical_columns and categorical_option == 'Label Encoding':
        encoder = LabelEncoder()
        # Encode categorical columns with LabelEncoder
        for column in categorical_columns:
            dataset[column] = encoder.fit_transform(dataset[column])
       
    elif categorical_columns and categorical_option == 'Helmert Encoding':
        encoder = HelmertEncoder(cols=categorical_columns)
        dataset[categorical_columns] = encoder.fit_transform(dataset[categorical_columns])
    
    elif categorical_columns and categorical_option == 'Sum Encoding':
        encoder = SumEncoder(cols=categorical_columns)
        dataset[categorical_columns] = encoder.fit_transform(dataset[categorical_columns])
    
    elif categorical_columns and categorical_option == 'Backward Difference Encoding':
        encoder = BackwardDifferenceEncoder(cols=categorical_columns)
        dataset[categorical_columns] = encoder.fit_transform(dataset[categorical_columns])
         
    elif categorical_columns and categorical_option == 'James-Stein Encoder':
        encoder = JamesSteinEncoder(cols=categorical_columns)
        dataset[categorical_columns] = encoder.fit_transform(dataset[categorical_columns])
    
    else:
        raise ValueError("Invalid input for encoding categorical data!!")
    
    if categorical_option!='None':
        print(categorical_option)
        with open('encoder.pkl', 'wb') as file:
            pickle.dump(encoder, file)

    # Feature scaling
    '''print("\nFeature Scaling:")
    print("1. Standard Scaling (Z-score)")
    print("2. Min-Max Scaling")
    print("3. Robust Scaling")
    print("4. Power Transformation (Yeo-Johnson)")
    print("5. Quantile Transformation")
    print("6. MaxAbsScaler")
    print("7. Normalizer")'''
    numerical_columns = list(set(dataset.columns) - set(categorical_columns))

    print(numerical_columns)
    print(categorical_columns)
    if scaling_option == 'Standard Scaling':
        scaler = StandardScaler(with_mean=False)  # Pass with_mean=False for sparse matrices
        dataset.iloc[:, :] = scaler.fit_transform(dataset)

    elif scaling_option == 'Min-Max Scaling':
        scaler = MinMaxScaler()
        dataset.iloc[:, :] = scaler.fit_transform(dataset)

    elif scaling_option == 'Robust Scaling':
        scaler = RobustScaler()
        dataset.iloc[:, :] = scaler.fit_transform(dataset)

    elif scaling_option == 'Power Transformation':
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        dataset.iloc[:, :] = scaler.fit_transform(dataset)

    elif scaling_option == 'Quantile Transformation':
        scaler = QuantileTransformer(output_distribution='uniform')
        dataset.iloc[:, :] = scaler.fit_transform(dataset)

    elif scaling_option == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
        dataset.iloc[:, :] = scaler.fit_transform(dataset)

    elif scaling_option == 'Normalizer':
        scaler = Normalizer()
        dataset.iloc[:, :] = scaler.fit_transform(dataset)

    else:
        print(scaling_option)
        raise ValueError("Invalid input for feature scaling technique.")
    
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    return dataset

def pretime(train_df):

  period = len(train_df)
  train_df['Date'] = pd.date_range('2015-01-08', periods=period, freq='W')
  train_df['Day'] = train_df['Date'].dt.day
  train_df['Month'] = train_df['Date'].dt.month
  train_df['Year'] = train_df['Date'].dt.year
  train_df['Quarter'] = train_df['Date'].dt.quarter

  return train_df


######################## Main Code Starts ################################
  
#dir path
raw_data_path = 'D:/Study/INTERNSHIP/FINALDUPLICATE/data/raw/'
processed_data_path = 'D:/Study/INTERNSHIP/FINALDUPLICATE/data/processed/preprocessed_data.csv'

merged_df = pd.read_csv(raw_data_path + 'train1.csv')

#transforms merged_df
merged_df.columns = ['id', 'week', 'center_id', 'city_code', 'region_code', 'center_type', 'op_area', 'meal_id', 'category', 'cuisine', 'checkout_price', 'base_price', 'emailer_for_promotion', 'homepage_featured', 'num_orders']
merged_df.drop(columns=['city_code','region_code','center_type','op_area'], inplace=True)

int_cols=['id','week','center_id','meal_id','emailer_for_promotion','homepage_featured','num_orders']
float_cols=['checkout_price','base_price']

merged_df[int_cols] = merged_df[int_cols].astype('int64')
merged_df[float_cols] = merged_df[float_cols].astype('float64')

filtered_df = merged_df[(merged_df['meal_id'] == int(meal_id)) & (merged_df['center_id'] == int(center_id))]
print(filtered_df.head())
max_week = filtered_df['week'].max()
checkout = filtered_df['checkout_price'].mean()
base = filtered_df['base_price'].mean()





merged_df.drop(['id','emailer_for_promotion','homepage_featured'], axis=1, inplace=True)
X=merged_df.drop(['num_orders'],axis=1)
y=merged_df['num_orders']

transformed_df = transform(X, missing_option, categorical_option, scaling_option)
transformed_df['num_orders']=y
print(transformed_df.head())

filtered_df = transformed_df[(transformed_df['meal_id'] == int(meal_id)) & (transformed_df['center_id'] == int(center_id))]
category = filtered_df['category'].iloc[0]
cuisine = filtered_df['cuisine'].iloc[0]

with open('D:/Study/INTERNSHIP/FINALDUPLICATE/params.json','r') as f:
    di=json.load(f)
    di['max']=str(max_week)
    di['category']=str(category)
    di['cuisine']=str(cuisine)
    di['checkout']=str(checkout)
    di['base']=str(base)

with open('D:/Study/INTERNSHIP/FINALDUPLICATE/params.json','w') as f:
    json.dump(di,f)

transformed_df.to_csv(processed_data_path, index=False)
print('done')