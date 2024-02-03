
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
from category_encoders import TargetEncoder, HelmertEncoder, SumEncoder, BackwardDifferenceEncoder, LeaveOneOutEncoder, JamesSteinEncoder, BinaryEncoder
def get_categorical_columns(dataset):
    # Assuming categorical columns have 'object' data type, you can adjust the condition based on your dataset
    return list(dataset.select_dtypes(include=['object']).columns)
def transform(dataset, missing_option, categorical_option, scaling_option):
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
        
        if missing_option == 1:
            imputer = SimpleImputer(strategy='mean')
        elif missing_option == 2:
            imputer = SimpleImputer(strategy='median')
        elif missing_option == 3:
            custom_value = float(input("Enter the custom value: "))
            imputer = SimpleImputer(strategy='constant', fill_value=custom_value)
        elif missing_option == 4:
            imputer = SimpleImputer(strategy='most_frequent')
        elif missing_option == 5:
            k_value = int(input("Set the value for k(no. of neighbors) "))
            imputer = KNNImputer(n_neighbors=k_value)
        elif missing_option == 6:
            imputer = IterativeImputer(max_iter=10, random_state=0)
        elif missing_option == 7:
            # Assumption -> 'target_column' is the column with missing values
            target_column = input("Enter the target column for linear regression imputation: ")
            imputer = IterativeImputer(estimator=LinearRegression(), target=target_column)
        elif missing_option == 8:
            imputer = IterativeImputer(max_iter=10, random_state=0, n_nearest_features=5)
        elif missing_option == 9:
            # Assumption -->  'target_column' is the column with missing values
            target_column = input("Enter the target column for autoencoder imputation: ")
            
            autoencoder = MLPRegressor(hidden_layer_sizes=(100, 50, 100), max_iter=500, random_state=1)
            imputer = IterativeImputer(estimator=autoencoder, target=target_column)
        else:
            raise ValueError("Invalid input for missing values handling technique!!")
        
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
    if categorical_columns and categorical_option == 1:
        encoder = OneHotEncoder()
        print(dataset.iloc[0])
        dataset = pd.DataFrame(encoder.fit_transform(dataset[categorical_columns]).toarray(), columns=encoder.get_feature_names_out(categorical_columns))
        
    elif categorical_columns and categorical_option == 2:
        dataset[categorical_columns] = dataset[categorical_columns].astype('category')
        for column in categorical_columns:
            dataset[column] = dataset[column].cat.codes
    
    elif categorical_columns and categorical_option == 3:
        encoder = TargetEncoder(cols=categorical_columns)
        dataset[categorical_columns] = encoder.fit_transform(dataset[categorical_columns], dataset[target_column])
        
    elif categorical_columns and categorical_option == 4:
        encoder = HelmertEncoder(cols=categorical_columns)
        dataset[categorical_columns] = encoder.fit_transform(dataset[categorical_columns])
    
    elif categorical_columns and categorical_option == 5:
        encoder = SumEncoder(cols=categorical_columns)
        dataset[categorical_columns] = encoder.fit_transform(dataset[categorical_columns])
    
    elif categorical_columns and categorical_option == 6:
        encoder = BackwardDifferenceEncoder(cols=categorical_columns)
        dataset[categorical_columns] = encoder.fit_transform(dataset[categorical_columns])
        
    elif categorical_columns and categorical_option == 7:
        encoder = LeaveOneOutEncoder(cols=categorical_columns)
        dataset[categorical_columns] = encoder.fit_transform(dataset[categorical_columns], dataset[target_column])
        
    elif categorical_columns and categorical_option == 8:
        encoder = JamesSteinEncoder(cols=categorical_columns)
        dataset[categorical_columns] = encoder.fit_transform(dataset[categorical_columns])
        
    elif categorical_columns and categorical_option == 9:
        encoder = BinaryEncoder(cols=categorical_columns)
        dataset[categorical_columns] = encoder.fit_transform(dataset[categorical_columns])
        
    else:
        raise ValueError("Invalid input for encoding categorical data!!")
    
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
    if scaling_option == 1 and numerical_columns:
        scaler = StandardScaler(with_mean=False)  # Pass with_mean=False for sparse matrices
        dataset.iloc[:, :] = scaler.fit_transform(dataset)
    elif scaling_option == 2:
        scaler = MinMaxScaler()
        dataset.iloc[:, :] = scaler.fit_transform(dataset)
    elif scaling_option == 3:
        scaler = RobustScaler()
        dataset.iloc[:, :] = scaler.fit_transform(dataset)
    elif scaling_option == 4:
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        dataset.iloc[:, :] = scaler.fit_transform(dataset)
    elif scaling_option == 5:
        scaler = QuantileTransformer(output_distribution='uniform')
        dataset.iloc[:, :] = scaler.fit_transform(dataset)
    elif scaling_option == 6:
        scaler = MaxAbsScaler()
        dataset.iloc[:, :] = scaler.fit_transform(dataset)
    elif scaling_option == 7:
        scaler = Normalizer()
        dataset.iloc[:, :] = scaler.fit_transform(dataset)
    else:
        raise ValueError("Invalid input for feature scaling technique.")
    
    return dataset
    
#dir path
raw_data_path = 'D:\\uni\\Python\\Pranav_Gryffindor_script\\ETL1 (2)\\ETL1\\data\\raw\\'
processed_data_path = 'D:\\uni\\Python\\Pranav_Gryffindor_script\\ETL1 (2)\\ETL1\\data\\processed\\preprocessed_data.csv'



#reads all csv files in the dir and makes a merged df of them
merged_df = pd.read_csv(raw_data_path + 'final.csv')

#transforms merged_df
merged_df.columns = ['id', 'week', 'center_id', 'city_code', 'region_code', 'center_type', 'op_area', 'meal_id', 'category', 'cuisine', 'checkout_price', 'base_price', 'emailer_for_promotion', 'homepage_featured', 'num_orders']
transformed_df = transform(merged_df,1,2,3)
transformed_df.to_csv(processed_data_path, index=False)
print('done')