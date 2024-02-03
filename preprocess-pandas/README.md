# Handling Missing Values 

## 1) Deletion (CCA):

- **Listwise Deletion:** Remove entire rows containing missing values. Suitable when the amount of missing data is relatively small, and removing those rows doesn't significantly affect the analysis.

- **Column-wise Deletion:** Remove entire columns (features) with missing values. Appropriate when a feature has a substantial amount of missing data or is not essential for the analysis.

## 2) Imputation:

### Mean, Median, or Mode Imputation (Simple Imputer): 
- Replace missing values with the mean, median, or mode of the observed values in the respective column. Suitable for numerical data but may not be ideal if the data has outliers.

### Arbitrary Value Imputation:
- Replace missing values with an arbitrary value. The value should be some unique value that is not there in the dataset.

### End of Distribution Imputation:
- Replace missing values by taking the end value of your distribution.

### Random Imputation:
- Replace missing value in a column with a random value taken from the same column. The distribution and variance remain the same. Good for linear algorithms.

### Missing Indicator:
- For every column that has missing values, create a new column that consists of only two values - true and false. False indicates no missing value was there, whereas True indicates a missing value. Using this, the model learns to differentiate between rows with missing and non-missing values.

### Automatically Select Value for Imputation:
- GridSearchCV

### Forward Fill or Backward Fill: 
- Fill missing values with the preceding (forward fill) or succeeding (backward fill) non-missing values. This is often used for time-series data.

### MICE algorithm (Multiple Imputation by Chained Equations)

### Imputation by Predictive Modeling: 
- Use machine learning algorithms to predict missing values based on other features. This is more advanced and may require dividing the dataset into two sets: one with complete data for training and one with missing values for prediction.

## 3) K-Nearest Neighbors (KNN) Imputation: 
- Estimate missing values by averaging the values of their k-nearest neighbors. This method is effective for both numerical and categorical data.

## 4) Data Augmentation:
- Create synthetic data to replace missing values, either by replicating existing data or generating new samples using techniques like bootstrapping.

The choice of the method depends on factors such as the amount of missing data, the nature of the data, the reason for missingness, and the requirements of the analysis or machine learning task. It's often a good practice to explore and understand the data before deciding on the most appropriate method for handling missing values.

# Choosing the Right Technique for Handling Missing Values

## 1) Understand the Nature of Missing Data:

Assess the pattern of missing values. Are they missing completely at random, missing at random, or missing not at random? Understanding the pattern can guide your choice of imputation methods.

## 2) Analyze the Missing Data Percentage:

If a feature has a very high percentage of missing values (e.g., more than 80%), consider whether it's reasonable to drop the entire feature or if it's critical for your analysis.

## 3) Consider the Type of Data:

- Categorical variables: For categorical data, you might consider using mode imputation or creating an additional category for missing values.
- Numerical variables: For numerical data, mean, median, or regression-based imputation methods might be suitable.

## 4) Evaluate the Impact on Data Distribution:

Before and after imputation, assess whether the distribution of the variable with missing values has significantly changed. Some imputation methods may introduce bias or distort the original data distribution.

## 5) Explore Imputation Techniques:

Try different imputation techniques and compare their performance. This could involve comparing mean, median, and mode imputation, as well as more advanced techniques like KNN imputation, regression imputation, or machine learning-based imputation.

## 6) Consider the Overall Analysis Goals:

The imputation method should align with the goals of your analysis or modeling task. For example, if predictive accuracy is crucial, machine learning-based imputation methods might be appropriate.

# Encoding Categorical Data

- Encoding categorical data is a crucial step in preparing your data for machine learning algorithms, as many algorithms require numerical input. There are several techniques for encoding categorical data, and the choice depends on the nature of your data. 

## 1) One-Hot Encoding:

One-hot encoding is a popular method for handling categorical variables. It creates binary columns for each category and represents the presence of a category with a 1 and its absence with a 0. This method is suitable when the categories are not ordinal (i.e., there is no inherent order).

## 2) Label Encoding:

Label encoding assigns a unique integer to each category. It's appropriate when there is an ordinal relationship between the categories. However, be cautious when using label encoding with algorithms that may interpret the encoded integers as ordinal values.

## 3) Ordinal Encoding:

Ordinal encoding manually assigns numerical values to categories based on their order. This is suitable when the categories have a meaningful order, and you want to preserve that information. If the income categories represent a range, and the order has meaning, ordinal encoding could be suitable.

## 4) Binary Encoding:

Binary encoding represents each category with a binary code. It's particularly useful when dealing with high-cardinality categorical features.

## 5) Frequency or Count Encoding:

Frequency encoding replaces categories with their frequency (or count) in the dataset. This can be beneficial when the frequency of occurrence is informative.

## 6) Target Encoding (Mean Encoding):

Replaces each category with the mean of the target variable for that category. Useful for binary classification problems. Helps encode information about the target variable into the categorical variable.

# Feature Scaling

## 1) Min-Max Scaling (Normalization):

Use when the features have a clear minimum and maximum, and you want to scale them to a specific range, often [0, 1]. Suitable for algorithms that rely on distances between data points, such as k-nearest neighbors.

## 2) Standardization (Z-score normalization):

Use when the features have a roughly Gaussian distribution and the algorithm assumes zero-centered data. Suitable for algorithms that assume normally distributed features like linear regression, logistic regression, and support vector machines.

## 3) Robust Scaling:

Use when your data contains outliers, as this method is less sensitive to them compared to standardization. Suitable for algorithms that are sensitive to outliers, such as clustering algorithms.

## 4) Max Abs Scaling:

Use when your data has a mix of positive and negative values and you want to scale each feature to the range [-1, 1]. Suitable for algorithms that are sensitive to the scale of features, such as neural networks.

## 5) Box-Cox Transformation:

Use when your data has a varying degree of skewness and you want to find a suitable power transformation. Suitable for data that may have different types of distributions.

Skewed data refers to the asymmetry or lack of symmetry in the distribution of a dataset. In a perfectly symmetric distribution, the two halves on either side of the center look like mirror images of each other. When a distribution is skewed, one tail is longer or stretched out compared to the other, and the data points are not evenly distributed around the mean.

# Handling Outliers

## 1. Trimming or Winsorizing:

Remove or replace extreme values with less extreme values. For example, values beyond a certain percentile can be set to the value at that percentile.

## 2. Imputation:

Replace outliers with a central value, such as the mean, median, or mode. This can be a simple and quick way to handle outliers, but it may not be suitable for all cases.

## 3. Capping or Flooring:

Set a threshold beyond which values are capped or floored. Values above the threshold are set to the threshold value, and values below the threshold are set to the threshold value.

## 4. Z-Score or Standard Score:

Use z-scores to identify and remove or transform outliers. Z-scores indicate how many standard deviations a data point is from the mean.

## 5. IQR (Interquartile Range) Method:

Identify outliers based on the interquartile range (IQR), which is the range between the first quartile (Q1) and third quartile (Q3). Values outside the range Q1-1.5 x IQR to Q3+1.5 x IQR are considered outliers.
