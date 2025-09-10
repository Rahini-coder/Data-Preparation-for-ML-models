### Prepare Data for ML Algorithms

## 1. Data Cleaning : Median Imputation

# Code to impute with numpy and pandas
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

df = pd.read_csv("procedures_new.csv")

#log transform for symmetric data as data is right skewed,compressing large values
df['Log_MIN'] = np.log(df['MIN'] +1)
df['Log_MAX'] = np.log(df['MAX'] +1)
df['Log_MODE'] = np.log(df['MODE'] +1)

#Calculate Median
median_MIN = df['Log_MIN'].median()
median_MAX = df['Log_MAX'].median()
median_MODE = df['Log_MODE'].median()

print (median_MIN) # 5.650691900365585
print (median_MAX) # 6.516193076042964
print (median_MODE) # 4.134526351854933

#impute missing values in log space
log_values_filled_MIN = df['Log_MIN'].fillna(median_MIN)
log_values_filled_MAX = df['Log_MAX'].fillna(median_MAX)
log_values_filled_MODE = df['Log_MODE'].fillna(median_MODE)

#exponential Inverse transform 
imputed_values_MIN = np.expm1(log_values_filled_MIN)
imputed_values_MAX = np.expm1(log_values_filled_MAX)
imputed_values_MODE = np.expm1(log_values_filled_MODE)

#Replace in original dataset
df["MIN_imputed"] = imputed_values_MIN
df["MAX_imputed"] = imputed_values_MAX
df["MODE_imputed"] = imputed_values_MODE
## Note : 
## log transform:
1. Mathematically log transformation looks like:
    y=log(x+1)
2. In Numpy,
    y = np.log1p(x)   # log(x+1) to handle zeros in input. log(0) is undefined. 
## Back transform : 
1. Mathematically, x = e^y - 1
2. x_back = np.expm1(y)   # exp(y) - 1
3. np.expm1(x) is a NumPy function that computes: expm1(x) = e^x - 1

# Code to impute with Scikit Imputer
#load the data into DataFrame
df = pd.read_csv("procedures_new.csv")

#creating data with numerical attributes only as median can be calculated on num attributes
#dropping categorical attributes
df_numerical = df.drop(columns= ['CODE','COMMENTS'], axis =1)

#Create SimpleImputer instance
imputer = SimpleImputer(strategy='median')

#fit the imputer instance to the training data using fit() method
imputer.fit(df_numerical)

#imputer has computed median for MIN,MAX,MODE attributes
print(imputer.statistics_)
print(df_numerical.median().values)

#Trained imputer to transfom training data
X = imputer.transform(df_numerical) # Reruens numpy array containing transformed MIN,MAX,MODE features

#Loading above returned numpy array to pandas DataFrame and reuse the original column labels from DataFrame df
df_new = pd.DataFrame(X,columns = df_numerical.columns,index= df_numerical.index)
print(df_new)

## 2. Random Sampling: Splitting dataset into train set and test using random sampling
df = pd.read_csv("procedures_new.csv")
train_set,test_set = train_test_split(df,test_size= 0.2,random_state= 42)
print(len(train_set))
print(len(test_set))
'''
## 3. Stratified Sampling: Splitting dataset into train set and test using Stratified sampling
## To reduce no of strata for 'CODE' column and each stratum should be large enough and to have
## sufficient number of instances for each stratum,pd.cut() is used to divide CODE attribute with 
## categories and group each code into one or the other category.
df = pd.read_csv("procedures_new.csv")
df.reset_index(drop=True, inplace=True)
df.index = range(0,len(df))
df['Bins'] = pd.cut(df.index,bins =5,labels=[1,2,3,4,5])
print(df)
X = df.drop(columns =['CODE','COMMENTS','Bins'],axis=1) #features
y = df['Bins']  # target variable/strata 
#print(df['Bins'].value_counts())
#df['Bins'].hist()
#plt.savefig("Histogram for CODE Categories.png")

### Stratified Sampling using scikit
## initalize stratifiedshufflesplit
split_init = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42) 

#create train & test indices which has to be looped to get actual data
#train_index and test_index are NumPy arrays containing integer indices
#representing the rows for the training and testing sets, respectively.
for train_index,test_index in split_init.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]    
#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)
print("Train class distribution:\n", y_train.value_counts(normalize=True))
print("Test class distribution:\n", y_test.value_counts(normalize=True))