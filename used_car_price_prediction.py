#!/usr/bin/env python
# coding: utf-8



# # USED CAR PREDICTION PRICE

# <img src="https://www.marketingdonut.co.uk/sites/default/files/styles/landing_pages_lists/public/usedcardealer1.jpg?itok=lSSEdwpY">

# # Problem definition
# 

# This is the first step of machine learning life cycle.Here we analyse what kind of problem is, how to solve it. 
# So for this project we are using a car dataset, where we want to predict the selling price of car based on its certain features.
# Since we need to find the real value, with real calculation, therefore this problem is regression problem. 
# We will be using regression machine learning algorithms to solve this problem.

# In[2]:


#loading required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Gathering

# In[3]:


#I have used the raw link for the csv file from my github repositiory.
#url = 'https://raw.githubusercontent.com/SamikshaBhavsar/ML-end_to_end_project/main/car_price_project/car_dataset.csv'
url='car_dataset.csv'
#read this file with the help of pandas.
dataset = pd.read_csv(url)

#if you have already downloaded the csv file into our project folder then use :
# dataset = pd.read_csv("car_dataset.csv")

#print first five rows of the dataset
dataset.head()


# # Data Preparation

# In[4]:


#checking no. of rows and columns in dataset
dataset.shape


# This dataset contains 301 rows and 9 columns

# In[5]:


#Checking the data type of columns.
#this step is important because sometimes dataset may contain wrong datatype of the feature.
dataset.info()


# Good! every data type is correctly mentioned. We need not to make any changes.

# In[6]:


#check statistical summary of all the columns with numerical values.
dataset.describe()


# In[7]:


#check if there is any missing value in the dataset
dataset.isnull().sum()


# There are no missing values in the dataset

# # Feature Engineering

# In[8]:


#adding a column with the current year
dataset['Current_Year']=2020
dataset.head(5)


# In[9]:


#creating a new column which will be age of vehicles; new feature
dataset['Vehicle_Age']=dataset['Current_Year'] - dataset['Year']
dataset.head(5)


# In[10]:


#getting dummies for these columns with help of pandas library
dataset=pd.get_dummies(dataset,columns=['Fuel_Type','Transmission','Seller_Type'],drop_first=True)

#dropping the columns which are redundant and irrelevant
dataset.drop(columns=['Year'],inplace=True)
dataset.drop(columns=['Current_Year'],inplace=True)
dataset.drop(columns=['Car_Name'],inplace=True)

#check out the dataset with new changes
dataset.head()


# <ul>Fuel_Type feature:
#     <li>Fuel is Petrol if Fuel_type_diesel = 0 ,Fuel_Type_Petrol = 1</li>
#     <li>Fuel is Diesel if Fuel_type_diesel = 1 ,Fuel_Type_Petrol = 0</li>
#     <li>Fuel is cng if Fuel_type_diesel = 0 ,Fuel_Type_Petrol = 0</li>
#    </ul>
# <ul>Transmission feature:
#     <li>transmission is manual if Transmission_Manual = 1</li> 
#     <li>transmission is automatic if Transmission_Manual = 0</li></ul>
# <ul>Seller_Type feature:
#     <li>Seller_Type is Individual if Seller_Type_Individual = 1 </li> 
#     <li>Seller_Type is dealer if Seller_Type_Individual = 0</li> </ul>
#     
# 

# ### Pairplot

# In[11]:


#to see pairwise relationships on our dataset we will check pairplot from seaborn library
sns.pairplot(dataset)


# ### Heat map

# In[12]:


#create correlation matrix
correlations = dataset.corr()
indx=correlations.index

#plot this correlation for clear visualisation
plt.figure(figsize=(26,22))
#annot = True , dsiplays text over the cells.
#cmap = "YlGnBu" is nothing but adjustment of colors for our heatmap
sns.heatmap(dataset[indx].corr(),annot=True,cmap="YlGnBu")
#amount of darkness shows how our features are correalated with each other 


# #### I have skipped the EDA part as the main idea is to create the ml model.
# #### Try to do some visualizations, in order to understand the features of this dataset.

# ### Features and target variable

# In[13]:


# taking all the features except "selling price"
X=dataset.iloc[:,1:]
# taking "selling price" as y , as it is our target variable
y=dataset.iloc[:,0]


# ### Feature Importance

# In[14]:


#checking and comparing the importance of features
from sklearn.ensemble import ExtraTreesRegressor
#creating object
model = ExtraTreesRegressor()
#fit the model
model.fit(X,y)

print(model.feature_importances_)


# In[15]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# considering top 5 important features
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# ### Splitting data into training and testing

# In[16]:


#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# # Fitting and evaluating different models
# Here I am using three models :
# 1. Linear Regression
# 2. Decision Tree
# 3. Random forest Regressor
# 
# I will fit these models and then choose one with the better accuracy.
# You can use any regression model as per your choice.

# ## Linear Regression Model

# In[17]:


from sklearn.linear_model import LinearRegression
#creating object for linear regression
reg=LinearRegression()
#fitting the linear regression model
reg.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg.predict(X_test)

#metrics
from sklearn import metrics
#print mean absolute error
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
#print mean squared error
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
#print the root mean squared error
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print R2 metrics score
R2 = metrics.r2_score(y_test,y_pred)
print('R2:',R2)


# ## Decision tree Model

# In[18]:


from sklearn.tree import DecisionTreeRegressor

#creating object for Decision tree
tree = DecisionTreeRegressor()

#fitting the decision tree model
tree.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = tree.predict(X_test)

#print errors
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
R2 = metrics.r2_score(y_test,y_pred)
print('R2:',R2)


# ## Random Forest Model

# In[19]:


from sklearn.ensemble import RandomForestRegressor

#creating object for Random forest regressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

#fitting the rf model
rf.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = rf.predict(X_test)

#print errors
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
R2 = metrics.r2_score(y_test,y_pred)
print('R2:',R2)


# #### We want our R2 score to be maximum and other errors to be minimum for better results

# ### Random forest regressor is giving better results. therefore we will hypertune this model and then fit, predict.

# # Hyperparamter tuning

# In[20]:


#n_estimators = The number of trees in the forest.
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[21]:


from sklearn.model_selection import RandomizedSearchCV

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[22]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[23]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# In[24]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[25]:


#fit the random forest model
rf_random.fit(X_train,y_train)


# In[26]:


#displaying the best parameters
rf_random.best_params_


# In[27]:


rf_random.best_score_


# # Final Predictions

# In[28]:


#predicting against test data
y_pred=rf_random.predict(X_test)
#print the erros
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
R2 = metrics.r2_score(y_test,y_pred)
print('R2:',R2)


# # Save the model

# In[29]:


import pickle
# open a file, where you ant to store the data
file = open('car_price_model_1.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[ ]:




