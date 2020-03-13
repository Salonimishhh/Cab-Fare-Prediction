#!/usr/bin/env python
# coding: utf-8

# # Cab Fare Prediction
# 
# This is to predict the fare amount for a cab ride in the city based on historical data from a pilot project.

# In[1]:


#Importing required libraries
import os #getting access to input files
import pandas as pd # Importing pandas for performing EDA
import numpy as np  # Importing numpy for Linear Algebric operations
import matplotlib.pyplot as plt # Importing for Data Visualization
import seaborn as sns # Importing for Data Visualization
from collections import Counter 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression #ML algorithm
from sklearn.model_selection import train_test_split #splitting dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn.model_selection import GridSearchCV    

get_ipython().magic(u'matplotlib inline')


# In[2]:


#Setting the working directory

os.chdir("D:/001. Data sc/Project/Cab_Fare_Prediction")
print(os.getcwd())


# In[3]:


#Loading the data:
train  = pd.read_csv("train_cab.csv",na_values={"pickup_datetime":"43"})
test   = pd.read_csv("test.csv")


# # UNDERSTANDING DATA
# 
# Studying the data in order to proceed further with data analysis and prediction of the required variable

# In[4]:


train.head() #checking first five rows of the training dataset


# In[5]:


test.head() #checking first five rows of the test dataset


# In[6]:


print("shape of training data is: ",train.shape) #checking the number of rows and columns in training data
print("shape of test data is: ",test.shape) #checking the number of rows and columns in test data


# In[7]:


train.dtypes #checking the data-types in training dataset


# In[8]:


test.dtypes #checking the data-types in test dataset


# In[9]:


train.describe() 


# In[10]:


test.describe()


# ## Data Cleaning & Missing Value Analysis :

# In[13]:


#Convert fare_amount from object to numeric
train["fare_amount"] = pd.to_numeric(train["fare_amount"],errors = "coerce") 
#Using errors=’coerce’. It will replace all non-numeric values with NaN.


# In[14]:


train.dtypes


# In[15]:


train.shape


# In[16]:


data=train.copy()
#train_data=data.copy()


# In[18]:


#counting the number of missing values for each variable
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls.index.name='Variables'
nulls.columns = ['Count of null']
nulls['Percentage'] = (nulls['Count of null']/train.shape[0])*100
nulls.reset_index(inplace=True)
nulls


# In[19]:


plt.subplots(figsize=(12,5))
sns.barplot(nulls['Variables'],nulls['Percentage'])

#Only two columns contained the missing data and the percentage is quite low hence we will remove the samples which have null values
# In[20]:


train.dropna(subset= ["pickup_datetime"])   #dropping NA values in datetime column


# In[24]:


#removing the samples containing null value
train=train.dropna()


# In[26]:


train.reset_index(drop=True,inplace=True)


# In[27]:


#checking again
train.isnull().sum()


# In[28]:


#new dimension of train data
train.shape


# In[59]:


# Here pickup_datetime variable is in object so we need to change its data type to datetime
train['pickup_datetime'] =  pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')


# In[60]:


### we will saperate the Pickup_datetime column into separate field like year, month, day of the week, etc

train['year'] = train['pickup_datetime'].dt.year
train['Month'] = train['pickup_datetime'].dt.month
train['Date'] = train['pickup_datetime'].dt.day
train['Day'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].dt.hour
train['Minute'] = train['pickup_datetime'].dt.minute


# In[62]:


train.dtypes #Re-checking datatypes after conversion


# In[63]:


test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"],format= "%Y-%m-%d %H:%M:%S UTC")


# In[64]:


### we will saperate the Pickup_datetime column into separate field like year, month, day of the week, etc

test['year'] = test['pickup_datetime'].dt.year
test['Month'] = test['pickup_datetime'].dt.month
test['Date'] = test['pickup_datetime'].dt.day
test['Day'] = test['pickup_datetime'].dt.dayofweek
test['Hour'] = test['pickup_datetime'].dt.hour
test['Minute'] = test['pickup_datetime'].dt.minute


# In[65]:


test.dtypes #Re-checking test datatypes after conversion


# # OUTLIER ANALYSIS
# We would be now plotting the box plots of several continuous variables
# In[66]:


#removing datetime missing values rows
train = train.drop(train[train['pickup_datetime'].isnull()].index, axis=0)
print(train.shape)
print(train['pickup_datetime'].isnull().sum()) #Checking the Datetime Variable 


# In[67]:


train["passenger_count"].describe() #Checking the passenger count variable 


# In[68]:


#We can see maximum number of passanger count is 5345 which is actually not possible.
#So reducing the passenger count to 6 (even if we consider the SUV)
#Also removing the values with passenger count of 0.
#There is one passenger count value of 0.12 which is not possible. Hence we will remove fractional passenger value

train = train.drop(train[train["passenger_count"]> 6 ].index, axis=0)
train = train.drop(train[train["passenger_count"] == 0 ].index, axis=0)
train = train.drop(train[train["passenger_count"] == 0.12 ].index, axis=0)
print(train["passenger_count"].describe())
print(train.shape)
print(train['passenger_count'].isnull().sum())

#removing passanger_count missing values rows
train = train.drop(train[train['passenger_count'].isnull()].index, axis=0)
print(train.shape)
print(train['passenger_count'].isnull().sum())


# In[69]:


##Checking the Fare Amount variable 
print(Counter(train["fare_amount"]<0))
train = train.drop(train[train["fare_amount"]<0].index, axis=0)
print(train.shape)

##make sure there is no negative values in the fare_amount variable column
print(train["fare_amount"].min())

##Also remove the row where fare amount is zero
train = train.drop(train[train["fare_amount"]<1].index, axis=0)
print(train.shape)

## ordering fare in descending to know whether the outliers are present
train["fare_amount"].sort_values(ascending=False)


# In[70]:


#Now we can see that there is a huge difference in 1st 2nd and 3rd position in decending order of fare amount
# so we will remove the rows having fare amounting more that 454 as considering them as outliers

train = train.drop(train[train["fare_amount"]> 454 ].index, axis=0)
print(train.shape)
print(train['fare_amount'].isnull().sum())

# eliminating rows for which value of "fare_amount" is missing
train = train.drop(train[train['fare_amount'].isnull()].index, axis=0)
print(train.shape)
print(train['fare_amount'].isnull().sum())
print(train["fare_amount"].describe())


# In[71]:


#Lattitude----(-90 to 90)
#Longitude----(-180 to 180)

# we need to drop the rows having lattitute and longitute out the range mentioned above

train = train.drop((train[train['pickup_latitude']<-90]).index, axis=0)
train = train.drop((train[train['pickup_latitude']>90]).index, axis=0)
train = train.drop((train[train['pickup_longitude']<-180]).index, axis=0)
train = train.drop((train[train['pickup_longitude']>180]).index, axis=0)
train = train.drop((train[train['dropoff_latitude']<-90]).index, axis=0)
train = train.drop((train[train['dropoff_latitude']>90]).index, axis=0)
train = train.drop((train[train['dropoff_longitude']<-180]).index, axis=0)
train = train.drop((train[train['dropoff_longitude']>180]).index, axis=0)


# In[72]:


print(train.shape)
print(train.isnull().sum())


# In[73]:


print(test.shape)
print(test.isnull().sum())


# In[74]:


##Calculating distance based on the given coordinates :
#As we know that we have given pickup longitute and latitude values and same for drop. 
#So we need to calculate the distance Using the haversine formula and we will create a new variable called distance
from math import radians, cos, sin, asin, sqrt

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km
# 1min 


# In[75]:


#application

train['distance'] = train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)
print("---------------TRAIN--------------")
print(train.head())
test['distance'] = test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)
print("---------------TEST---------------")
print(test.head())


# In[76]:


print("---------------TRAIN--------------")
print(train.nunique())
print("---------------TEST---------------")
print(test.nunique())


# In[77]:


##finding decending order of fare to get to know whether the outliers are presented or not
train['distance'].sort_values(ascending=False)

#### As we can see that top 23 values in the distance variables are very high It means more than 8000 Kms distance they have travelled

#### Also just after 23rd value from the top, the distance goes down to 127, which means these values are showing some outliers. We need to remove these values
# In[79]:


print("----------TRAIN-----------")
print(Counter(train['distance'] == 0))
print(Counter(train['fare_amount'] == 0))
print("----------TEST-----------")
print(Counter(test['distance'] == 0))


# In[80]:


##we will remove the rows whose distance value is zero

train = train.drop(train[train['distance']== 0].index, axis=0)
print(train.shape)

##we will remove the rows whose distance values is very high which is more than 129kms
train = train.drop(train[train['distance'] > 130 ].index, axis=0)
print(train.shape)

print(train.head())

#### We have split the pickup date time variable into varaibles like month, year, day etc. So we dont need to have that pickup_Date variable now. Hence we can drop that, Also distance has been calculated using pickup and drop longitudes and latitudes so we will also drop pickup and drop longitudes and latitudes variables.
# In[81]:


drop = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'Minute']
train = train.drop(drop, axis = 1)
train.head()


# In[82]:


train['passenger_count'] = train['passenger_count'].astype('int64')
train['year'] = train['year'].astype('int64')
train['Month'] = train['Month'].astype('int64')
train['Date'] = train['Date'].astype('int64')
train['Day'] = train['Day'].astype('int64')
train['Hour'] = train['Hour'].astype('int64')

train.dtypes


# In[83]:


drop_test = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'Minute']
test = test.drop(drop_test, axis = 1)
print("--------------HEAD------------")
print(test.head())
print("------------DATA TYPES-------------")
print(test.dtypes)


# # Data Visualization :
Visualization of following:

1. Number of Passengers effects the the fare
2. Pickup date and time effects the fare
3. Day of the week does effects the fare
4. Distance effects the fare
# In[91]:


# Count plot on passenger count
plt.figure(figsize=(15,7))
sns.countplot(x="passenger_count", data=train)


# In[92]:


#Relationship beetween number of passengers and Fare

plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=10)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.show()


# # Observations :
#    By seeing the above plots we can easily conclude that:
# 1. single travelling passengers are most frequent travellers.
# 2. At the sametime we can also conclude that highest Fare are coming from single & double travelling passengers.

# In[93]:


#Relationship between date and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['Date'], y=train['fare_amount'], s=10)
plt.xlabel('Date')
plt.ylabel('Fare')
plt.show()


# In[94]:


plt.figure(figsize=(15,7))
train.groupby(train["Hour"])['Hour'].count().plot(kind="bar")
plt.show()

Lowest cabs at 5 AM and highest at and around 7 PM i.e the office rush hours
# In[95]:


#Relationship between Time and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['Hour'], y=train['fare_amount'], s=10)
plt.xlabel('Hour')
plt.ylabel('Fare')
plt.show()

From the above plot We can observe that the cabs taken at 7 am and 23 Pm are the costliest. Hence we can assume that cabs taken early in morning and late at night are costliest
# In[96]:


#impact of Day on the number of cab rides
plt.figure(figsize=(15,7))
sns.countplot(x="Day", data=train)

Observation : The day of the week does not seem to have much influence on the number of cabs ride
# In[97]:


#Relationships between day and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['Day'], y=train['fare_amount'], s=10)
plt.xlabel('Day')
plt.ylabel('Fare')
plt.show()

The highest fares seem to be on a Sunday, Monday and Thursday, and the low on Wednesday and Saturday. May be due to low demand of the cabs on saturdays the cab fare is low and high demand of cabs on sunday and monday shows the high fare prices
# In[98]:


#Relationship between distance and fare 
plt.figure(figsize=(15,7))
plt.scatter(x = train['distance'],y = train['fare_amount'],c = "g")
plt.xlabel('Distance')
plt.ylabel('Fare')
plt.show()

# It is quite obvious that distance will effect the amount of fare
# # Feature Scaling :

# In[99]:


#Normality check of training data is uniformly distributed or not-

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='green')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# In[101]:


#since skewness of target variable is high, apply log transform to reduce the skewness-
train['fare_amount'] = np.log1p(train['fare_amount'])

#since skewness of distance variable is high, apply log transform to reduce the skewness-
train['distance'] = np.log1p(train['distance'])


# In[102]:


#Normality Re-check to check data is uniformly distributed or not after log transformartion

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='green')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()

Here we can see bell shaped distribution. Hence our continous variables are now normally distributed, we will use not use any Feature Scalling technique. i.e, Normalization or Standarization for our training data
# In[103]:


#Normality check for test data is uniformly distributed or not-

sns.distplot(test['distance'],bins='auto',color='green')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# In[104]:


#since skewness of distance variable is high, apply log transform to reduce the skewness-
test['distance'] = np.log1p(test['distance'])


# In[105]:


#rechecking the distribution for distance
sns.distplot(test['distance'],bins='auto',color='green')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()

As we can see a bell shaped distribution. Hence our continous variables are now normally distributed, we will use not use any Feature Scalling technique. i.e, Normalization or Standarization for our test data
# # Applying ML ALgorithms:

# In[106]:


##train test split for further modelling
X_train, X_test, y_train, y_test = train_test_split( train.iloc[:, train.columns != 'fare_amount'], 
                         train.iloc[:, 0], test_size = 0.20, random_state = 1)


# In[107]:


print(X_train.shape)
print(X_test.shape)


# ### Linear Regression Model

# In[108]:


# Building model on top of training dataset
fit_LR = LinearRegression().fit(X_train , y_train)

#prediction on train data
pred_train_LR = fit_LR.predict(X_train)

#prediction on test data
pred_test_LR = fit_LR.predict(X_test)

##calculating RMSE for test data
RMSE_test_LR = np.sqrt(mean_squared_error(y_test, pred_test_LR))

##calculating RMSE for train data
RMSE_train_LR= np.sqrt(mean_squared_error(y_train, pred_train_LR))

print("Root Mean Squared Error For Training data = "+str(RMSE_train_LR))
print("Root Mean Squared Error For Test data = "+str(RMSE_test_LR))

##calculate R^2 for train data
from sklearn.metrics import r2_score
r2_score(y_train, pred_train_LR)


# In[109]:


##calculate R^2 for test data
r2_score(y_test, pred_test_LR)


# ### Decision tree Model : 

# In[110]:


fit_DT = DecisionTreeRegressor(max_depth = 2).fit(X_train,y_train)

#prediction on train data
pred_train_DT = fit_DT.predict(X_train)

#prediction on test data
pred_test_DT = fit_DT.predict(X_test)

##calculating RMSE for train data
RMSE_train_DT = np.sqrt(mean_squared_error(y_train, pred_train_DT))

##calculating RMSE for test data
RMSE_test_DT = np.sqrt(mean_squared_error(y_test, pred_test_DT))

print("Root Mean Squared Error For Training data = "+str(RMSE_train_DT))
print("Root Mean Squared Error For Test data = "+str(RMSE_test_DT))

## R^2 calculation for train data
r2_score(y_train, pred_train_DT)



# In[111]:


## R^2 calculation for test data
r2_score(y_test, pred_test_DT)


# ### Random Forest Model :

# In[112]:


fit_RF = RandomForestRegressor(n_estimators = 200).fit(X_train,y_train)

#prediction on train data
pred_train_RF = fit_RF.predict(X_train)

#prediction on test data
pred_test_RF = fit_RF.predict(X_test)

##calculating RMSE for train data
RMSE_train_RF = np.sqrt(mean_squared_error(y_train, pred_train_RF))

##calculating RMSE for test data
RMSE_test_RF = np.sqrt(mean_squared_error(y_test, pred_test_RF))

print("Root Mean Squared Error For Training data = "+str(RMSE_train_RF))
print("Root Mean Squared Error For Test data = "+str(RMSE_test_RF))

## calculate R^2 for train data
r2_score(y_train, pred_train_RF)


# In[113]:


#calculate R^2 for test data
r2_score(y_test, pred_test_RF)

