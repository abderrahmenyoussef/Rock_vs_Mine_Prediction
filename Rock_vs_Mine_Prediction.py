

import numpy as np # used for arrays 
import pandas as pd # used for loading our data and numbers into data frames (good tables)
from sklearn.model_selection import train_test_split # function to split our data into test data and training data so we don't need to do it manually 
from sklearn.linear_model import LogisticRegression # import our logetic regression model 
from sklearn.metrics import accuracy_score # this is used to find the accuracy of our model 

# **************Data collection and data processing*****************

# Loading the dataset to a pandas dataframe

sonar_data = pd.read_csv(r'C:\Users\abdou\Desktop\Machine_Learning_Projects\Sonar_rock_vs_mine\sonar_data.csv', header=None) # None because our data has no headers 
sonar_data.head()  # this function displays the first 5 rows of our data set 

# number of rows and columns 

sonar_data.shape

# quick overview of your data (statistics measures of the data)

sonar_data.describe()

#to know how many rocks and mines exist 

sonar_data[60].value_counts() #the values must be close so the model can have a good accuracy and more we have data the model is gonna be better

#mean value for each column for rock and mine

sonar_data.groupby(60).mean()

#sperate the data (values) and the labels (rock or mine)

X=sonar_data.drop(columns=60,axis=1) #axis = 1 for column and 0 for row // we are stocking here all the values except the last column 
Y=sonar_data[60] # stocking the 60th column that contains rocks and mines 

print(X)
print(Y)

# we are going to split the data into training nad test data
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.1 , stratify=Y, random_state=1) #test_size = 0.1 means we want to have 10% of the data to be test data // stratify=Y means our data will be splitted based on the number of rocks and mines so we can have almost an equal number of rocks and mines for the test data and training data //  random_state=1 to split the data in a particular order 

# see how many test data and training data 
print(X.shape , X_train.shape , X_test.shape )

print(X_train)
print(Y_train)

# Model training -----> logestic regression model 

model = LogisticRegression() # load this logesticregression function into the variable model 

# training the logestic regression model with training data 

model.fit(X_train,Y_train)

# Model evalution 
# accuracy on the training data ( )>70% it means good and it depends on the amount of data we use  )
X_train_prediction = model.predict(X_train) 

training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

print("Acuuracy on training data : ",training_data_accuracy) #we got 83.4%

#acucuracy on the test data 
X_test_prediction = model.predict(X_test) 

test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

print("Acuuracy on test data : ",test_data_accuracy) # we got 76% 

# at the end we got a trained logestic regression model 

# now we are gonna make a predective system that can predict wether the object is rock or mine using the sonar data

input_data=(0.0158,0.0239,0.0150,0.0494,0.0988,0.1425,0.1463,0.1219,0.1697,0.1923,0.2361,0.2719,0.3049,0.2986,0.2226,0.1745,0.2459,0.3100,0.3572,0.4283,0.4268,0.3735,0.4585,0.6094,0.7221,0.7595,0.8706,1.0000,0.9815,0.7187,0.5848,0.4192,0.3756,0.3263,0.1944,0.1394,0.1670,0.1275,0.1666,0.2574,0.2258,0.2777,0.1613,0.1335,0.1976,0.1234,0.1554,0.1057,0.0490,0.0097,0.0223,0.0121,0.0108,0.0057,0.0028,0.0079,0.0034,0.0046,0.0022,0.0021)
#changing the input data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predecting for one instance so the model want be confused 

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # making the input data into 2D array format (1 means one column)

predection = model.predict(input_data_reshaped) 

print(predection)

if predection[0] =="R" :
    print("The object is a Rock")
else :
    print ("The object is a Mine")








