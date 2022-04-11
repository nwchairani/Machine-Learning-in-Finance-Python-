"""
Machine Learning in Finance 
@author: Novia Widya Chairani
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
#import machine learning packages
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix


#set dates
start_date = dt.datetime(2010,1,1)
end_date = dt.datetime(2019,12,31)

#set lags
lags=2

#get the stock data from Yahoo Finance
df = web.DataReader('C',data_source='yahoo',start=start_date,end=end_date)

#create a new dataframe
#we want to use additional features: lagged returns...today's returns, yesterday's returns etc
tslag = pd.DataFrame(index=df.index)
tslag["Today"] = df["Adj Close"]

# Create the shifted lag series of prior trading period close values
range(0, lags)
for i in range(0, lags):
    tslag["Lag%s" % str(i+1)] = df["Adj Close"].shift(i+1)

#create the returns DataFrame
dfret = pd.DataFrame(index=tslag.index)
dfret["Today"] = tslag["Today"].pct_change()

#create the lagged percentage returns columns
for i in range(0, lags):
    dfret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()
        
#because of the shifts there are NaN values ... we want to get rid of those NaNs
dfret.drop(dfret.index[:4], inplace=True)

#"Direction" column (+1 or -1) indicating an up/down day (0 indicates daily non mover)
dfret["Direction"] = np.sign(dfret["Today"])

#Replace where nonmover with down day (-1)
dfret["Direction"]=np.where(dfret["Direction"]==0, -1, dfret["Direction"] ) 

# Use the prior two days of returns as predictor 
# values, with todays return as a continuous response
x = dfret[["Lag1"]]
y = dfret[["Today"]]

"""
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
"""

#Alternative test/train split
# The test data is split into two parts: Before and after 1st Jan 2018.
start_test = dt.datetime(2018,1,1)

# Create training and test sets
x_train = x[x.index < start_test]
x_test = x[x.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]


####################################
#Regression
####################################

#we use Decision Trees as the machine learning model
model=DecisionTreeRegressor(max_depth = 10)
#train the model on the training set
results=model.fit(x_train, y_train)

plt.figure(figsize=(22,16))
plot_tree(results, filled=True)

#make an array of predictions on the test set
y_pred = model.predict(x_test)
model.score(x_test, y_test)
#predict an example
x_example=[[0.01]]
yhat=model.predict(x_example)


#####################################
#Classification
#####################################

#plot log2 function (the measure of entropy)
plt.figure()
plt.plot(np.linspace(0.01,1),np.log2(np.linspace(0.01,1)))
plt.xlabel("P(x)")
plt.ylabel("log2(P(x))")
plt.show()

# Use the prior two days of returns as predictor 
# values, with direction as the discrete response
x = dfret[["Lag1","Lag2"]]
y = dfret["Direction"]

#Alternative test/train split
# The test data is split into two parts: Before and after 1st Jan 2018.
start_test = dt.datetime(2018,1,1)

# Create training and test sets
x_train = x[x.index < start_test]
x_test = x[x.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]


#we use Decision Trees as the machine learning model
model=DecisionTreeClassifier(criterion = 'entropy', max_depth = 19)
#train the model on the training set
results=model.fit(x_train, y_train)

#make an array of predictions on the test set
y_pred = model.predict(x_test)

#predict an example
x_example=[[0.01,0.01]]
yhat=model.predict(x_example)

#output the hit-rate and the confusion matrix for the model
print("Confusion matrix: \n%s" % confusion_matrix(y_pred, y_test))
print("Accuracy of decision tree model on test data: %0.3f" % model.score(x_test, y_test))

#plot decision tree
dt_feature_names = list(x.columns)
dt_target_names = ['Sell','Buy'] #dt_target_names = [str(s) for s in y.unique()]
plt.figure(figsize=(22,16))
plot_tree(results, filled=True, feature_names=dt_feature_names, class_names=dt_target_names)
plt.show()

#pruning (choosing max_depth parameter using 5 fold cross validation)
depth = []
for i in range(1,20):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=i)
    # Perform 5-fold cross validation k=5
    scores = cross_val_score(estimator=model, X=x_train, y=y_train, cv=5)
    depth.append((scores.mean(),i))
    
print(max(depth))



#####################################################################
#Machine learning play around
#####################################################################


# Create a random dataset
rng = np.random.RandomState(1)
x = (5 * rng.rand(80, 1))
y = np.sin(x)
y[::5] += 3 * (0.5 - rng.rand(16,1))

plt.figure()
plt.scatter(x, y, s=20, edgecolor="blue",
            c="lightblue", label="data")

# Fit regression model
model1=DecisionTreeRegressor(max_depth=1)
results=model1.fit(x, y)

x_test=np.linspace(0.0, 5.0, num=80)
x_test
x_test=x_test.reshape(-1,1)
x_test
y1 = model1.predict(x_test)

# Plot the results
plt.figure()
plt.scatter(x, y, s=20, edgecolor="blue",
            c="lightblue", label="data")
plt.plot(x_test, y1, color="red",
         label="max_depth=1", linewidth=2)
#plt.axvline(3.373, color='red', linestyle='dashed', linewidth=1)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()

#plot decision tree
plt.figure(figsize=(22,16))
plot_tree(results, filled=True, impurity=False)

######
#Ensemble
######
ensemble_model=BaggingRegressor(model1, n_estimators=100, random_state=0)
ensemble_model.fit(x,y)
y_pred = ensemble_model.predict(x_test)
#print("Confusion matrix: \n%s" % confusion_matrix(y_pred, y_test))
print("Accuracy of bagging model on test data: %0.3f" % ensemble_model.score(x,y))
