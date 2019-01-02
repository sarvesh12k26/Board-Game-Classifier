import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split

games=pandas.read_csv('games.csv')
print(games.columns)
print(games.shape)

#Plot Histogram wrt average rating
plt.hist(games['average_rating'])
plt.show()

#Check games with user ratings as 0
print(games[games['average_rating']==0].iloc[0])
#Check games with user ratings not 0
print(games[games['average_rating']!=0].iloc[0])

#Remove games with rating as 0
games = games[games['users_rated']>0]
#Remove rows with missing values
games=games.dropna(axis=0)
plt.hist(games['average_rating'])
plt.show()

#Correalation matrix
corrmat=games.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True)
plt.show()

#Data Preprocessing
#get all columns from dataframe
columns=games.columns.tolist()
#Filter unwanted ones
columns=[c for c in columns if c not in ['bayes_average_rating','average_rating','type','name','id']]
#Dependent feature
target='average_rating'

#print(train[columns])

#Training and Test datasets
from sklearn.model_selection import train_test_split
train=games.sample(frac=0.8,random_state=1)
#get rows not in training set
test=games.loc[~games.index.isin(train.index)]
print(train.shape)
print(test.shape)

#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linearregressor=LinearRegression()
linearregressor.fit(train[columns],train[target])

predictions=linearregressor.predict(test[columns])
mean_squared_error(predictions,test[target])

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rfregressor=RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)
rfregressor.fit(train[columns],train[target])

predictions=rfregressor.predict(test[columns])
mean_squared_error(predictions,test[target])

