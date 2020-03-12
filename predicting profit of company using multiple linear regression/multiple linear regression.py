import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading data

data=pd.read_csv("50.csv")


#analyzing data


data_info=data.info()
describe=data.describe()
heatmap=data.corr()
sns.heatmap(heatmap)


#spliting data

x=data.iloc[:,:-1]
y=data.iloc[:,4]

#if we have just two data we can convert it into lable
# encoding but here we have three so we are using onehot
# encoding

states=pd.get_dummies(x['State'],drop_first=True) #dummies is use to convert categorial features
 
#droping states colum from x to concat new state
x=x.drop(["State"],axis=1)

#concat encoded variables back to list
x=pd.concat([x,states],axis=1)

#spliting data into train and test

from sklearn.model_selection import train_test_split

X_train,X_Test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#creating Model

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting
y_pred=regressor.predict(X_Test)

#to check accuracy
from sklearn.metrics import r2_score

score=r2_score(y_test,y_pred)
score



'''Y_TEST=pd.Series(y_test)
print(Y_TEST)
Y_PRED=pd.Series(y_pred)

new=pd.concat([Y_TEST, Y_PRED], axis=1)
new=new.dropna()


df = new[['Profit', 0]] '''

