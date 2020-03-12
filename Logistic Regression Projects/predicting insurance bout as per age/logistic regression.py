#importing packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing data and analyzing data

url="https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/insurance_data.csv"
data=pd.read_csv(url)
data
data.info()

#visualizing data
plt.scatter(data.age,data.bought_insurance,color="red")
plt.show()

#spliting data into dependent and independent variable

x=data.iloc[:,:1]
y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


#creating logistic regression model

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model=model.fit(X_train,y_train)

y_pred=model.predict(X_test)

#cChecking accuracy
print("Accuracy:")
print(model.score(X_test,y_pred))


