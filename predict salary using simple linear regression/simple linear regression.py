import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv("Salary_Data.csv")
data

'''url="https://raw.githubusercontent.com/krishnaik06/simple-Linear-Regression/master/Salary_Data.csv"
c=pd.read_csv(url)
c''' #to read online csv file

#spliting data int dependent and independent variable

x=data.iloc[:,:-1].values
x
y=data.iloc[:,1].values
y

#divind data  into traing and test data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


#implementing simple linear regression

from sklearn.linear_model import LinearRegression

simple=LinearRegression()
simple.fit(X_train,y_train)

#predict value

y_predict=simple.predict(X_test)


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, simple.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#test result with complete data as training set

simple2=LinearRegression()
simple2.fit(x,y)

#predicting y that is dependent variable

y_p=simple2.predict(x)


print(np.mean(np.abs((y_predict-y_test)/y_test))*100)


error=np.absolute(np.subtract(y_test,y_predict))
error
for i in error:
    print("|E| :",i)
    i=i+1



#mape solution 2
print(np.mean(np.abs(y_test-y_predict)/y_test)*100)
