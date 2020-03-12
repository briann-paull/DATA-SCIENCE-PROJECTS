# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#using linear reg
from sklearn.linear_model import LinearRegression
linear_reg1=LinearRegression()
linear_reg1.fit(x,y)

#fitting polynomailfeature
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
x_poly=poly_reg.fit_transform(x)

#using polynimal feature
linear_reg2=LinearRegression()
linear_reg2.fit(x_poly,y)

#visualizing linear reg
plt.scatter(x,y,color="red")
plt.plot(x,linear_reg1.predict(x),color="blue")
plt.show()

#visualizing polynoial reg
plt.scatter(x,y,color="red")
plt.plot(x,linear_reg2.predict(x_poly),color="green")
plt.show()

#predicted value using linear reg
linear_pred=linear_reg1.predict(x)
linear_pred

#predicted value using polynomail reg
poly_pred=linear_reg2.predict(x_poly)
poly_pred


#visualizing difference between polynomial reg and linear reg

fig, ax = plt.subplots()
plt.scatter(x,y,color="red")
# Using set_dashes() to modify dashing of an existing line
line1, = ax.plot(linear_pred, label='Linear')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
# Using plot(..., dashes=...) to set the dashing when creating a line
line2, = ax.plot(poly_pred, dashes=[6, 2], label='polynomial')
ax.legend()
plt.show()

#to check accuracy
from sklearn.metrics import r2_score

score1=r2_score(y,linear_pred)
score2=r2_score(y,poly_pred)



from sklearn.metrics import r2_score
print("accuracy using linear regression = ",score1,
      "accuracy using polynomial regression = ",score2)
