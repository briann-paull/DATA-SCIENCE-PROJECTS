#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading predefined data
from sklearn.datasets import load_iris
iris=load_iris()
print(" ")

#exploring variables
print(dir(iris))
print(" ")
print(iris.feature_names)
print(" ")
print(iris.target)
print(" ")
print(" ")


pd=pd.DataFrame(iris.data,columns=iris.feature_names)
pd
pd["target"]=iris.target

print(pd[pd.target==1].head()) #tosee from where the particular data start

pd['flower_name'] =pd.target.apply(lambda x: iris.target_names[x])


pd0=pd[pd.target==0]  #to print columns based on spcific target
pd1=pd[pd.target==1]
pd2=pd[pd.target==2]


#visualizig difference between sepal and petals pf particular data
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(pd0['sepal length (cm)'], pd0['sepal width (cm)'],color="green",marker='+')
plt.scatter(pd1['sepal length (cm)'], pd1['sepal width (cm)'],color="blue",marker='.')
plt.show()

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(pd0['petal length (cm)'], pd0['petal width (cm)'],color="green",marker='+')
plt.scatter(pd1['petal length (cm)'], pd1['petal width (cm)'],color="blue",marker='.')
plt.show()



x=pd.drop(["target","flower_name"],axis="columns")
y=pd["target"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.svm import SVC

model=SVC()
model.fit(x_train,y_train)

print(model.score(x_test,y_test))

y_pred=model.predict(x_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


