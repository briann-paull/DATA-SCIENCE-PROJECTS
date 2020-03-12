#importing packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing Dataset

from sklearn.datasets import load_digits
digits=load_digits()

#reading content of data
print(dir(digits))
'''output wil have...
1)data will have data of set
2)images will have image of data
3)image will have all images
4)target will have array of output
5)target name will have name of image'''

#print(digits.data[0]) #to se data of first value
#print(digits.target[0:5])

plt.gray()
#for i in range (5):
#    plt.matshow(digits.images[i])#to see mutiple images
plt.matshow(digits.images[0])#to see images

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.1,random_state=2)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
print(model.score(X_test, y_test))


print(model.predict(digits.data[0:5]))


plt.matshow(digits.images[9])
print(digits.target[9])


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

    
    
