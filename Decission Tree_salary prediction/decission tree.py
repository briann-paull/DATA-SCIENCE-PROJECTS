#getting packages


"""Terms to remember"""
#entropy shoould be low...entropy tells us features which provide pure data
#gini inpurity is nothing but impurity in sample


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#readind dataset
data=pd.read_csv("sal.csv")
print(data.info())
print(data.head(5))



#splitting data into x and y
input=data.drop("salary_more_then_100k",axis="columns")
target=data["salary_more_then_100k"]


#label encoding categorical data
from sklearn.preprocessing import LabelEncoder
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()

#fitting encoding to data
input["company_n"]=le_company.fit_transform(input["company"])
input["job_n"]=le_job.fit_transform(input["job"])
input["degree_n"]=le_degree.fit_transform(input["degree"])


#droping old column
input_n=input.drop(["company","job","degree"],axis="columns")
input_n


#creating tree model
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(input_n,target)


#score
model.score(input_n,target)
print(model.score(input_n,target))

#predicting output
y_pred=model.predict(input_n)
