import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns


print("packages imported")

print(("_")*50)



data=pd.read_csv("red.csv") #imported data
print(data.head(5)) #to read head

print(("_")*50)

discription=data.info()

print(data.isnull().any()) #to see null value

print(data.describe()) #summary of data

print(("_")*50)



Quality_unique_value=data["quality"].unique() #ot find out unique value of quality

number=data.quality.value_counts().sort_index() #sort index for sorting in index order...script for values as per variable

print(("_")*50)


condition=[(data["quality"] >=7),
           (data["quality"] <=4)
        ]
rating=["best","ok"]
data['rating']=np.select(condition,rating,default="fine")  #assigning conditin as strings
number2=data.rating.value_counts()
number2


sns.heatmap(data.corr()) #crealted heatmap for variable interdependence
plt.show()  

correlation=data.corr()
correlation['quality'].sort_values(ascending=True) #sort correletion in order
print(correlation)

dataa=data.replace(["fine","ok"],np.NaN)
dataa 

best_data=dataa.dropna()
best_data #i have removed best and ok wine to select best and top 5 fine

#sns.countplot(x="quality",data=best_data)


#pd.plotting.scatter_matrix(best_data, alpha = 0.3, figsize = (40,40), diagonal = 'kde'); #to see linearity of variable

# we found top mosteffect variable towards quality so we will cross check and see what effects more

qf1 = data[['density', 'quality']] 
#here we have used data have larger picture on variable effecting quality

fig, axs = plt.subplots(ncols=1,figsize=(10,6))
sns.barplot(x='quality', y='density', data=qf1, ax=axs)
plt.title('density VS vquality')
plt.tight_layout()
plt.show()
plt.gcf().clear() #creating bar

qf1 = data[['pH', 'quality']]

fig, axs = plt.subplots(ncols=1,figsize=(10,6))
sns.barplot(x='quality', y='pH', data=qf1, ax=axs)
plt.tight_layout()
plt.show()
plt.gcf().clear()

qf1 = data[['sulphates', 'quality']]

fig, axs = plt.subplots(ncols=1,figsize=(10,6))
sns.barplot(x='quality', y='sulphates', data=qf1, ax=axs)
plt.tight_layout()
plt.show()
plt.gcf().clear()

qf1 = data[['alcohol', 'quality']]

fig, axs = plt.subplots(ncols=1,figsize=(10,6))
sns.barplot(x='quality', y='alcohol', data=qf1, ax=axs)
plt.tight_layout()
plt.show()
plt.gcf().clear()



new_data=best_data.loc[:,["pH","sulphates","alcohol","quality","rating"]] #DENSITY ALMOST HAVE NO EFFECT SO I REMOVED IT
number1=new_data.max()


print("wine with below properties is the best wine :")
print(number1)


print("")
print("")

print("  top 5 wines ")

top5=new_data.nlargest(5,"alcohol",keep="last")
print(top5)

print("")
print("")

print("  bottom 5 wines ")

bottom5=data.nsmallest(5, ['alcohol','quality'],keep="last")

print(bottom5)

print("")
print("")

print("  bottom =")


bottom=data.nsmallest(1, ['alcohol','quality'],keep="last") #CREATING DATAFRAME
bottom=pd.DataFrame(bottom)
print(bottom)

print("")
print("")

print("  top =")

top=best_data.nlargest(1,["pH","alcohol","quality"],keep="last")#CREATING DATAFRAME
top=pd.DataFrame(top)
print(top)


df = pd.concat([top, bottom], axis=0)#.reset_index() TO CONACAT
df


sns.heatmap(df.corr(),vmin=0, vmax=1) #to show non linearity




