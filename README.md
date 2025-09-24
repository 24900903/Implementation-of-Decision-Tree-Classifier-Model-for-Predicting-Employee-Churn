# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```python
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Harisha .S 
RegisterNumber: 212224230087
import pandas as pd
import numpy as np
df=pd.read_csv("/content/Employee.csv")
print(df.head())

df.info()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()

df["left"].value_counts()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("HARISHA.S")
print("212224230087")
print(y_pred)

from sklearn.metrics import confusion_matrix,classification_report

accuracy=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("accuracy:",accuracy)
print("confusion Matrix:")
print(cm)
print("classification Reprt:")
print(cr)

dt.predict(pd.DataFrame([[0.6,0.9,8,292,6,0,1,2]],columns=["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]))
```
## Output:
<img width="889" height="325" alt="image" src="https://github.com/user-attachments/assets/a56a03cf-0b47-404e-aee1-9e88aa28b7a2" />

<img width="432" height="184" alt="image" src="https://github.com/user-attachments/assets/f691bbcf-b425-4668-bbb4-09f6c646a847" />

<img width="511" height="322" alt="image" src="https://github.com/user-attachments/assets/ac11c651-b5da-4065-9327-259c591f1a63" />

<img width="721" height="411" alt="image" src="https://github.com/user-attachments/assets/e7d8bdf9-a07c-406e-8f7c-4e2c8d6a1a01" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
