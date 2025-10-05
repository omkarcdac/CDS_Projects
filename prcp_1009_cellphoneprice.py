import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('datasets_11167_15520_train.csv')

data

data.head()

data.info()

data.describe()

data.plot(kind='box',subplots=True,layout=(7,3),figsize=(15,20),sharex=False,sharey=False)
plt.tight_layout()
plt.show()

for fc in data.select_dtypes(include='number').columns:
  Q1=data[fc].quantile(0.25)
  Q3=data[fc].quantile(0.75)
  IQR=Q3-Q1
  lower_bond=Q1-1.5*IQR
  upper_bond=Q3+1.5*IQR
  df_removed = data[(data[fc] >= lower_bond) & (data[fc] <= upper_bond)]
  print("Original:\n", data)
print("\nAfter Removing Outliers:\n", df_removed)

data

X=data.iloc[:,:-1]
y=data.price_range

X

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

X_train

X_test

y_train

y_test

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
X_trains=sc.fit_transform(data)

X_trains

X_scaled=pd.DataFrame(X_train,columns=data.columns)

X_scaled

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

y_pred

from sklearn.metrics import accuracy_score,classification_report

accuracy=accuracy_score(y_test,y_pred)

accuracy

classification=classification_report(y_test,y_pred)

print(classification)

