import os
os.getcwd()
os.chdir("C://Users//Nagnanamus//Desktop//vamsi//insurance")
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv",sep=',')
print(data)

df = pd.DataFrame(data)
print(df)

df.head(5)

df.tail(5)

df.columns

df.dtypes

miss_val_perc = (df.isna().sum()/len(df))*100
miss_val_perc

cat_cols=df.select_dtypes(include=['object']).columns
cat_cols


le = LabelEncoder()
df.loc[:,['sex', 'smoker', 'region']] = df.loc[:,['sex', 'smoker', 'region']].apply(le.fit_transform)

x = df.drop(['charges'],axis=1)
y = df.charges

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
lr_score = r2_score(y_test,y_pred)
lr_mse = mean_squared_error(y_test,y_pred)

print(lr_score)
print(lr_mse)

plt.scatter(y_test,y_pred,color="red",linewidths=2)
plt.show()

from sklearn.metrics import PredictionErrorDisplay
PredictionErrorDisplay.from_predictions(y_test,y_pred= y_pred, kind= "actual_vs_predicted")
plt.show()


import pickle
pickle.dump(lr,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
model.predict(x_test)