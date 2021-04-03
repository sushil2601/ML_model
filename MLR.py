import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/home/sushil/Desktop/50_Startups.csv')

df.head()

X = df.iloc[:,:-1]

X.head()

y = df.iloc[:,4]

y.head()

states = pd.get_dummies(X,drop_first = True)

states.head()

X.head()

X = X.drop('State',axis = 1)

X.head()

X = pd.concat([X,states],axis = 1)

X.head()

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

regression = LinearRegression()

regression.fit(X_train,y_train)

y_pred = regression.predict(X_test)

from sklearn.metrics import r2_score

score = r2_score(y_test,y_pred)

print(score)

