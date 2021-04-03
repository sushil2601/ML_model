import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/home/sushil/Documents/code_ml/advertising.csv')

df.head()

df.shape

df.columns

df.isnull().sum()

X = df.iloc[:,:-1]

X.head()

y = df.iloc[:,3]

y.head()

plt.hist(df['TV'])

plt.boxplot(df.TV)

plt.hist(df.Radio)

plt.boxplot(df.Radio)

plt.hist(df.Newspaper)

plt.boxplot(df.Newspaper)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression

model = smf.ols('Sales~TV+Radio+Newspaper',data = df).fit()

model.params

model.summary()

### model2

model2 = smf.ols('Sales~TV+Radio',data = df).fit()

model2.params

model2.summary()

model3 = LinearRegression()

model3.fit(X_train,y_train)

y_pred = model3.predict(X_test)

y_pred

from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)

print(score)

