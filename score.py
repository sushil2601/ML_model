import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

score = pd.read_csv('/home/sushil/Documents/Data_sets/student_scores.csv')

score.head()

score.describe()

score.info()

score.shape

score.columns

score.corr()

plt.hist(score.Scores)

sns.boxplot(score.Scores)

plt.hist(score.Hours)

sns.boxplot(score.Hours)

sns.scatterplot(x = "Hours" , y = "Scores" , data = score)

sns.heatmap(score)

x = score.iloc[:, :-1].values

x

y = score.iloc[:,1].values

y

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

model = LinearRegression()

model.fit(x_train,y_train)

model.score(x_train,y_train)

print(model.intercept_)   

print(model.coef_)

y_pred = model.predict(x_test)

y_pred

model.score(x_test,y_test)

df = pd.DataFrame({'Actual' :y_test,'predicted' :y_pred})

df

plt.scatter(x = df['Actual'], y = df['predicted']);plt.xlabel("Pedicted");plt.ylabel("Actual")

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

meanAbsoluteError = mean_absolute_error(y_test, y_pred)

meanSquaredError = mean_squared_error(y_test, y_pred)

meanAbsoluteError

meanSquaredError

rootMeanSquaredError = np.sqrt(meanSquaredError)

rootMeanSquaredError

