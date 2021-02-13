import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/home/sushil/Documents/Assingment/SLR-ASS4/Salary_Data.csv')

df.head()

df.shape

df.describe()

df.corr()

df['Salary'].value_counts().head()

df.isnull().any()

plt.hist(df.Salary)

sns.boxplot(df.Salary)

plt.hist(df.YearsExperience)

plt.boxplot(df.YearsExperience)

sns.scatterplot(data=df, x="YearsExperience", y="Salary",palette="deep",sizes=(20, 200),legend = 'full')

sns.stripplot(x=df["YearsExperience"])

sns.stripplot(y=df['Salary'])

sns.stripplot(x='YearsExperience' , y = 'Salary' , data = df)

sns.violinplot(x="YearsExperience", y="Salary", data=df)

sns.lineplot(data=df, x="YearsExperience", y="Salary")

sns.heatmap(df)

plt.plot(df.YearsExperience,df.Salary,'bo');plt.xlabel("YearsExperience");plt.ylabel("Salary")

from sklearn.linear_model import LinearRegression

import statsmodels.formula.api as smf

model1 = smf.ols('Salary~YearsExperience',data = df).fit()

model1.params

model1.summary()

print(model1.conf_int(0.005))

pred = model1.predict(df.iloc[:,0])

pred.head()

plt.scatter(x = df['YearsExperience'],y = df['Salary'],color ='red');plt.plot(df['YearsExperience'],pred,color = 'black');plt.xlabel('Experience');plt.ylabel('Salary')

model2 = smf.ols('Salary~np.log(YearsExperience)',data = df).fit()

model2.params

model2.summary()

print(model2.conf_int(0.001))

pred2 = model2.predict(df.iloc[:,0])

pred2.head()

plt.scatter(x= df['YearsExperience'] , y = df['Salary'] , color = 'green');plt.plot(df['YearsExperience'],pred2,color = 'red');plt.xlabel('YearsExperience');plt.ylabel('Salary')

model3 = smf.ols('np.log(Salary)~YearsExperience' , data = df).fit()

model3.params

model3.summary()

print(model3.conf_int(0.001))

pred_log= model3.predict(df.iloc[:,0])

pred_log.head()

pred3 = np.exp(pred_log)

pred3.head()

plt.scatter(x = df['YearsExperience'],y = df['Salary'] , color = 'blue');plt.plot(df['YearsExperience'],pred3 , color = 'black');plt.xlabel('YearsExperience');plt.ylabel('Salary')

df['YearsExperience_sq'] = df.YearsExperience*df.YearsExperience

quad_model = smf.ols('Salary~YearsExperience_sq+YearsExperience',data = df).fit()

quad_model.params

quad_model.summary()

quad_model1 = smf.ols('np.log(Salary)~YearsExperience_sq+YearsExperience',data = df).fit()

quad_model1.params

quad_model1.summary()

print(quad_model1.conf_int(0.001))

pred_quad = quad_model.predict(df)

pred_quad.head()

quad_model.conf_int(0.05)

resid = quad_model.resid_pearson

resid

plt.plot(quad_model.resid_pearson,'o');plt.axhline(y = 0,color = 'green');plt.xlabel('observarion number');plt.ylabel('standarized residual')

plt.scatter(x = pred_quad , y = df.Salary);plt.xlabel("Pedicted");plt.ylabel("Actual")

plt.hist(quad_model.resid_pearson)

