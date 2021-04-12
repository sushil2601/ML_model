import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn import datasets

from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

X = iris.data

Y = iris.target

scaler = StandardScaler()

X_std = scaler.fit_transform(X)

clf = LogisticRegression(random_state = 0,multi_class = 'ovr')

clf

model = clf.fit(X_std , Y)

model

new_observation = [[.5,.5,.5,.5]]

model.predict(new_observation)

model.predict_proba(new_observation)

