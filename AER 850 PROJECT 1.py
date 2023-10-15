
import pandas as pd
# from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Step 1: Reading the data file
file_path = "Project 1 Data.csv"

df = pd.read_csv(file_path)

from sklearn.model_selection import train_test_split
x=df[['X','Y','Z']]
y=df['Step']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
clf=LinearRegression()
clf.fit(X_train,y_train)
clf.predict(X_test)
clf.score(X_test,y_test)

from sklearn.model_selection import StratifiedShuffleSplit
df["Step_cat"] = pd.cut(df["Step"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["Step_cat"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)
strat_train_set = strat_train_set.drop(columns=["Step_cat"], axis = 1)
strat_test_set = strat_test_set.drop(columns=["Step_cat"], axis = 1)