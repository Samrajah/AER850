
import pandas as pd
# from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Step 1: Reading the data file
file_path = "Project 1 Data.csv"

df = pd.read_csv(file_path)

#Step 2: Data Visulization
from mpl_toolkits.mplot3d import Axes3D
summary_stats = df.describe()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['X'], df['Y'], df['Z'], c='b', marker='o', label='Data Points')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


ax.set_title('Coordinate Location')


plt.show()

from sklearn.model_selection import train_test_split
x=df[['X','Y','Z']]
y=df['Step']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
clf=LinearRegression()
clf.fit(X_train,y_train)
clf.predict(X_test)
clf.score(X_test,y_test)

