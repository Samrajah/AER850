
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

#Step 3: Correlation Analysis
target_variable = 'Step'


correlations = df.corr(method='pearson')[[target_variable]].drop(target_variable)

plt.figure(figsize=(10, 6))
sns.heatmap(correlations, annot=True, cmap='coolwarm', cbar=False)
plt.title('Pearson Correlation with Target Variable')
plt.show()


#Step 4: 
from sklearn.model_selection import train_test_split, GridSearchCV
x=df[['X','Y','Z']]
y=df['Step']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

MLmodels = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

param_grid_logistic = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
param_grid_decision_tree = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
param_grid_random_forest = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]}

param_grids = [param_grid_logistic, param_grid_decision_tree, param_grid_random_forest]

best_MLmodels = {}

for i, MLmodel_name in enumerate(MLmodels.keys()):
    grid_search = GridSearchCV(MLmodels[MLmodel_name], param_grids[i], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_MLmodels[MLmodel_name] = grid_search.best_estimator_
    
for MLmodel_name, MLmodel in best_MLmodels.items():
    from sklearn.metrics import accuracy_score
    y_pred = MLmodel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{MLmodel_name} Accuracy: {accuracy}')


    