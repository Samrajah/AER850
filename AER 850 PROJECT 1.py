
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

#Step 5: 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

metrics = {}

for MLmodel_name, MLmodel in best_MLmodels.items():
    y_pred = MLmodel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    metrics[MLmodel_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

for MLmodel_name, metric in metrics.items():
    print(f"Metrics for {MLmodel_name}:")
    print(f"Accuracy: {metric['Accuracy']:.4f}")
    print(f"Precision: {metric['Precision']:.4f}")
    print(f"Recall: {metric['Recall']:.4f}")
    print(f"F1 Score: {metric['F1 Score']:.4f}")
    print()

chosen_MLmodel_name = "Decision Tree" 
chosen_MLmodel = best_MLmodels[chosen_MLmodel_name]
y_pred = chosen_MLmodel.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#Step 6:
import joblib

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

joblib.dump(decision_tree_model, 'decision_tree_model.joblib')
loaded_model = joblib.load('decision_tree_model.joblib')

new_data_set = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]


Maintenance_Step= loaded_model.predict(new_data_set)

print("The corresponding Maintenance Step is:",Maintenance_Step)










    