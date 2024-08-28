import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

#!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
auto_mpg = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets

tree = DecisionTree() 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#X_train, y_train
y_train= pd.Series(y_train.squeeze())
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
y_test= pd.Series(y_test.squeeze())
y_pred= pd.Series(y_pred)

tree.plot("Q3a_tree")
# Ensure indices of y_pred and y_test match
y_pred = y_pred.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
# Now calculate RMSE
print("RMSE: ", rmse(y_pred, y_test))

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn import tree 
import matplotlib.pyplot as plt

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Root Mean Squared Error: {mse**0.5}")

# Plot the tree
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, rounded=True)
plt.title("Decision Tree Regressor")
plt.savefig("Q3b_tree.png")
print(f"Depth of the decision tree: {model.get_depth()}")
plt.close()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn import tree 
import matplotlib.pyplot as plt

model = DecisionTreeRegressor(max_depth=5, random_state=42) # using a tree of same max_depth as our custom decision tree
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Root Mean Squared Error: {mse**0.5}")

# Plot the tree
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, rounded=True)
plt.title("Decision Tree Regressor")
plt.savefig("Q3b_tree(depth=5).png")
print(f"Depth of the decision tree: {model.get_depth()}")
plt.close()