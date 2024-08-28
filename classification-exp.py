import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Testing Data')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training vs Testing Data Split')
plt.show()

tree = DecisionTree()  
X_train_df= pd.DataFrame(X_train)
y_train_df= pd.Series(y_train)
y_test_df= pd.Series(y_test)
X_test_df= pd.DataFrame(X_test)
tree.fit(X_train_df, y_train_df)
y_hat = tree.predict(X_test_df)

tree.plot("Q2a_tree")
print("Criteria :", criteria)
print("Accuracy: ", accuracy(y_hat, y_test_df))
for cls in y_test_df.unique():
    print("Precision: ", precision(y_hat, y_test_df, cls))
    print("Recall: ", recall(y_hat, y_test_df, cls))

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# Assuming DecisionTree, accuracy, precision, and recall functions are already defined

# Initialize the KFold with 5 splits
kf = KFold(5)

# Variables for storing the optimal max depth and its corresponding accuracy
opt_max_depth = 0
opt_accuracy = 0

# Get the split data for cross-validation
splt_data = list(kf.split(X_train_df))

# Loop over different max_depth values
for max_depth in range(0, 9):
    avg_acc = 0
    
    # Perform cross-validation
    for train_ind, val_ind in splt_data:
        tree = DecisionTree(max_depth=max_depth)
        tree.fit(X_train_df.iloc[train_ind], y_train_df.iloc[train_ind])
        y_val_hat = tree.predict(X_train_df.iloc[val_ind])
        
        # Convert to NumPy array to avoid index issues
        acc = accuracy(y_val_hat, y_train_df.iloc[val_ind].values)
        avg_acc += acc / 5  # Average accuracy over the 5 folds
    
    # Check if this max_depth is the best so far
    if np.round(opt_accuracy, decimals=10) < np.round(avg_acc, decimals=10):
        opt_max_depth = max_depth
        opt_accuracy = avg_acc

# Output the optimal max depth and corresponding accuracy
print("Optimum Max depth is {}".format(opt_max_depth))
print("The average accuracy over validation data for max_depth {} is:- {}".format(opt_max_depth, opt_accuracy))

# Train the decision tree on the entire training dataset with the optimal max depth
tree = DecisionTree(max_depth=opt_max_depth)
tree.fit(X_train_df, y_train_df)

# Make predictions on the test data
y_hat = tree.predict(X_test_df)

# Output the evaluation metrics
print("After changing max_depth to {}".format(opt_max_depth))
print("Accuracy :- {}".format(accuracy(y_hat, y_test_df.values)))
print("Precision :- ")
print("            For y=1 :- {}".format(precision(y_hat, y_test_df.values, 1)))
print("            For y=0 :- {}".format(precision(y_hat, y_test_df.values, 0)))
print("Recall :- ")
print("            For y=1 :- {}".format(recall(y_hat, y_test_df.values, 1)))
print("            For y=0 :- {}".format(recall(y_hat, y_test_df.values, 0)))


