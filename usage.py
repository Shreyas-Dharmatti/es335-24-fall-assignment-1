"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
# Test case 1
# Real Input and Real Output

import pandas as pd
import numpy as np

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

for i, criteria in enumerate(["information_gain", "gini_index"]):
    tree = DecisionTree(criterion=criteria)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    filename = f"testcase1_tree_{i}_{criteria}"
    tree.plot(filename=filename)
    print("Criteria:", criteria)
    print("RMSE: ", rmse(y_hat, y))
    print("MAE: ", mae(y_hat, y))

# Test case 2
# Real Input and Discrete Output

import pandas as pd
import numpy as np
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for i, criteria in enumerate(["information_gain", "gini_index"]):
    tree = DecisionTree(criterion=criteria)  
    tree.fit(X, y)  
    y_hat = tree.predict(X)  
    filename = f"testcase2_tree_{i}_{criteria}"  
    tree.plot(filename=filename)  
    
    # Print evaluation metrics
    print("Criteria:", criteria)
    print("Accuracy:", accuracy(y_hat, y))
    for cls in y.unique():
        print(f"Precision for class {cls}:", precision(y_hat, y, cls))
        print(f"Recall for class {cls}:", recall(y_hat, y, cls))


# Test case 3
# Discrete Input and Discrete Output

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(P)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")

# One-hot encode the categorical features
X = pd.get_dummies(X, dtype=int)

for i, criteria in enumerate(["information_gain", "gini_index"]):
    tree = DecisionTree(criterion=criteria)  
    tree.fit(X, y)  
    y_hat = tree.predict(X)  
    filename = f"testcase3_tree_{i}_{criteria}"  
    tree.plot(filename=filename)  
    print("Criteria:", criteria)
    print("Accuracy:", accuracy(y_hat, y))
    for cls in y.unique():
        print(f"Precision for class {cls}:", precision(y_hat, y, cls))
        print(f"Recall for class {cls}:", recall(y_hat, y, cls))


# Test case 4
# Discrete Input and Real Output

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(P)})
y = pd.Series(np.random.randn(N))

# One-hot encode the categorical features
X = pd.get_dummies(X, dtype=int)

for i, criteria in enumerate(["information_gain", "gini_index"]):
    tree = DecisionTree(criterion=criteria)  
    tree.fit(X, y)  
    y_hat = tree.predict(X)  
    filename = f"testcase4_tree_{i}_{criteria}"  
    tree.plot(filename=filename)  
    
    # Print evaluation metrics
    print("Criteria:", criteria)
    print("RMSE:", rmse(y_hat, y))
    print("MAE:", mae(y_hat, y))
