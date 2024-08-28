"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

from dataclasses import dataclass
from typing import Literal

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index", "mse"]  # criterion will vary based on tree type
    max_depth: int  # The maximum depth the tree can grow to
    tree: object = None  # The actual tree (ClassificationTree, RegressionTree, etc.)
    filename: str = "Decision_tree"

    def __init__(self, criterion='gini_index', max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.filename=filename

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        X_is_real = check_ifreal(pd.concat([X[col] for col in X.columns[1:]], ignore_index=True)) 
        y_is_real = check_ifreal(y) 

        if X_is_real:
            if y_is_real:
                self.tree = RegressionTree(criterion='mse', max_depth=self.max_depth)
            else:
                self.tree = ClassificationTreeRealInput(criterion=self.criterion, max_depth=self.max_depth)
            self.tree.fit(X, y)
        else:  
            if y_is_real:
                self.tree = RegressionTree(criterion='mse', max_depth=self.max_depth)
            else:
                self.tree = ClassificationTree(criterion=self.criterion, max_depth=self.max_depth)
            self.tree.fit(X, y)
            
        

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """
        return self.tree.predict(X)

    def plot(self, filename) -> None:
        """
        Function to plot the tree
        """
        self.tree.plot(filename)