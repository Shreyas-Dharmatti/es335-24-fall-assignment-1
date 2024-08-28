import pandas as pd
import numpy as np
from collections import Counter
#%config InlineBackend.figure_format = 'retina'
import graphviz
from tree.utils import *

def check_ifreal(y: pd.Series):
    """Check if the target variable is continuous or categorical."""
    if pd.api.types.is_float_dtype(y):
        return True
    if pd.api.types.is_string_dtype(y):
        return False
    if y.dtype.name == 'category':
        return False
    if pd.api.types.is_integer_dtype(y):
        # Explicitly check for categorical dtype
        # Check if the number of unique values is small
        if len(y.unique()) < 10:
            return False
    return True

class RegressionTree:
    def __init__(self, criterion='mse', max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series, depth=0):
        """Recursively build the regression tree"""
        if depth == self.max_depth or len(y) < 2:
            return np.mean(y)

        # Find the best attribute and value to split on
        best_attr, best_value = self.opt_split_attribute(X, y)

        if best_attr is None:
            return np.mean(y)  # Return the mean if no valid split is found

        tree = {best_attr: {}}

        # Split the data based on the best attribute and value
        X_left, y_left, X_right, y_right = self.split_data(X, y, best_attr, best_value)

        # Recursively build the tree
        tree[best_attr]['<= {:.3f}'.format(best_value)] = self.fit(X_left, y_left, depth + 1)
        tree[best_attr]['> {:.3f}'.format(best_value)] = self.fit(X_right, y_right, depth + 1)

        self.tree = tree
        return tree

    def predict_one(self, x: pd.Series):
        """Predict a single sample by traversing the tree"""
        node = self.tree
        while isinstance(node, dict):
            attr = next(iter(node))
            if x[attr] <= float(next(iter(node[attr]))[3:]):
                node = node[attr]['<= {:.3f}'.format(float(next(iter(node[attr]))[3:]))]
            else:
                node = node[attr]['> {:.3f}'.format(float(next(iter(node[attr]))[3:]))]
        return node

    def predict(self, X: pd.DataFrame):
        """Traverse the tree to make predictions"""
        return np.array([self.predict_one(x) for _, x in X.iterrows()])

    def opt_split_attribute(self, X: pd.DataFrame, y: pd.Series):
        """Find the best attribute and value to split on based on the chosen criterion (MSE or Variance Reduction)."""
        best_gain = -float('inf')
        best_attr, best_value = None, None

        for feature in X.columns:
            values = sorted(X[feature].unique())
            # Check midpoints between consecutive values
            split_points = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

            for value in split_points:
                X_left, y_left, X_right, y_right = self.split_data(X, y, feature, value)
                gain = self.variance_reduction(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_attr, best_value = feature, value

        return best_attr, best_value

    def split_data(self, X: pd.DataFrame, y: pd.Series, attribute, value):
        """Split the data based on an attribute and its value."""
        mask = X[attribute] <= value
        return X[mask], y[mask], X[~mask], y[~mask]
        
    def variance_reduction(self, y, y_left, y_right):
        """Calculate the variance reduction or MSE reduction as a split criterion."""
        var_total = np.var(y)
        var_left = np.var(y_left)
        var_right = np.var(y_right)
        
        # Weighted average of the variance in the left and right splits
        weighted_var = (len(y_left) * var_left + len(y_right) * var_right) / len(y)
        
        return var_total - weighted_var
    
    def create_dot(self, tree_dict=None, node_id=0, parent=None, branch_label="", leaf_color="lightblue"):
        """Recursively convert the nested tree structure into DOT format with thresholds in the node labels."""
        if tree_dict is None:
            tree_dict = self.tree
    
        dot_string = ""
        current_node = f"node_{node_id}"
    
        # Base case: If it's a leaf node, return the value
        if not isinstance(tree_dict, dict):
            label = f"Output: {tree_dict:.3f}"
            dot_string += f'{current_node} [label="{label}", shape=ellipse, style=filled, fillcolor="{leaf_color}"];\n'
            if parent is not None:
                dot_string += f'{parent} -> {current_node} [label="{branch_label}"];\n'
            return dot_string, node_id
    
        # Recursive case: Iterate through the tree structure
        for feature, branches in tree_dict.items():
            true_threshold = None
            false_threshold = None
    
            # Separate the true and false branches and capture their thresholds
            for threshold, subtree in branches.items():
                if "<=" in threshold:
                    false_threshold = threshold
                    false_branch = subtree
                else:
                    true_threshold = threshold
                    true_branch = subtree
    
            # Create a label that includes the feature and both thresholds
            label = f"Column {feature}\\nValue {true_threshold}"
            dot_string += f'{current_node} [label="{label}"];\n'
    
            if parent is not None:
                dot_string += f'{parent} -> {current_node} [label="{branch_label}"];\n'
    
            # Recursively generate the true branch
            if true_branch is not None:
                node_id += 1
                subtree_str, node_id = self.create_dot(true_branch, node_id=node_id, parent=current_node, branch_label="True", leaf_color=leaf_color)
                dot_string += subtree_str
    
            # Recursively generate the false branch
            if false_branch is not None:
                node_id += 1
                subtree_str, node_id = self.create_dot(false_branch, node_id=node_id, parent=current_node, branch_label="False", leaf_color=leaf_color)
                dot_string += subtree_str
    
        return dot_string, node_id


    def plot(self, filename="tree"):
        """Generate and display the tree as a graph using Graphviz"""
        dot_string = "digraph G {\n"
        dot_tree, _ = self.create_dot(self.tree)
        dot_string += dot_tree
        dot_string += "}\n"

        # Render the tree using Graphviz
        graph = graphviz.Source(dot_string)
        graph.render(filename)  # Use the filename parameter
        #display(graph)  # Display in Jupyter Notebook

class ClassificationTreeRealInput:
    def __init__(self, criterion='gini_index', max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series, depth=0):
        """Recursively build the classification tree"""
        if depth == self.max_depth or len(y.unique()) == 1:
            return Counter(y).most_common(1)[0][0]

        # Find the best attribute and value to split on
        best_attr, best_value = self.opt_split_attribute(X, y)

        if best_attr is None:
            return Counter(y).most_common(1)[0][0]

        tree = {best_attr: {}}

        # Split the data based on the best attribute and value
        X_left, y_left, X_right, y_right = self.split_data(X, y, best_attr, best_value)

        # Recursively build the tree
        tree[best_attr]['<= {:.3f}'.format(best_value)] = self.fit(X_left, y_left, depth + 1)
        tree[best_attr]['> {:.3f}'.format(best_value)] = self.fit(X_right, y_right, depth + 1)

        self.tree = tree
        return tree

    def predict_one(self, x: pd.Series):
        """Predict a single sample by traversing the tree"""
        node = self.tree
        while isinstance(node, dict):
            attr = next(iter(node))
            if x[attr] <= float(next(iter(node[attr]))[3:]):
                node = node[attr]['<= {:.3f}'.format(float(next(iter(node[attr]))[3:]))]
            else:
                node = node[attr]['> {:.3f}'.format(float(next(iter(node[attr]))[3:]))]
        return node

    def predict(self, X: pd.DataFrame):
        """Traverse the tree to make predictions"""
        return np.array([self.predict_one(x) for _, x in X.iterrows()])

    def opt_split_attribute(self, X: pd.DataFrame, y: pd.Series):
        """Find the best attribute and value to split on based on the chosen criterion."""
        best_gain = -float('inf')
        best_attr, best_value = None, None

        for feature in X.columns:
            values = sorted(X[feature].unique())
            # Check midpoints between consecutive values
            split_points = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

            for value in split_points:
                X_left, y_left, X_right, y_right = self.split_data(X, y, feature, value)
                gain = self.information_gain(y, y_left, y_right) if self.criterion == 'entropy' else self.gini_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_attr, best_value = feature, value

        return best_attr, best_value

    def split_data(self, X: pd.DataFrame, y: pd.Series, attribute, value):
        """Split the data based on an attribute and its value."""
        mask = X[attribute] <= value
        return X[mask], y[mask], X[~mask], y[~mask]

    def gini_gain(self, y, y_left, y_right):
        """Calculate the Gini gain of a split."""
        total_gini = self.gini_index(y)
        weighted_gini = (len(y_left) * self.gini_index(y_left) + len(y_right) * self.gini_index(y_right)) / len(y)
        return total_gini - weighted_gini

    def gini_index(self, y):
        """Calculate the Gini index of a dataset."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum([p ** 2 for p in probabilities])

    def information_gain(self, y, y_left, y_right):
        """Calculate the information gain of a split (using entropy)."""
        total_entropy = self.entropy(y)
        weighted_entropy = (len(y_left) * self.entropy(y_left) + len(y_right) * self.entropy(y_right)) / len(y)
        return total_entropy - weighted_entropy

    def entropy(self, y):
        """Calculate the entropy of a dataset."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    def create_dot(self, tree_dict=None, node_id=0, parent=None, branch_label="", leaf_color="lightblue"):
        """Recursively convert the nested tree structure into DOT format with thresholds in the node labels."""
        if tree_dict is None:
            tree_dict = self.tree
    
        dot_string = ""
        current_node = f"node_{node_id}"
    
        # Base case: If it's a leaf node, return the value
        if not isinstance(tree_dict, dict):
            label = f"Output: {tree_dict:.3f}"
            dot_string += f'{current_node} [label="{label}", shape=ellipse, style=filled, fillcolor="{leaf_color}"];\n'
            if parent is not None:
                dot_string += f'{parent} -> {current_node} [label="{branch_label}"];\n'
            return dot_string, node_id
    
        # Recursive case: Iterate through the tree structure
        for feature, branches in tree_dict.items():
            true_threshold = None
            false_threshold = None
    
            # Separate the true and false branches and capture their thresholds
            for threshold, subtree in branches.items():
                if "<=" in threshold:
                    false_threshold = threshold
                    false_branch = subtree
                else:
                    true_threshold = threshold
                    true_branch = subtree
    
            # Create a label that includes the feature and both thresholds
            label = f"Column {feature}\\nValue {true_threshold}"
            dot_string += f'{current_node} [label="{label}"];\n'
    
            if parent is not None:
                dot_string += f'{parent} -> {current_node} [label="{branch_label}"];\n'
    
            # Recursively generate the true branch
            if true_branch is not None:
                node_id += 1
                subtree_str, node_id = self.create_dot(true_branch, node_id=node_id, parent=current_node, branch_label="True", leaf_color=leaf_color)
                dot_string += subtree_str
    
            # Recursively generate the false branch
            if false_branch is not None:
                node_id += 1
                subtree_str, node_id = self.create_dot(false_branch, node_id=node_id, parent=current_node, branch_label="False", leaf_color=leaf_color)
                dot_string += subtree_str
    
            return dot_string, node_id

    def plot(self, filename="tree"):
        """Generate and display the tree as a graph using Graphviz"""
        dot_string = "digraph G {\n"
        dot_tree, _ = self.create_dot(self.tree)
        dot_string += dot_tree
        dot_string += "}\n"

        # Render the tree using Graphviz
        graph = graphviz.Source(dot_string)
        graph.render(filename)  # Use the filename parameter
        #display(graph)  # Display in Jupyter Notebook

class ClassificationTree:
    def __init__(self, criterion='gini_index', max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series, depth=0):
        """Recursively build the classification tree"""
        if len(y) == 0:
            return None 
        if depth == self.max_depth or len(y.unique()) == 1:
            return Counter(y).most_common(1)[0][0]

        # Find the best attribute to split on
        best_attr = self.opt_split_attribute(X, y)

        if best_attr is None:
            return Counter(y).most_common(1)[0][0]

        tree = {best_attr: {}}

        # Split the data based on the best attribute
        X_left, y_left, X_right, y_right = self.split_data(X, y, best_attr)

        # Recursively build the tree
        tree[best_attr][0] = self.fit(X_left, y_left, depth + 1)
        tree[best_attr][1] = self.fit(X_right, y_right, depth + 1)

        self.tree = tree
        return tree

    def predict_one(self, x: pd.Series):
        """Predict a single sample by traversing the tree"""
        node = self.tree
        while isinstance(node, dict):
            attr = next(iter(node))
            node = node[attr][x[attr]]
        return node

    def predict(self, X: pd.DataFrame):
        """Traverse the tree to make predictions"""
        return np.array([self.predict_one(x) for _, x in X.iterrows()])

    def opt_split_attribute(self, X: pd.DataFrame, y: pd.Series):
        """Find the best attribute to split on based on the chosen criterion."""
        best_gain = -float('inf')
        best_attr = None

        for feature in X.columns:
            X_left, y_left, X_right, y_right = self.split_data(X, y, feature)
            gain = self.information_gain(y, y_left, y_right) if self.criterion == 'entropy' else self.gini_gain(y, y_left, y_right)

            if gain > best_gain:
                best_gain = gain
                best_attr = feature

        return best_attr

    def split_data(self, X: pd.DataFrame, y: pd.Series, attribute):
        """Split the data based on an attribute"""
        mask = X[attribute] == 0
        return X[mask], y[mask], X[~mask], y[~mask]

    def gini_gain(self, y, y_left, y_right):
        """Calculate the Gini gain of a split."""
        total_gini = self.gini_index(y)
        weighted_gini = (len(y_left) * self.gini_index(y_left) + len(y_right) * self.gini_index(y_right)) / len(y)
        return total_gini - weighted_gini

    def gini_index(self, y):
        """Calculate the Gini index of a dataset."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum([p ** 2 for p in probabilities])

    def information_gain(self, y, y_left, y_right):
        """Calculate the information gain of a split (using entropy)."""
        total_entropy = self.entropy(y)
        weighted_entropy = (len(y_left) * self.entropy(y_left) + len(y_right) * self.entropy(y_right)) / len(y)
        return total_entropy - weighted_entropy

    def entropy(self, y):
        """Calculate the entropy of a dataset."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def create_dot(self, tree_dict=None, node_id=0, parent=None, branch_label="", leaf_color="lightblue"):
        """
        Recursively convert the nested tree structure into DOT format with split index in the node labels.
        This version handles tree structures like {'0_0': {...}, ...} where keys are a combination of feature index and split index.
        """
        if tree_dict is None:
            tree_dict = self.tree
    
        dot_string = ""
        current_node = f"node_{node_id}"
    
        # Base case: If it's a leaf node, return the value
        if not isinstance(tree_dict, dict):
            label = f"Output: {tree_dict}"
            dot_string += f'{current_node} [label="{label}", shape="ellipse", style="filled", fillcolor="{leaf_color}"];\n'
            if parent is not None:
                dot_string += f'{parent} -> {current_node} [label="{branch_label}"];\n'
            return dot_string, node_id
    
        # Recursive case: Iterate through the tree structure
        for key, branches in tree_dict.items():
            feature, split_index = key.split('_')
            label = f"Column: {feature}\\nValue: {split_index}"
            dot_string += f'{current_node} [label="{label}", shape="box"];\n'

            if parent is not None:
                dot_string += f'{parent} -> {current_node} [label="{branch_label}"];\n'
    
            # Generate true branch (usually indexed by 0)
            if 0 in branches:
                node_id += 1
                subtree_str, node_id = self.create_dot(branches[0], node_id=node_id, parent=current_node, branch_label="True", leaf_color=leaf_color)
                dot_string += subtree_str
    
            # Generate false branch (usually indexed by 1)
            if 1 in branches:
                node_id += 1
                subtree_str, node_id = self.create_dot(branches[1], node_id=node_id, parent=current_node, branch_label="False", leaf_color=leaf_color)
                dot_string += subtree_str
    
        return dot_string, node_id

    def plot(self, filename="tree"):
        """Generate and display the tree as a graph using Graphviz"""
        dot_string = "digraph G {\n"
        dot_tree, _ = self.create_dot(self.tree)
        dot_string += dot_tree
        dot_string += "}\n"

        # Render the tree using Graphviz
        graph = graphviz.Source(dot_string)
        graph.render(filename)  # Use the filename parameter
        #display(graph)  # Display in Jupyter Notebook