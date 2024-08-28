from typing import Union
import pandas as pd
def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size, "y_hat and y must have the same size"
    
    correct_predictions = 0
    total_predictions = y.size

    for i in range(total_predictions):
        if y_hat[i] == y[i]:
            correct_predictions += 1

    return correct_predictions / total_predictions

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:

    true_positives = 0
    false_positives = 0

    for i in range(y.size):
        if y_hat[i] == cls:
            if y[i] == cls:
                true_positives += 1
            else:
                false_positives += 1

    if true_positives + false_positives == 0:
        return 0.0  # Avoid division by zero

    return true_positives / (true_positives + false_positives)

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    
    true_positives = 0
    false_negatives = 0

    for i in range(y.size):
        if y[i] == cls:
            if y_hat[i] == cls:
                true_positives += 1
            else:
                false_negatives += 1

    if true_positives + false_negatives == 0:
        return 0.0  # Avoid division by zero

    return true_positives / (true_positives + false_negatives)

import math

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
   
    assert y_hat.size == y.size, "y_hat and y must have the same size"

    sum_squared_errors = 0.0
    n = y.size

    for i in range(n):
        error = y_hat[i] - y[i]
        sum_squared_errors += error ** 2

    mean_squared_error = sum_squared_errors / n
    return math.sqrt(mean_squared_error)

def mae(y_hat: pd.Series, y: pd.Series) -> float:
  
    assert y_hat.size == y.size, "y_hat and y must have the same size"

    sum_absolute_errors = 0.0
    n = y.size

    for i in range(n):
        error = abs(y_hat[i] - y[i])
        sum_absolute_errors += error

    return sum_absolute_errors / n
