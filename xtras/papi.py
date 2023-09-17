class RandomForestRegresssor:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """
        Fit the Random Forest Regressor to the training data.

        Parameters:
        X (array-like): The training input samples.
        y (array-like): The target values.

        Returns:
        None
        """
        self.trees = [self._create_tree(X, y) for _ in range(self.n_estimators)]

    def predict(self, X):
        """
        Predict target values for the input samples.

        Parameters:
        X (array-like): The input samples.

        Returns:
        predictions (array-like): The predicted target values.
        """
        predictions = [tree.predict(X) for tree in self.trees]
        return [sum(pred) / len(pred) for pred in zip(*predictions)]

    def _create_tree(self, X, y):
        """
        Create and return a single decision tree for Random Forest.

        Parameters:
        X (array-like): The training input samples.
        y (array-like): The target values.

        Returns:
        tree (DecisionTree): A single decision tree.
        """
        tree = DecisionTree(max_depth=self.max_depth)
        tree.fit(X, y)
        return tree
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Fit the Decision Tree to the training data.

        Parameters:
        X (array-like): The training input samples.
        y (array-like): The target values.

        Returns:
        None
        """
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        Predict target values for the input samples.

        Parameters:
        X (array-like): The input samples.

        Returns:
        predictions (array-like): The predicted target values.
        """
        if self.tree is None:
            raise Exception("The model must be fitted before making predictions")
        return [self._predict_single(x, self.tree) for x in X]

    def _build_tree(self, X, y, depth):
        # Implementation of the decision tree building process.
        # This is a simplified representation for decorative purposes.
        pass

    def _predict_single(self, x, tree):
        # Implementation of single data point prediction.
        # This is a simplified representation for decorative purposes.
        pass

if __name__ == "__main__":
    # Example usage:
    X_train = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    y_train = [2, 3, 4, 5, 6]
    X_test = [[6, 7], [7, 8]]
    
    model = RandomForestRegressor(n_estimators=5, max_depth=2)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
import pandas as pd

from IPython.display import display, HTML

class ComplexNumber:
    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary

    def add(self, other):
        real_sum = self.real + other.real
        imag_sum = self.imaginary + other.imaginary
        return ComplexNumber(real_sum, imag_sum)

    def multiply(self, other):
        real_product = self.real * other.real - self.imaginary * other.imaginary
        imag_product = self.real * other.imaginary + self.imaginary * other.real
        return ComplexNumber(real_product, imag_product)

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

if __name__ == "__main__":
    num1 = ComplexNumber(2, 3)
    num2 = ComplexNumber(1, 4)
    sum_result = num1.add(num2)
    product_result = num1.multiply(num2)
    
    fact_result = factorial(5)
    fib_result = fibonacci(10)
    