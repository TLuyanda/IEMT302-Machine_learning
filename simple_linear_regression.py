#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
    # Generate some example data
    # X: feature (independent variable), y: target (dependent variable)
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 1.3, 3.75, 2.25])

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Plot the results
    plt.scatter(X, y, color='blue', label='Original data')
    plt.plot(X, y_pred, color='red', label='Fitted line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # Print model parameters
    print(f'Coefficient: {model.coef_[0]}')
    print(f'Intercept: {model.intercept_}')

if __name__ == "__main__":
    main()
