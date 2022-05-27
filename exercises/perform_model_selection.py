from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    epsilon = np.random.normal(0, noise, n_samples)
    polynom = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    noiseless_y = polynom(X)
    y = noiseless_y + epsilon
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2.0 / 3)
    train_X, train_y, test_X, test_y = train_X[0].to_numpy(), train_y.to_numpy(), \
                                       test_X[0].to_numpy(), test_y.to_numpy()
    # scatter plot
    fig = go.Figure([go.Scatter(x=X, y=noiseless_y, mode='lines', name="True (noiseless) Model"),
                     go.Scatter(x=train_X, y=train_y, mode='markers', name="Train Set"),
                     go.Scatter(x=test_X, y=test_y, mode='markers',
                                marker=dict(color="#0000ff"), name="Test Set"),
                     ])
    fig.update_layout(height=400, title_text=f"True model and test and size sets (noise = {noise})")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    k_s = np.arange(11)
    train_errors, test_errors = np.zeros(11), np.zeros(11)
    for k in k_s:
        errors = cross_validate(PolynomialFitting(k), train_X, train_y, mean_square_error, 5)
        train_errors[k], test_errors[k] = errors
    fig = go.Figure([go.Scatter(x=k_s, y=train_errors, name="Train error"),
                     go.Scatter(x=k_s, y=test_errors, name="Validation error")])
    fig.update_layout(title_text=f"Error as a function of polynomial degree (noise = {noise})",
                      xaxis_title="Polynomial degree", yaxis_title="Error Rate")
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = int(np.argmin(test_errors))  # index and value are the same in that case
    model = PolynomialFitting(best_k).fit(train_X, train_y)
    error = mean_square_error(test_y, model.predict(test_X))
    print(f"Best k = {best_k}, Test Error = {round(error, 2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x, train_y = X[:n_samples, :], y[:n_samples]
    test_x, test_y = X[n_samples:, :], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0.0001, 2, n_evaluations)
    ridge_validation_errors, ridge_train_errors = np.zeros(n_evaluations), np.zeros(n_evaluations)
    lasso_validation_errors, lasso_train_errors = np.zeros(n_evaluations), np.zeros(n_evaluations)

    for idx, lam in enumerate(lambdas):
        ridge_train_errors[idx], ridge_validation_errors[idx] = cross_validate(
            RidgeRegression(lam), train_x, train_y, mean_square_error)
        lasso_train_errors[idx], lasso_validation_errors[idx] = cross_validate(
            Lasso(lam), train_x, train_y, mean_square_error)

    fig = go.Figure([go.Scatter(x=lambdas, y=ridge_train_errors, name="Ridge Train"),
                         go.Scatter(x=lambdas, y=ridge_validation_errors, name="Ridge Validation"),
                         go.Scatter(x=lambdas, y=lasso_train_errors, name="Lasso Train"),
                         go.Scatter(x=lambdas, y=lasso_validation_errors, name="Lasso Validation")])
    fig.update_layout(title_text="Ridge and Lasso errors as function of lambda")
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_lambda = lambdas[np.argmin(ridge_validation_errors)]
    best_lasso_lambda = lambdas[np.argmin(lasso_validation_errors)]
    print(f"Best regularization parameter for ridge: {best_ridge_lambda}")
    print(f"Best regularization parameter for lasso: {best_lasso_lambda}")

    least_square_model = LinearRegression().fit(train_x, train_y)
    ridge_model = RidgeRegression(best_ridge_lambda).fit(train_x, train_y)
    lasso_model = Lasso(best_lasso_lambda).fit(train_x, train_y)

    least_square_test_err = least_square_model.loss(test_x, test_y)
    ridge_model_test_err = ridge_model.loss(test_x, test_y)
    lasso_model_test_err = mean_square_error(test_y, lasso_model.predict(test_x))
    print(f"Least Squares test error: {least_square_test_err}")
    print(f"Ridge Model test error: {ridge_model_test_err}")
    print(f"Lasso Model test error: {lasso_model_test_err}")


if __name__ == '__main__':
    np.random.seed(0)
    # questions 1-3
    select_polynomial_degree()
    # question 4
    select_polynomial_degree(noise=0)
    # question 5
    select_polynomial_degree(n_samples=1500, noise=10)

    # question 7-8
    select_regularization_parameter()
