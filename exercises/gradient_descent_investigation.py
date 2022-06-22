import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from plotly.subplots import make_subplots

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def callback(gradient_descent, args):
        values.append(args[1])
        weights.append(args[0])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    lowest_loss = []
    L1_convergence = go.Figure()
    L2_convergence = go.Figure()
    for eta in etas:
        for model, name in ((L1, "L1"), (L2, "L2")):
            cur_LR = FixedLR(eta)
            callback, values, weights = get_gd_state_recorder_callback()

            GD_solver = GradientDescent(cur_LR, callback=callback)
            GD_solver.fit(model(init.copy()), None, None)
            title = f"{name} module as function of iteration number (eta = {eta})".format(name=name, eta=eta)
            plot_descent_path(model, np.array(weights), title)  # .show()
            if eta == 0.01:
                plot_descent_path(model, np.array(weights), title).show()

            fig = go.Scatter(x=np.arange(GD_solver.max_iter_), y=values, mode='lines',
                             name=f"learning rate {eta}".format(name=name, eta=eta))
            if name == "L1":
                L1_convergence.add_trace(fig)
            else:
                L2_convergence.add_trace(fig)
            lowest_loss.append(np.min(values))
    L1_convergence.update_layout(title="L1 convergence rate as function of iteration number")
    L1_convergence.update_xaxes(title_text="Iteration Number")
    L1_convergence.update_yaxes(title_text=f"{name} Norm".format(name=name))
    L1_convergence.show()

    L2_convergence.update_layout(title="L2 convergence rate as function of iteration number")
    L2_convergence.update_xaxes(title_text="Iteration Number")
    L2_convergence.update_yaxes(title_text=f"{name} Norm".format(name=name))
    L2_convergence.show()

    print(lowest_loss)
    lowest_loss = np.array(lowest_loss)
    lowest_loss.shape = (len(etas), 2)
    for i in range(len(lowest_loss)):
        print("eta = ", etas[i], "L1: ", lowest_loss[i][0], "  L2: ", lowest_loss[i][1])


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = make_subplots(rows=2, cols=2, subplot_titles=gammas, horizontal_spacing=0.01, vertical_spacing=0.05)
    figs = []
    for i, gamma in enumerate(gammas):
        cur_LR = ExponentialLR(eta, gamma)
        convergences = []
        callback, values, weights = get_gd_state_recorder_callback()
        GD_solver = GradientDescent(cur_LR, callback=callback)

        GD_solver.fit(L1(init.copy()), None, None)

        fig.add_trace(go.Scatter(x=np.arange(GD_solver.max_iter_), y=values, name=gamma),
                      row=(i // 2) + 1, col=(i % 2) + 1)
        print(f"min L1 with decay rate of {gamma} is {np.min(values)}")

    # Plot algorithm's convergence for the different values of gamma
    fig.update_layout(title="L1 Convergence rate as a function of iteration number")
    # fig.show()

    # Plot descent path for gamma=0.95
    L1_module = L1(init.copy())
    lr = ExponentialLR(eta, 0.95)
    callback, values, weights = get_gd_state_recorder_callback()
    GradientDescent(learning_rate=lr, callback=callback).fit(L1_module, None, None)
    fig = plot_descent_path(L1, np.array(weights), title="L1 norm with exponential learning rate gamma=0.95")
    fig.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = (X_train - X_train.mean()) / X_train.std(), y_train, (
            X_test - X_test.mean()) / X_test.std(), y_test

    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

    # # Plotting convergence rate of logistic regression over SA heart disease data
    callback, values, weights = get_gd_state_recorder_callback()
    gd = GradientDescent(callback=callback, learning_rate=FixedLR(1e-4), max_iter=20000)
    lg = LogisticRegression(solver=gd, include_intercept=True).fit(X_train, y_train)
    fpr, tpr, thresholds = roc_curve(y_train, lg.predict_proba(X_train))
    alphas = np.arange(0, 1, 0.01)

    fig = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                                     name="Random Class Assignment", showlegend=False),
                          go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=alphas, name="", showlegend=False,
                                     marker_size=5,
                                     hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
                    layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                                     xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                                     yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    # fig.show()
    best_alpha_index = np.argmax(tpr - fpr)
    best_alpha_value = thresholds[best_alpha_index]
    print(f"Best alpha: {best_alpha_value}")
    error = misclassification_error(y_test, lg.predict_proba(X_test) >= best_alpha_value)
    print("Test error for alpha: ", error)

    # # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    # alpha = 0.05
    # modules = ["l1", "l2"]
    # lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    #
    # callback, values, weights = get_gd_state_recorder_callback()
    # for module in modules:
    #     cv_scores = []
    #     for lam in lambdas:
    #         lg = LogisticRegression(include_intercept=True, penalty=module, alpha=alpha, lam=lam,
    #                                 solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000,
    #                                                        callback=callback))
    #         train_score, validation_score = cross_validate(lg, X_train, y_train, misclassification_error)
    #         cv_scores.append(validation_score)
    #
    #     best_lambda = lambdas[np.argmin(np.array(cv_scores))]
    #     print(f"best lambda for {module} penalty: {best_lambda}")
    #     callback, values, weights = get_gd_state_recorder_callback()
    #     lg = LogisticRegression(penalty=module, alpha=alpha, lam=best_lambda,
    #                             solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000,
    #                                                    callback=callback)).fit(X_train, y_train)
    #     error = misclassification_error(y_test, lg.predict(X_test))
    #     print(f"Test error for {module} penalty and {best_lambda} is: {error}")
    lambdas = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    for i in range(1, 3):
        callback, values, weights = get_gd_state_recorder_callback()
        validation_scores = np.zeros(lambdas.shape[0])
        train_scores = np.zeros(lambdas.shape[0])
        for j, lam in enumerate(lambdas[0:5]):
            model = LogisticRegression(penalty=f"l{i}", lam=lam, solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000,
                                                       callback=callback))
            train_score, validation_score = cross_validate(model,
                                                           X_train,
                                                           y_train,
                                                           misclassification_error)
            validation_scores[j] = validation_score
            train_scores[j] = train_score
        best_lambda = lambdas[np.argmin(validation_scores)]
        model = LogisticRegression(penalty=f"l{i}", lam=best_lambda, solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000,
                                                       callback=callback))
        model.fit(X_train, y_train)
        loss = model.loss(X_test, y_test)
        print(f"Best lambda for model with regularization L{i} is: {loss} with lambda {best_lambda}")


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
