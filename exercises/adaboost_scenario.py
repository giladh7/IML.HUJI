import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    if noise == 0:
        title_suffix = "(without noise)"
    else:
        title_suffix = "(with noise = {noise})".format(noise=noise)
    # Question 1: Train- and test errors of AdaBoost in noiseless case

    adaboost_model = AdaBoost(wl=lambda: DecisionStump(), iterations=n_learners).fit(train_X, train_y)
    test_errors, train_errors = np.zeros(n_learners), np.zeros(n_learners)
    for iteration in range(n_learners):
        test_errors[iteration] = adaboost_model.partial_loss(test_X, test_y, iteration + 1)
        train_errors[iteration] = adaboost_model.partial_loss(train_X, train_y, iteration + 1)
    fig = go.Figure([go.Scatter(x=np.arange(n_learners + 1), y=test_errors, name="Test Error"),
                     go.Scatter(x=np.arange(n_learners + 1), y=train_errors, name="Train Error")])
    fig.update_layout(title_text="Test and train errors as a function of learners number " + title_suffix,
                      xaxis_title="Number of learners", yaxis_title="Error rate")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    if noise == 0:
        fig = make_subplots(rows=2, cols=2, subplot_titles=["T = {t}".format(t=t) for t in T])
        test_y_symbols = (test_y == 1).astype(int)
        for idx, t in enumerate(T):
            fig.add_traces([go.Scatter(x=test_X[:, 0], y=test_X[:, 1], showlegend=False, mode='markers',
                                       marker=dict(color=test_y, symbol=test_y_symbols,
                                                   colorscale=[custom[0], custom[-1]],
                                                   line=dict(color="black", width=1))),
                            decision_surface(lambda x: adaboost_model.partial_predict(x, T=t), lims[0], lims[1],
                                             showscale=False)],
                           rows=idx // 2 + 1, cols=idx % 2 + 1)
        fig.update_layout(title="Decision boundary of ensemble by number of iterations")
        fig.show()

    # Question 3: Decision surface of best performing ensemble
    if noise == 0:
        ensemble_size = np.argmin(test_errors) + 1
        accuracy = 1 - np.min(test_errors)
        title = "Best ensemble size decision boundary (size = {size}, accuracy = {accuracy})". \
            format(size=ensemble_size, accuracy=accuracy)
        fig = go.Figure([decision_surface(lambda x: adaboost_model.partial_predict(x, ensemble_size), lims[0], lims[1],
                                          showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers', showlegend=False,
                                    marker=dict(color=test_y, colorscale=class_colors(2)))])
        fig.update_layout(title=title)
        fig.show()
        print(np.argmin(test_errors))
        print(np.min(test_errors))

    # Question 4: Decision surface with weighted samples
    if noise == 0:
        transform_coef = 15
    else:
        transform_coef = 5
    fig = go.Figure([decision_surface(lambda x: adaboost_model.partial_predict(x, n_learners), lims[0], lims[1],
                                      showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers', showlegend=False,
                                marker=dict(color=train_y, colorscale=class_colors(2),
                                            size=(adaboost_model.D / adaboost_model.D.max()) * transform_coef))])
    fig.update_layout(title="Decision surface with weighted samples " + title_suffix)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
