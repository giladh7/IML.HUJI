from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
from ...metrics.loss_functions import misclassification_error
import numpy as np
from itertools import product

# constants
GREATER_THAN_ONE = 2


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_features = X.shape[1]
        min_error = GREATER_THAN_ONE  # just a dummy greater than one for the first loop
        for sign, feature in product([-1, 1], range(n_features)):
            cur_threshold, cur_error = self._find_threshold(X[:, feature], y, sign)
            if cur_error < min_error:
                min_error = cur_error
                self.threshold_, self.j_, self.sign_ = cur_threshold, feature, sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        n_samples = values.size
        sort_index = np.argsort(values)
        values, labels = values[sort_index], labels[sort_index]
        signed_labels = np.sign(labels)  # needed for handling weighted labels
        # divide by the first value - all samples classified as sign
        prediction = np.full(labels.size, sign)
        thr, thr_err = values[0], np.abs(labels[prediction != signed_labels]).sum() / n_samples

        for idx, value in enumerate(values[1:]):
            prediction[idx] = -sign
            cur_err = np.abs(labels[prediction != signed_labels]).sum() / n_samples
            if cur_err < thr_err:
                thr, thr_err = value, cur_err

        # last option - all samples classified as -sign
        prediction = np.full(labels.size, -sign)
        last_thr, last_thr_err = np.inf, np.abs(labels[prediction != signed_labels]).sum() / n_samples
        if last_thr_err < thr_err:
            thr, thr_err = last_thr, last_thr_err

        return thr, thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))
