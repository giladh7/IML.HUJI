from typing import NoReturn
from IMLearn.base.base_estimator import BaseEstimator
import numpy as np
import pandas as pd


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # classes are the all options for response, n_k records the count of each y_i
        self.classes_, n_k = np.unique(y, return_counts=True)
        if len(X.shape) == 1:
            n_features = 1
        else:
            n_features = X.shape[1]
        n_classes = self.classes_.shape[0]
        # creating self.pi according to y
        self.pi_ = n_k / y.shape[0]

        # crating mu by sum the match rows(samples)
        sum_by_class = pd.DataFrame(X).groupby(y).sum()
        self.mu_ = np.array(sum_by_class.T / n_k).T

        # creating cov by iterate over classes
        self.vars = np.zeros((n_classes, n_features))
        for idx, class_ in enumerate(self.classes_):
            X_by_class = X[y == class_]
            self.vars[idx] = np.var(X_by_class, axis=0)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # choose the highest "likelihood" per sample
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        d = X.shape[1]  # dimension of x - number of features
        total_likelihood = np.zeros((X.shape[0], self.classes_.size))
        # calculate likelihood and normal coef per class according to gaussian formulas
        for class_num in range(self.classes_.size):
            normal_coef = 1 / np.sqrt((self.vars[class_num]) * (2 * np.pi))
            X_by_class = X - self.mu_[class_num]
            exponent = np.exp(-0.5 / self.vars[class_num] * X_by_class ** 2)
            total_likelihood[:, class_num] = np.prod(exponent * normal_coef, axis=1)

        return total_likelihood

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
        from ...metrics import misclassification_error
        return misclassification_error(self.predict(X), y)
