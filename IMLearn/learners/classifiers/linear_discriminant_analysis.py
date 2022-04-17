from typing import NoReturn
from IMLearn.base.base_estimator import BaseEstimator
import numpy as np
import pandas as pd
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # classes are the all options for response, n_k records the count of each y_i
        self.classes_, n_k = np.unique(y, return_counts=True)
        n_features = X.shape[1]
        n_classes = self.classes_.shape[0]
        # creating self.pi according to y
        self.pi_ = n_k / y.shape[0]

        # crating mu by sum the match rows(samples)
        sum_by_class = pd.DataFrame(X).groupby(y).sum()
        self.mu_ = np.array(sum_by_class.T / n_k).T

        # creating cov by iterate over classes
        self.cov_ = np.zeros((n_features, n_features))
        for idx, class_ in enumerate(self.classes_):
            X_by_class = X[y == class_] - self.mu_[idx]
            self.cov_ += X_by_class.T @ X_by_class
        self.cov_ /= (y.shape[0] - n_classes)
        self._cov_inv = inv(self.cov_)

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
        a_k = self.mu_ @ self._cov_inv.T
        b_k = np.log(self.pi_) - 0.5 * np.diag(self.mu_ @ self._cov_inv @ self.mu_.T)
        bayes_optimal_classifier = X @ a_k.T + b_k
        if len(X.shape) == 1:
            return self.classes_[np.argmax(bayes_optimal_classifier)]
        return self.classes_[np.argmax(bayes_optimal_classifier, axis=1)]

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
        normal_coef = 1 / np.sqrt((det(self.cov_) * ((2 * np.pi) ** d)))
        total_likelihood = np.zeros((X.shape[0], self.classes_.size))
        for class_num in range(self.classes_.size):
            X_by_class = X - self.mu_[class_num]
            total_likelihood[:, class_num] = np.exp(
                -0.5 * np.diag(X_by_class @ self._cov_inv @ X_by_class.T)) \
                                             * self.pi_[class_num] \
                                             * normal_coef
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


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
print(clf.predict_proba([[-0.8, -1]]))
lda = LDA()
lda.fit(X, y)
print(lda.likelihood(np.array([[-0.8, -1]])))
