from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).drop_duplicates()
    # omit samples with too small prices
    full_data = full_data[full_data["price"] >= 1000]
    # omit samples with negative bedroom number
    full_data = full_data[full_data["bedrooms"] >= 0]
    # omit samples with negative or zero bathrooms number
    full_data = full_data[full_data["bathrooms"] >= 0]
    # omit samples with 0 or negative floors number
    full_data = full_data[full_data["floors"] >= 1]
    # omit samples with antique build year
    full_data = full_data[full_data["yr_built"] >= 1800]
    # # omit samples with too many bedrooms
    # full_data = full_data[full_data['bedrooms'] <= 30]
    # drop the most expensive sells - explain at the pdf
    # full_data = full_data[full_data["price"] < np.quantile(full_data['price'], 0.99)]

    full_data["does_renovation"] = full_data["yr_renovated"].apply(lambda reno: 0 if reno == 0 else 1)
    # scaling the sqft_living with log, because it has too high values
    full_data["sqft_living"] = np.log(full_data["sqft_living"])

    # handle zip code with dummy variables
    dummy_vars = pd.get_dummies(full_data["zipcode"], prefix='zip')
    full_data = pd.concat([full_data, dummy_vars], axis=1)
    full_data = full_data.drop('zip_98002.0', axis=1)
    # fill NAN's in view to 0
    full_data["view"].fillna(0)
    # creating house age feature from yr_built and date
    full_data = full_data[full_data["date"].notnull()]
    full_data["date"] = pd.to_datetime(full_data["date"])
    # creating house age from yr_built and date
    full_data["house_age"] = full_data["date"].dt.year - full_data["yr_built"]
    response = full_data["price"]
    full_data = full_data.drop(['id', 'date', 'zipcode', 'yr_built', 'yr_renovated', 'price'], axis=1)
    return full_data, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    generic_plot_title = "Correlation between {feature} to response (r={pearson})"
    response_std = np.std(y)
    for feature in X.columns:
        feature_response_cov = y.cov(X[feature])
        feature_response_std = np.std(X[feature])
        cur_pearson = feature_response_cov / feature_response_std / response_std
        fig = px.scatter(x=X[feature], y=y,
                         title=generic_plot_title.format(feature=feature, pearson=cur_pearson),
                         labels=dict(x=feature, y="house price"))
        # fig.write_image(output_path + "/" + feature + ".jpeg", engine="orca")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    path_for_data = "C:/Users/gilad/Desktop/uni/current semester/" \
                    "IML.HUJI/datasets/house_prices.csv"
    design_matrix, response = load_data(path_for_data)

    # Question 2 - Feature evaluation with respect to response
    path_for_saved_plots = "C:/Users/gilad/Desktop/uni/current semester/IML.HUJI/temp"
    # feature_evaluation(design_matrix, response, path_for_saved_plots)

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(design_matrix, response)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    model = LinearRegression()
    percentages = np.arange(10, 101) / 100
    average_loss = np.zeros(91)
    for idx, percent in enumerate(percentages):
        cur_percent_losses = np.zeros(10)
        for i in range(10):
            cur_sample = X_train.sample(frac=percent)
            cur_results = y_train[cur_sample.index]
            model.fit(cur_sample.to_numpy(), cur_results)
            cur_percent_losses[i] = model.loss(X_test.to_numpy(), y_test.to_numpy())
        average_loss[idx] = cur_percent_losses.mean()
    fig = px.line(x=percentages, y=average_loss, title="sample_mean_graph_title",
                  labels=dict(x="Percentage", y="Average loss"))
    fig.show()
