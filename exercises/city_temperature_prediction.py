import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

NAN_THRESHOLD = -70
DATE_COLUMN = 2
COUNTRY = "Israel"

pio.templates.default = "simple_white"


def load_data(filename: str) -> (pd.DataFrame):
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=[DATE_COLUMN]).drop_duplicates()
    # drop samples with Nan - represented by -72.777778
    df = df[df["Temp"] > NAN_THRESHOLD]
    # add day of year column
    df["DayOfYear"] = df["Date"].dt.dayofyear

    return df.drop(["Date"], axis=1)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data_path = "../datasets/City_Temperature.csv"
    design_matrix = load_data(data_path)

    # Question 2 - Exploring data for specific country
    # scatter plot of DayOfYear and Temp colored by year
    country_df = design_matrix[design_matrix["Country"] == COUNTRY]
    string_years = country_df["Year"].astype(str)
    fig = px.scatter(country_df, x="DayOfYear", y="Temp", color=string_years,
                     title="Relation between day of year and temperatue in {country}".format(country=COUNTRY),
                     labels={"DayOfYear": "Day of Year", "y": "Temperature"})
    # fig.show()

    # bar plot of std by month
    country_by_month_std = country_df.groupby('Month', as_index=False).agg('std')
    fig = px.bar(country_by_month_std, x='Month', y='Temp',
                 title="Standard deviation of temperatures by month",
                 labels={"Temp": "Std of temperatue"})
    # fig.show()

    # Question 3 - Exploring differences between countries
    by_month_by_country = design_matrix.groupby(['Country', 'Month'], as_index=False
                                                ).Temp.agg(['std', 'mean']).reset_index()
    fig = px.line(by_month_by_country, x='Month', y='mean', error_y='std',
                  color='Country', title="Average Temperature by Month",
                  labels={"mean": "Average Temperature"})
    # fig.show()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
