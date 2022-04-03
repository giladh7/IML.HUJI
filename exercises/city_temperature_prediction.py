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
    X_train, y_train, X_test, y_test = split_train_test(country_df['DayOfYear'], country_df["Temp"])
    k_values = np.arange(1, 11)
    loss_values = np.zeros(10)
    for idx,k in enumerate(k_values):
        polynom_model = PolynomialFitting(k)
        polynom_model.fit(X_train, y_train)
        loss_values[idx] = np.round(polynom_model.loss(X_test, y_test), 2)
        print("loss of polynomial model of degree {k}: {loss}".
              format(k=k, loss=loss_values[idx]))

    fig = px.bar(x=k_values, y=loss_values, title="Loss as function of polynomial degree(k)",
                 labels={"x": "Degree (k)", "y": "Mean Loss"})

    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    K = 5
    polyfit_model = PolynomialFitting(K)
    polyfit_model.fit(country_df['DayOfYear'], country_df['Temp'])
    countries = ['South Africa', 'The Netherlands', 'Jordan']
    countries_error = []
    for country in countries:
        current_days = design_matrix[design_matrix['Country'] == country]['DayOfYear']
        current_temp = design_matrix['Temp'][current_days.index]
        countries_error.append(polyfit_model.loss(current_days, current_temp))
    fig = px.bar(x=countries, y=countries_error,
                 title="Error of model trained israel records on different countries",
                 labels={"x": "countries", "y": "Model Error"})
    fig.show()
