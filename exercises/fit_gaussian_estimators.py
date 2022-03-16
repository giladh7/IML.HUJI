from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

# constants
MIU = 10
SIGMA = 1
SAMPLES_NUM = 1000
MULTI_MIU = np.array([0, 0, 4, 0])
MULTI_COV = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(MIU, SIGMA, SAMPLES_NUM)
    estimated_gaussian = UnivariateGaussian()
    estimated_gaussian.fit(samples)
    print((estimated_gaussian.mu_, estimated_gaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    sample_size = np.arange(10, 1010, 10, dtype=int)  # 10,20,30...990,1000
    samples_number = int((1000 - 10) / 10) + 1
    distance = np.zeros(samples_number)  # distance from the real miu per sample
    cur_estimated_gaussian = UnivariateGaussian()
    for i, size in enumerate(sample_size):
        cur_samples = samples[:size]
        cur_estimated_gaussian.fit(cur_samples)
        distance[i] = abs(cur_estimated_gaussian.mu_ - MIU)
    sample_mean_graph_title = "Absolute Error of miu as a function of the sample size"
    fig = px.line(x=sample_size, y=distance, title=sample_mean_graph_title,
                  labels=dict(x="Sample size", y="Absolute Error"))
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = estimated_gaussian.pdf(samples)
    Empirical_PDF_title = "The pdf of the samples according to the estimated gaussian"
    fig = px.scatter(x=samples, y=pdfs, title=Empirical_PDF_title,
                     labels=dict(x="Samples", y="PDF values"))
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multi_samples = np.random.multivariate_normal(MULTI_MIU, MULTI_COV, SAMPLES_NUM)
    multi_gaussian = MultivariateGaussian()
    multi_gaussian.fit(multi_samples)
    print(multi_gaussian.mu_)
    print(multi_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    MODEL_NUM_PER_FEATURE = 200
    f1, f3 = np.linspace(-10, 10, MODEL_NUM_PER_FEATURE), np.linspace(-10, 10, MODEL_NUM_PER_FEATURE)
    f1_f3_cartesian_prod = np.transpose(np.array([np.repeat(f1, len(f3)), np.tile(f3, len(f1))]))
    # create an extended array of the product fill with 0 in f2 and f4
    values = np.zeros((MODEL_NUM_PER_FEATURE**2, 4))
    values[:, 0] = f1_f3_cartesian_prod[:, 0]
    values[:, 2] = f1_f3_cartesian_prod[:, 1]
    values_log_likelihood = np.zeros(len(values))
    for idx,miu in enumerate(values):
        values_log_likelihood[idx] = multi_gaussian.log_likelihood(miu, MULTI_COV, multi_samples)
    # creating the heatmap
    title = "Log-Likelihood of Different Models"
    fig = go.Figure(go.Heatmap(x=values[:, 2], y=values[:, 0], z=values_log_likelihood),
              layout=go.Layout(title=title))
    fig.update_layout(xaxis_title="f3 values", yaxis_title="f1 values")
    fig.show()

    # Question 6 - Maximum likelihood
    best_model_idx = np.argmax(values_log_likelihood)
    best_model = values[best_model_idx][0], values[best_model_idx][2]
    print(best_model)  # the pair f1,f3 with the max log-likelihood
    print(values_log_likelihood[best_model_idx])  # the log-likelihood itself


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
