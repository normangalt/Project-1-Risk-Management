import numpy as np
from scipy.optimize import minimize

portfolio_size = 0.8
def equall_weights(factors_number):
    return np.repeat((portfolio_size/factors_number), factors_number)

def mean_var(factors_number, sigma_, y_, mean_):
    def con0(weights):
        return weights.sum() - portfolio_size

    def func_(weights, args):
        return (-1)*(weights.T.dot(args[2])) - (args[1]/2)*(weights.T.dot(args[0].dot(weights)))

    constraints = [{"type":"eq", "fun":con0}]
    return minimize(func_, np.array([[0] for _ in range(factors_number)]),
                     args = [sigma_, y_, mean_], constraints = constraints,
                     bounds = ((0, None) for _ in range(factors_number))).x

def min_var(factors_number, sigma_):
    def con0(weights):
        return weights.sum() - portfolio_size

    def func_(weights, args):
        sigma_ = args[0]
        return weights.T.dot(sigma_.dot(weights))

    constraints = [{"type":"eq", "fun":con0}]
    return minimize(func_, np.array([[0] for _ in range(factors_number)]),
                     args = [sigma_], constraints = constraints,
                     bounds = ((0, None) for _ in range(factors_number))).x

def cvar(factors_number, data, b):
    def con0(weights):
        return weights.sum() - portfolio_size

    def CVaR_(weights, args):
        data = args[0]*(-1)
        columns_length = len(data.columns)
        b = args[1]
        b_percentile_ = data.quantile(1-b, axis = 0).values
        b_percentile_ = b_percentile_.reshape((columns_length, 1))
        indexes = data > b_percentile_.T
        data = data[indexes]
        data = data.mean(axis = 0).values.reshape((columns_length, 1))
        return weights.T.dot(data)

    constraints =  [{"type":"eq", "fun":con0}]
    return minimize(CVaR_, np.array([[0] for _ in range(factors_number)]),
                    args = [data, b], constraints = constraints,
                    bounds = ((0, None) for _ in range(factors_number))).x
