import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portfolios import equall_weights, mean_var, min_var, cvar
import statsmodels.api as smf

# Initializing supporting variables
group_number = 5
group_factors_names_ = ["Value", "Momentum", "Investment", "Profitability"]

period = 120
gam_ = 5
b = 0.95
number_of_portfolios = 4

# Reading and exploring data
data = pd.read_excel("Data.xlsm")

factors_number = len(group_factors_names_)
in_sample = data[data["Date"] < "1/11/1999"]
out_of_sample = data[data["Date"] >= "1/11/1999"]

# Task 1
task_1_3_factors_names_ = ["Market"] + group_factors_names_
task_1_3_data = in_sample[task_1_3_factors_names_]

plt.plot(in_sample["Date"], task_1_3_data[task_1_3_factors_names_].cumsum())

plt.xlabel("Year")
plt.ylabel("Return")

plt.legend(prop = {'size':10}, loc ="best", labels = task_1_3_factors_names_)
plt.show()

input("(Task 2) Press enter to continue")

# Task 2
std_ = task_1_3_data.std()
mean_ = task_1_3_data.mean()

sharpe_ratio = mean_/std_
print(sharpe_ratio)

input("(Task 3) Press enter to continue")

# Task 3
var95 = task_1_3_data.quantile(b)
print(var95)

input("(Task 4) Press enter to continue")

# Task 4
benchmark_portfolio = out_of_sample["Market"]
factors_portfolios = out_of_sample.drop(columns = ["Date"])

factors_portfolios = factors_portfolios[group_factors_names_]

sample_length = len(out_of_sample)
age_length = sample_length - period

weights = np.empty((factors_number, age_length, number_of_portfolios))

for time in range(age_length):
    period_end  = time + 120
    
    window_data = factors_portfolios.iloc[time:period_end]
    
    mean = window_data.mean()
    sigma = window_data.cov()

    portfolio = 0
    portfolio_weights = equall_weights(factors_number)
    weights[:, time, portfolio] = portfolio_weights
    
    portfolio += 1
    portfolio_weights = min_var(factors_number, sigma)
    weights[:, time, portfolio] = portfolio_weights
    
    portfolio += 1
    portfolio_weights = mean_var(factors_number, sigma, gam_, mean)
    weights[:, time, portfolio] = portfolio_weights

    portfolio += 1
    portfolio_weights = cvar(factors_number, window_data, b)
    weights[:, time, portfolio] = portfolio_weights

returns = np.empty((factors_number, sample_length - period, number_of_portfolios)) 
benchmark_portfolio_returns = np.empty((sample_length - period, 1))

for period_index in range(1, age_length):
    window_data = factors_portfolios.iloc[period_index:period_index + 120]
    benchmark_portfolio_values = benchmark_portfolio.iloc[period_index:period_index + 120].to_numpy()

    benchmark_portfolio_returns[period_index, 0] = np.sum(benchmark_portfolio_values)
    benchmark_portfolio_values = benchmark_portfolio_values.reshape((period, 1))

    for portfolio_index in range(number_of_portfolios):
        portfolio_weights = weights[:, period_index - 1, portfolio].reshape((factors_number, 1))

        portfolio_return = benchmark_portfolio_values + window_data * portfolio_weights.T
        returns[:, period_index, portfolio] = portfolio_return.sum()

returns = returns.sum(axis = 0)

x_axis = out_of_sample["Date"][:age_length].values
portfolios_names = ["Market", "Equall weights", "Min-Variance", "Mean-Variance", "CVaR"]

market_std_ = np.std(benchmark_portfolio_returns)
market_mean_ = np.mean(benchmark_portfolio_returns)

print("Market portfolio Sharpe Ratio: {}.".format(market_mean_/market_std_))
print("Market portfolio Sharpe Ratio: {}.".format(np.quantile(benchmark_portfolio_returns, b)), "\n")
plt.plot(x_axis, benchmark_portfolio_returns.cumsum())

for portfolio_index in range(number_of_portfolios):
    portfolio_return = returns[:, portfolio_index]
    plt.plot(x_axis, portfolio_return.cumsum())
        
    std_ = portfolio_return.std()
    mean_ = portfolio_return.mean()

    sharpe_ratio = mean_/std_
    print("Sharpe Ratio of the {} portfolio: {}.".format(portfolios_names[portfolio_index + 1], sharpe_ratio))

    var95 = np.quantile(portfolio_return, b)
    print("VaR5 of the {} portfolio: {}.".format(portfolios_names[portfolio_index + 1], var95), "\n")

plt.legend(labels = portfolios_names, loc = "upper left", prop = {"size":6})
plt.show()

#Task 5
shrinkage_coefficients = np.arange(0, 1, 0.05)[::4]

target_cov_ = data.drop(columns=["Date"])
target_cov_ = target_cov_[group_factors_names_].cov()

target_mean_ = np.nanmean(data.drop(columns=["Date"]))
portfolio_index = portfolios_names.index("Mean-Variance")

weights = np.empty((factors_number, age_length, 1))
for shrinkage_coefficient in shrinkage_coefficients:
    for time in range(age_length):
        period_end  = time + 120
        
        window_data = factors_portfolios.iloc[time:period_end]
        
        mean = (1 - shrinkage_coefficient) * window_data.mean() + shrinkage_coefficient * target_mean_
        sigma = (1 - shrinkage_coefficient) * window_data.cov() + shrinkage_coefficient * target_cov_

        portfolio_weights = mean_var(factors_number, sigma, gam_, mean)
        weights[:, time, 0] = portfolio_weights

    returns_Task_5 = np.empty((factors_number, age_length, 1)) 
    for period_index in range(1, age_length):
        window_data = factors_portfolios.iloc[period_index:period_index + 120]
        benchmark_portfolio_values = benchmark_portfolio.iloc[period_index:period_index + 120].to_numpy()
        benchmark_portfolio_values = benchmark_portfolio_values.reshape((period, 1))

        portfolio_weights = weights[:, period_index - 1, 0]

        portfolio_return = benchmark_portfolio_values + window_data * portfolio_weights.T
        returns_Task_5[:, period_index, 0] = portfolio_return.sum()

    returns_Task_5 = returns_Task_5.sum(axis = 0)

    std_ = returns_Task_5.std()
    mean_ = returns_Task_5.mean()

    sharpe_ratio = mean_/std_
    print("Shrinkage coefficient: {}.".format(shrinkage_coefficient))
    print("Sharpe Ratio of the {} portfolio: {}.".format(portfolios_names[portfolio_index], sharpe_ratio))

    var95 = np.quantile(returns_Task_5, b)
    print("VaR5 of the {} portfolio: {}.".format(portfolios_names[portfolio_index], var95), "\n")

    plt.plot(out_of_sample["Date"][:age_length],
            returns_Task_5.cumsum(),
            label = "Mean-Variance:" + str(shrinkage_coefficient), linestyle = "dashed")

x_axis = out_of_sample["Date"][:age_length].values
portfolios_names = ["Market", "Equall weights", "Min-Variance", "Mean-Variance", "CVaR"]

benchmark_portfolio_returns = np.array([sum(benchmark_portfolio[index:index+120]) for index in range(age_length)])

plt.plot(x_axis, benchmark_portfolio_returns.cumsum(), color = 'magenta', linestyle = "solid")

plt.plot(x_axis,
        returns[:, portfolios_names.index("Equall weights") - 1].cumsum(),
        color = 'red', 
        linestyle = 'dotted',
        label = 'Equall weights')

plt.plot(x_axis,
        returns[:, portfolios_names.index("Mean-Variance") - 1].cumsum(),
        color = 'green',
        linestyle = 'dotted',
        label = 'Mean-Variance')

plt.legend(loc = "upper left", prop = {"size":5})
plt.show()

# Task 6(1)
aggregated_returns = np.array([factors_portfolios.iloc[index:index+120].sum()
                               for index in range(age_length)])
model = smf.OLS(returns[:, portfolios_names.index("CVaR") - 1], aggregated_returns)
results = model.fit()

print("Parameters of the model:")
print(results.params)
print("T statistics values")
print(results.tvalues)

# Task 6(2)
factors_number = len(data.columns) - 2

weights = equall_weights(factors_number)
returns_Task_6_2 = np.empty((factors_number, age_length, 1)) 

factors = data.drop(columns = ["Date", "Market"])
aggregated_returns = np.array([factors.iloc[index:index+120].sum()
                               for index in range(age_length)])

for period_index in range(1, age_length):
    window_data = factors.iloc[period_index:period_index + 120]
    benchmark_portfolio_values = benchmark_portfolio.iloc[period_index:period_index + 120].to_numpy()
    benchmark_portfolio_values = benchmark_portfolio_values.reshape((period, 1))

    portfolio_return = benchmark_portfolio_values + window_data * weights.T
    returns_Task_6_2[:, period_index, 0] = portfolio_return.sum()

returns_Task_6_2 = returns_Task_6_2.sum(axis = 0)
portfolio_name = "Equall weights - seven factors:"

model = smf.OLS(endog = returns_Task_6_2, exog = aggregated_returns)
results = model.fit()

print("Parameters of the model:")
print(results.params)
print("T statistics values")
print(results.tvalues)

# Task 7
benchmark_portfolio = out_of_sample["Market"]
factors_portfolios = out_of_sample.drop(columns = ["Date"])

factors_portfolios = factors_portfolios[group_factors_names_]

sample_length = len(out_of_sample)
age_length = sample_length - period

factors_number = len(group_factors_names_)
weights = np.empty((factors_number, age_length, number_of_portfolios))

L1 = 0.007
L2 = 0.006

tc = lambda X, L1, L2: np.multiply(L1, X - 0.007) + np.multiply(L2, np.power(X, 2))

for time in range(age_length):
    period_end  = time + 120
    
    window_data = factors_portfolios.iloc[time:period_end]
    
    mean = window_data.mean()
    sigma = window_data.cov()

    portfolio = 0
    portfolio_weights = equall_weights(factors_number)
    weights[:, time, portfolio] = portfolio_weights
    
    portfolio += 1
    portfolio_weights = min_var(factors_number, sigma)
    weights[:, time, portfolio] = portfolio_weights

    portfolio += 1
    portfolio_weights = mean_var(factors_number, sigma, gam_, mean)
    weights[:, time, portfolio] = portfolio_weights

    portfolio += 1
    portfolio_weights = cvar(factors_number, window_data, b)
    weights[:, time, portfolio] = portfolio_weights

returns_tc = np.empty((factors_number, sample_length - period, number_of_portfolios)) 
for period_index in range(1, age_length):
    window_data = factors_portfolios.iloc[period_index:period_index + 120]
    benchmark_portfolio_values = benchmark_portfolio.iloc[period_index:period_index + 120].to_numpy()
    benchmark_portfolio_values = benchmark_portfolio_values.reshape((period, 1))

    for portfolio_index in range(number_of_portfolios):
        portfolio_weights = weights[:, period_index - 1, portfolio]

        portfolio_return = benchmark_portfolio_values + window_data * portfolio_weights.T
        transaction_cost = tc(portfolio_return, L1, L2)
        portfolio_return -=  transaction_cost
        returns_tc[:, period_index, portfolio] = portfolio_return.sum()

returns_tc = returns_tc.sum(axis = 0)

x_axis = out_of_sample["Date"][:age_length].values
portfolios_names = ["Market", "Equall weights", "Min-Variance", "Mean-Variance", "CVaR"]

benchmark_portfolio_returns = np.array([sum(benchmark_portfolio[index:index+120]) for index in range(age_length)])

print("With transactions cost: ")
for portfolio_index in range(number_of_portfolios):
    portfolio_return = returns_tc[:, portfolio_index]

    mean_ = portfolio_return.mean()
    std_ = np.sqrt(np.mean(abs(portfolio_return - portfolio_return.sum()/len(portfolio_return))**2))

    print("Mean of the {} portfolio: {}.".format(portfolios_names[portfolio_index + 1], mean_))
    print("Standart deviation of the {} portfolio: {}.".format(portfolios_names[portfolio_index + 1], std_))

    sharpe_ratio = mean_/std_
    print("Sharpe Ratio of the {} portfolio: {}.".format(portfolios_names[portfolio_index + 1], sharpe_ratio))

    var95 = np.quantile(portfolio_return, b)
    print("VaR5 of the {} portfolio: {}.".format(portfolios_names[portfolio_index + 1], var95))

#Task 8

market_data = data["Market"]
market_data_returns = np.sort([sum(market_data[index:index+120]) for index in range(len(market_data) - period)])

for value_index in range(20):
    print(f"Value #{value_index}: {market_data_returns[value_index]}.", "\n")

#Task 9
# Task 10
