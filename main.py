import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

# Define the stocks in our portfolio

stocks = ['AAPL', 'GME', 'TSLA', 'AMZN', 'DB']

# Define the start and the end date

start_date = '2010-01-01'
end_date = '2022-01-01'

# Average trading days in a year
NUM_TRADING_DAYS = 252


def download_data():
    # Get values from the start of 2020 till 2022
    stock_data = {}
    for stock in stocks:
        # Closing prices for start and end date
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(stock_data)


def calculate_returns(data):
    # To measure values in comparable metric, we normalise hence the log
    log_returns = np.log(data / data.shift(1))
    return log_returns[1:]


def generate_portfolios(returns):
    ' This method will generate random portfolios for which we can ascertain the efficient frontier'

    portfolio_means = []
    portfolio_volatility = []
    portfolio_weights = []

    for _ in range(10000):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_volatility.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() *
                                                               NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_volatility)


def plot_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def statistics(weight, returns):
    portfolio_return = np.sum(returns.mean() * NUM_TRADING_DAYS)
    portfolio_volatility = np.sqrt(np.dot(weight.T, np.dot(returns.cov() *
                                                           NUM_TRADING_DAYS, weight)))

    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])


def min_function_sharpe(weights, returns):
    # We make the assumption that the risk-free assets is 0
    return -statistics(weights, returns)[2]


def optimise_portfolio(weights, returns):
    # The constraints of MPT is that the sum of the weights is 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # The weight could also be one if the stock is invested into a single item

    bounds = tuple((0, 1) for _ in range(len(stocks)))
    # This wil return the optimal portfolio with the highest Sharpe ratio
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds,
                                 constraints=constraints)


def print_optimal_portfolio(optimum, returns):
    print('Optimum Portfolio', optimum['x'].round(3))
    print('Expected return,volatility and sharpe ratio:', statistics(optimum['x'].round(3), returns))


if __name__ == '__main__':
    # Firstly get the data from the Yahoo Finance API
    dataset = download_data()

    # Normalise the data and then calculate the returns
    log_daily_returns = calculate_returns(dataset)

    # Generate random portfolios with different weights ratios
    weights, means, volatility = generate_portfolios(log_daily_returns)

    # Draw the portfolios
    plot_portfolios(means, volatility)

    # Apply optimisation techniques
    optimum = optimise_portfolio(weights, log_daily_returns)

    # Print optimal portfolio
    print_optimal_portfolio(optimum, log_daily_returns)
