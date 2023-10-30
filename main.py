import market


n_firms = 10
n_cons = 10000
T = 100

# Initialize the market object
market_object = market.MarketData(n_firms, n_cons, T)

market_object.generate_simulated_data()

print(f"The true parameters in this object are {market_object.alpha_sd}, {market_object.alpha_mean} and {market_object.beta_0}")