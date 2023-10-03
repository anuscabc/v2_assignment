import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

class Market_Data: 
      # definition of the method also with stype checking 
    """This is an onject containing a market simulation
    The object can be easily used py practitioners to generate data for the purpose of 
    testing their econometric techniques.
    The data generation process can have: 
    1. no random coefficients (simple logit model)
    2. random coefficients logit model on all coefficients

    -> The idea is to provide an intuitive example of how it is possible to use Monte Carlo simulation 
    (intro to making it more approachable and decrease barrier to entry into structural estimation)

    """
    def __init__(
            self, 
            n_firms:int, 
            n_consumers:int,
            n_chars:int,
            T:int, 
            x_min:float=1.,
            x_max:float=6.,
            c_mean:float=0.,
            c_sd:float=0.5,
            xi_mean:float = 0., 
            xi_sd:float = 0.2,
            price_sd = 0.05, 
            gamma = 0.9, 
            alpha_mean = 0.5, 
            alpha_sd = 0.05, 
            seed:int=100

        ):

        # Setting the seed of the simulation for replicability
        self.seed = seed
        np.random.seed(self.seed)

        self.n_firms = n_firms
        self.n_consumers = n_consumers
        self.n_chars = n_chars
        self.T = T

        # Generate cost data 
        self.c_mean = c_mean
        self.c_sd = c_sd
        self.costs = np.random.gumbel(self.c_mean, self.c_sd, size=(self.n_firms*self.T, 1))


        # Generate exogenous demand shock that affects prices
        #  product-market-specific unobserved fixed effect
        self.xi_mean = xi_mean
        self.xi_sd = xi_sd
        self.xi = np.random.normal(self.xi_mean, self.xi_sd, (self.n_firms*self.T, 1))

        # Generate price data from cost and product fixed effects 
        self.price_sd = price_sd
        # This is another parameter of how much the current price affects the product characteristics 
        self.gamma = gamma 
        self.prices = np.random.gumbel(self.gamma*self.xi + self.costs,
                                       self.price_sd, size=(self.n_firms*self.T, 1))





        # Generate the price data from costs and the exogenous demand shock 
        # Prices are not an equilibirum price, they are just coming from an easy pricing rule that takes into account costs and 

        # These are values to be filled in once the rest of the data is produced 
        self.market_shares = np.zeros((self.n_firms, self.T))
        self.mean_indirect_utilities = np.zeros((self.n_firms, self.T))
        self.profits = np.zeros((self.n_firms, self.T))
        self.markups = np.zeros((self.n_firms, self.T))



        # Randomly generate stochastic elements
        # parameters for the characterisitcs 
        self.x_min = x_min
        self.x_max = x_max
        # Generate the product characteristics
        self.produc_chars = np.random.uniform(self.x_min, self.x_max, (self.n_firms, self.n_chars))


        # Generate the stuff for the random coefficients 
        # 1. The random coefficients on price: 
        self.alpha_mean = alpha_mean
        self.alpha_sd = alpha_sd
        self.alpha_0 = -np.exp(self.alpha_mean + (self.alpha_sd)**2/2)

        # 2. The random coefficients on all other product characteristics: 

        self.beta_0 = np.random.normal(2, 1, (self.n_chars, 1))


    
        # Getting all the mean indirect utilities: 






    def gen_mean_indirect_utilities(self, t): 
        """Comutation of mean indirect utilities per period

        Args:
            t (_type_): Timde period for the computation 

        Returns:
            _type_: vector (n_firms) mean indirect utilities vector
        """
        price_r = self.prices[t:t+self.T]
        mean_indirect_utilities = self.produc_chars@self.beta_0 + self.alpha_0*price_r + self.xi[t*self.T:t*self.T+self.T]
        return mean_indirect_utilities
    


    def gen_mean_utilities_all_probs_market_shares(self, t): 
        """Function that generates mean utilities on which to perform the estimation"""
        # Fill in the mean indirect utility vector 
        price_r = np.reshape(self.prices[t:t+self.T], (1, self.n_firms))
        # alpha_0 = -np.exp(self.mu + (self.sigma)**2/2)

        # beta = np.array([self.beta1, self.beta2, self.beta3])
        mean_indirect_utility = self.produc_chars@beta + alpha_0*(price)
        mean_indirect_utlity_for_utility = np.repeat(mean_indirect_utility, self.n_consumers, axis=0)

        alpha_i = np.reshape((-(np.exp(self.mu + self.sigma*v_p))+np.exp(self.mu + (self.sigma)**2/2)), (self.n_consumers, 1))
        random_coeff = np.ravel((alpha_i*price_r).T)

        u = mean_indirect_utlity_for_utility + random_coeff + e
        u_r = np.reshape(u, (self.n_firms, self.n_consumers))
        sum_u = np.sum(np.exp(u_r), axis =0)

        all_probs = np.exp(u_r)/(1 + sum_u)
        market_shares = np.sum(all_probs, axis=1)/self.n_consumers

        return market_shares, all_probs, mean_indirect_utility
    


    def __str__(self) -> str:
        return f"Market with {self.n_firms} firms and {self.n_consumers} consumers over {self.T} time periods. \n Firms sell differentiated product which have {self.n_chars} product characteristics"
    

    