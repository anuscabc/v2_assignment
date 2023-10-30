import numpy as np
import pandas as pd

class MarketData: 
    """
    This object represents a market environment that can be used to generate
    simulated data, used for heterogeneous product market demand estimation. 
    The object can be easily used py practitioners to generate data for the 
    purpose of testing their econometric techniques.

    The data generation process can have: 
    1. no random coefficients (simple logit model)
    2. random coefficients on price
    """
    def __init__(
            self, 
            n_firms:int, 
            n_consumers:int,
            T:int, 
            n_chars:int =1,
            x_min:float = 4.,
            x_max:float = 5.,
            c_mean:float = 1.2,
            c_sd:float = 0.01,
            xi_mean:float = 0., 
            xi_sd:float = 0.2,
            price_sd:float = 0.02, 
            gamma:float = 0.9, 
            alpha_mean:float = -1.7, 
            alpha_sd:float = 0.5, 
            seed:int = 100
        ):

        # Set seed of the simulation for replicability
        self.seed = seed
        np.random.seed(self.seed)

        # General market characteristics
        self.n_firms = n_firms
        self.n_consumers = n_consumers
        self.n_chars = n_chars
        self.T = T

        # Generate cost data 
        self.c_mean = c_mean
        self.c_sd = c_sd
        self.costs = np.random.lognormal(self.c_mean, self.c_sd, 
                                         size=(self.n_firms*self.T, 1))

        # Generate exogenous demand shock that affect prices
        self.xi_mean = xi_mean
        self.xi_sd = xi_sd
        self.xi = np.random.normal(self.xi_mean, self.xi_sd, 
                                   (self.n_firms*self.T, 1))

        # Generate price data from cost and product fixed effects 
        self.price_sd = price_sd
        self.gamma = gamma 
        self.prices = np.random.lognormal(
                        self.gamma*self.xi + 0.5*self.costs,
                        self.price_sd, size=(self.n_firms*self.T, 1)
                    )

        # Firm-level market characteristics
        self.market_shares = np.zeros((self.n_firms*self.T, 1))
        self.mean_indirect_utilities = np.zeros((self.n_firms*self.T, 1))
        self.profits = np.zeros((self.n_firms*self.T, 1))
        self.markups = np.zeros((self.n_firms*self.T, 1))

        # Generate the product characteristics
        self.x_min = x_min
        self.x_max = x_max
        self.produc_chars = np.random.uniform(self.x_min, self.x_max, (self.n_firms, self.n_chars))

        # Generate random coefficients 
        # 1. The random coefficients on price: 
        self.alpha_mean = alpha_mean
        self.alpha_sd = alpha_sd

        # 2. No random coeff on product characteristics: 
        self.beta_0 = -1.

        # 3. Generating the random shocks for the model
        self.v_p = np.random.normal(0, 1, (self.n_consumers*self.T, 1))
        self.alpha_i = np.zeros((self.n_consumers*self.T, 1)) 
        self.random_coeff_price = np.zeros((self.n_consumers*self.n_firms*self.T, 1))

        # Fill in the attributes that we initialized above
        self.gen_price_random_coeff()
        self.gen_all_time_period_mean_indirect_utilities()
        self.gen_market_share()


    def gen_market_share(self):
        """
        TODO
        """
        # TODO add comments
        for t in range(self.T):
            c_times_f = self.n_consumers*self.n_firms
            coeff_per_T = self.random_coeff_price[t*c_times_f : (t + 1) * c_times_f]
            mean_utility_per_T = self.mean_indirect_utilities[t*self.n_firms : (t + 1)*self.n_firms] 
            repeat_u = np.repeat(mean_utility_per_T, self.n_consumers, axis=0)
                                 
            u = repeat_u + coeff_per_T 
            u_r = np.reshape(u, (self.n_firms, self.n_consumers))
            sum_u = np.sum(np.exp(u_r), axis=0)

            all_probs = np.exp(u_r)/(1 + sum_u)
            market_shares = np.sum(all_probs, axis=1)/self.n_consumers

            self.market_shares[t*self.n_firms : (t + 1)*self.n_firms] = np.reshape(market_shares, (self.n_firms, 1))
    
    def gen_all_time_period_mean_indirect_utilities(self): 
        """
        TODO
        """
        for t in range(self.T):
            price_r = self.prices[t*self.n_firms : (t + 1)*self.n_firms]
            mean_indirect_utilities_period = (
                self.produc_chars*self.beta_0 
                + self.alpha_mean*price_r 
                + self.xi[t*self.n_firms : (t + 1) * self.n_firms]
            )
            self.mean_indirect_utilities[t*self.n_firms:(t + 1)*self.n_firms] = mean_indirect_utilities_period

    def gen_price_random_coeff(self): 
        """
        TODO
        """
        for t in range(self.T):
            price_r = self.prices[t*self.n_firms : (t + 1)*self.n_firms].reshape(1, self.n_firms)
            period_v_p = self.v_p[t*self.n_consumers : (t + 1)*self.n_consumers]
            alpha_i_per_period = np.reshape(self.alpha_sd*period_v_p, 
                                             (self.n_consumers, 1))
            c_times_f = self.n_consumers*self.n_firms
            self.alpha_i[t*self.n_consumers: (t + 1)*self.n_consumers] = alpha_i_per_period 
            self.random_coeff_price[t*c_times_f : (t + 1)*c_times_f] = np.reshape(np.ravel(
                                        (alpha_i_per_period*price_r).T), (c_times_f, 1))
            

    def generate_simulated_data(self):


        """
        Saves dataframe with simulation data 
        # Generating a dataframe that ccan be used for estimation in Python that has the following values: 
        # Time period - t 
        # Firm id - i
        # Market share - s
        # Price - p 
        # Costs - c
        # Characteristics - All the characteristics that you have generated depending on the function
        """
        
        # This is such that data nicely stored
        time1 = np.reshape(np.repeat(np.array(range(1, self.T+1)), self.n_firms), (self.n_firms*self.T, 1))
        products1 = np.reshape(np.tile(np.array(range(1, self.n_firms+1)), self.T), (self.n_firms*self.T, 1))

        # For the product characteristics 
        product1 = np.tile(self.produc_chars,(self.T, 1))
        char_names = [f'char{i+1}' for i in range(0, self.n_chars)]
        df_products = pd.DataFrame(product1, columns=char_names)
    
        # Generate the dataframe with all the information
        df_simulation = pd.DataFrame(
            {'market_ids': time1.T[0],
             'firm_ids':products1.T[0],
             'shares':self.market_shares.T[0],
             'prices':self.prices.T[0], 
             'cost':self.costs.T[0],
             'xi':self.xi.T[0]}
        )
        df_final = df_simulation.merge(df_products, left_index=True, right_index=True)
        df_final.to_csv('simulation_data.csv', index=False)
        return df_final
    
    


    def __str__(self) -> str:
        return f"Market with {self.n_firms} firms and {self.n_consumers} consumers over {self.T} time periods. \n Firms sell differentiated product which have {self.n_chars} product characteristics"
    

    