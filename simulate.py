import numpy as np
import pandas as pd

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
            c_mean:float=1.2,
            c_sd:float=0.05,
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
        self.costs = np.random.lognormal(self.c_mean, self.c_sd, size=(self.n_firms*self.T, 1))


        # Generate exogenous demand shock that affects prices
        #  product-market-specific unobserved fixed effect
        self.xi_mean = xi_mean
        self.xi_sd = xi_sd
        self.xi = np.random.normal(self.xi_mean, self.xi_sd, (self.n_firms*self.T, 1))

        # Generate price data from cost and product fixed effects 
        self.price_sd = price_sd
        # This is another parameter of how much the current price affects the product characteristics 
        self.gamma = gamma 
        self.prices = np.random.lognormal(self.gamma*self.xi + self.costs,
                                       self.price_sd, size=(self.n_firms*self.T, 1))





        # Generate the price data from costs and the exogenous demand shock 
        # Prices are not an equilibirum price, they are just coming from an easy pricing rule that takes into account costs and 

        # These are values to be filled in once the rest of the data is produced 
        self.market_shares = np.zeros((self.n_firms*self.T, 1))
        self.mean_indirect_utilities = np.zeros((self.n_firms*self.T, 1))
        self.profits = np.zeros((self.n_firms*self.T, 1))
        self.markups = np.zeros((self.n_firms*self.T, 1))



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
        self.alpha_0_sd = (np.exp(alpha_sd**2) - 1) * np.exp(2*alpha_mean + alpha_sd**2)

        # 2. The random coefficients on all other product characteristics: 

        self.beta_0 = np.random.normal(2, 1, (self.n_chars, 1))
        self.beta_sd = np.absolute(np.random.gumbel(0, 0.04, (self.n_chars, 1)))

        # 3. Generating the random shocks for the model -> might be better that they 
        # generate in module rather than as function attribute if they do not need to write t
        # themselves the code 
        self.v_p = np.random.normal(0, 1, (self.n_consumers*self.T, 1))
        # Getting all the mean indirect utilities: 
        self.random_coeff_price = np.zeros((self.n_consumers*self.n_firms*self.T, 1))
        self.random_coeff_car = np.zeros((self.n_consumers*self.n_firms*self.T, 1))
        self.all_random_coeff = np.zeros((self.n_consumers*self.n_firms*self.T, 1))




        # Fill in the attributes that we initialized above
        self.gen_all_random_coeff()
        self.gen_all_time_period_mean_indirect_utilities()
        self.gen_market_share()
    
     
    def gen_market_share(self):
        """
        """
        for t in range(0, self.T, 1):
            coeff_per_T = self.all_random_coeff[t*self.n_consumers*self.n_firms:self.n_consumers*self.n_firms+ t*self.n_consumers*self.n_firms]
            mean_utility_per_T = self.mean_indirect_utilities[t*self.n_firms:self.n_firms+ t*self.n_firms] 
            repeat_u = np.repeat(mean_utility_per_T, self.n_consumers, axis=0)

            u = repeat_u  + coeff_per_T 
            u_r = np.reshape(u, (self.n_firms, self.n_consumers))
            sum_u = np.sum(np.exp(u_r), axis =0)

            all_probs = np.exp(u_r)/(1 + sum_u)
            market_shares = np.sum(all_probs, axis=1)/self.n_consumers

            self.market_shares[t*self.n_firms:self.n_firms+ t*self.n_firms] = np.reshape(market_shares, (self.n_firms, 1))
    

    def gen_all_time_period_mean_indirect_utilities(self): 
        """
        """
        for t in range(0, self.T, 1):
            price_r = self.prices[t*self.n_firms:self.n_firms+t*self.n_firms]
            mean_indirect_utilities_period = self.produc_chars@self.beta_0 + self.alpha_0*price_r + self.xi[t*self.n_firms:t*self.n_firms+self.n_firms]
            self.mean_indirect_utilities[t*self.n_firms:self.n_firms+ t*self.n_firms] = mean_indirect_utilities_period

    def gen_price_random_coeff(self): 
        """
        """
        for t in range(0, self.T, 1):
            price_r = self.prices[t*self.n_firms:self.n_firms+t*self.n_firms].reshape(1, self.n_firms)
            period_v_p = self.v_p[t*self.n_consumers:self.n_consumers+ t*self.n_consumers]
            alpha_i_per_period = np.reshape((-(np.exp(self.alpha_mean+ self.alpha_sd*period_v_p))+np.exp(self.alpha_mean + (self.alpha_sd)**2/2)), (self.n_consumers, 1))
            self.random_coeff_price[t*self.n_consumers*self.n_firms
                                    :self.n_consumers*self.n_firms+
                                      t*self.n_consumers*self.n_firms] = np.reshape(np.ravel(
                                          (alpha_i_per_period*price_r).T), (self.n_consumers*self.n_firms, 1))
        # return self.random_coeff_price

    def gen_char_random_coefficient(self):
        """
        """
        
        for t in range(0, self.T, 1): 
            v_p_period = self.v_p[t*self.n_consumers:
                                  self.n_consumers+t*self.n_consumers].reshape(1, self.n_consumers)
            random_car= self.beta_sd*v_p_period
            cars = np.reshape(
                np.ravel(np.matmul(self.produc_chars,random_car).T)
                , (self.n_firms*self.n_consumers, 1))
            self.random_coeff_car[t*self.n_consumers*self.n_firms:self.n_consumers*self.n_firms+ t*self.n_consumers*self.n_firms] = cars

    
    def gen_all_random_coeff(self):
        """
        """

        self.gen_price_random_coeff()
        self.gen_char_random_coefficient()
        self.all_random_coeff = self.random_coeff_price + self.random_coeff_car
    

    # Generating a dataframe that ccan be used for estimation in Python that has the following values: 
    # Time period - t 
    # Firm id - i
    # Market share - s
    # Price - p 
    # Costs - c
    # Characteristics - All the characteristics that you have generated depending on the function


    def generate_simulated_data(self):
        """Saves dataframe with simulation data 
        """
        
        # This is such that data nicely stored
        time1 = np.reshape(np.repeat(np.array(range(1, self.T+1)), self.n_firms), (self.n_firms*self.T, 1))
        products1 = np.reshape(np.tile(np.array(range(1, self.n_firms+1)), self.T), (self.n_firms*self.T, 1))

        # For the product characteristics 
        product1 = np.tile(self.produc_chars,(self.T, 1))
        char_names = [f'char{i+1}' for i in range(0, self.n_chars)]

        
        df_products = pd.DataFrame(product1, columns=char_names)
    


        # Generate the dataframe with all the information
        df_simulation = pd.DataFrame({'market_ids': time1.T[0],
                                    'firm_ids':products1.T[0],
                                    'market_share':self.market_shares.T[0],
                                    'price':self.prices.T[0], 
                                    'cost':self.costs.T[0],
                                    'xi':self.xi.T[0]
                                    })
        

        df_final = df_simulation.merge(df_products, left_index=True, right_index=True)

        # Add the product characteristics 
        return df_final

    # The function that you call when you print the object 
    def __str__(self) -> str:
        return f"Market with {self.n_firms} firms and {self.n_consumers} consumers over {self.T} time periods. \n Firms sell differentiated product which have {self.n_chars} product characteristics"
    

    