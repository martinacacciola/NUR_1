#!/usr/bin/env python
import numpy as np
from math import exp, log10

'''
This script containes the code for Exercise 1.
'''

def poisson_prob(l, k):
    ''' 
    Calculates the Poisson probability for a given lambda and k
    At each passage, we work with 32 bit floats and in log space to avoid overflow.
    By taking the logarithm, we transform multiplications into additions and divisions into subtractions.
    This helps keep the intermediate results within a manageable range.
    We then take the exponent of the final result, ensuring both numerical stability and correct results.
  
    Inputs:
    l: lambda value
    k: k value
    Output:
    Poisson probability for a given lambda and k

    We divide the calculation into:
    1. Calculate lamba^k in log space
    2. Calculate e^-lambda
    3. Calculate k! and the cumulative sum of log(k!) for all k from 2. 
    The reason for starting from 2 is that the first two values of k! are 0 and 1, not useful for the calculation.
    The cumulative sum is useful for next calculation
    4. Calculate the Poisson probability in log space and convert it back to normal space.
    
    '''

    k_values = np.float32(np.arange(k + 1))
    
    # 1. log(lamba^k)
    lam_k_log = np.float32(k_values * log10(l)) 
    # 2. e^-lambda
    e_neg_lambda = np.float32(log10(exp(-l))) 
    # 3. Factorial calculation: log(k!) and cumulative sum of log(k!) for all k from 2
    log_fact_k = np.float32(np.log10(np.arange(2, k + 1)))
    tot_log_fact = np.float32(np.cumsum(log_fact_k))
    # 4. Poisson probability in log space and conversion
    log_p_k = np.float32(lam_k_log[2:] + e_neg_lambda - tot_log_fact)
    # Terms k=0,1 are calculated separately to maintain accuracy
    # The remaining k are obtained by exponentiating the logarithmic Poisson probabilities
    poisson_distribution = np.float32(np.concatenate([[(l**k) * exp(-l)] for k in k_values[:2]] + [np.power(10, log_p_k)]))
    
    return poisson_distribution

# Test the function for the given values
parameter_values = [(1, 0), (5, 10), (3, 21), (2.6, 40), (101, 200)]

# Save Poisson probability results to a text file
output_data = []
for lambda_val, k_val in parameter_values:
    poisson_distribution = poisson_prob(lambda_val, k_val)
    output_data.append([lambda_val, k_val, poisson_distribution[int(k_val)]])

np.savetxt('poisson_output.txt', np.array(output_data), header='Lambda\tK\tPoisson Probability', delimiter='\t')
