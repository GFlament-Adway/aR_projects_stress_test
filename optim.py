import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from proba_merton import calculate_probability_of_default, delta, get_x_g

def optimize_Y_t(G, rho, risk_grade_mapping):
    def objective(Y_t):
        d_sum = 0
        for g_1 in range(len(G)):
            for g in range(len(G)-1):
                d = (risk_grade_mapping[G[g+1]] - calculate_probability_of_default(G[g+1], Y_t))**2
                d /= delta(get_x_g(G[g+1]), get_x_g(G[g]), Y_t, rho) * (1 - delta(get_x_g(G[g+1]), get_x_g(G[g]), Y_t, rho))
                d_sum += d
        return d_sum

    result = minimize_scalar(objective)
    return result.x