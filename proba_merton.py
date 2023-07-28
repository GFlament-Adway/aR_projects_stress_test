import numpy as np
from scipy.stats import norm


def get_x_g(risk_grade_mapping, grade):
    return norm.ppf(risk_grade_mapping[grade])

def delta(x_g_plus_1, x_g, Y_t, rho):
    return norm.cdf((x_g_plus_1 - np.sqrt(rho) * Y_t) / np.sqrt(1 - rho)) - norm.cdf((x_g - np.sqrt(rho) * Y_t) / np.sqrt(1 - rho))

def calculate_probability_of_default(Y_T, pi_hat_list):
    prob_default_list = [[0 for _ in range(len(Y_T))] for _ in range(len(pi_hat_list))]
    for j,t  in enumerate(Y_T):
        for i,pi_hat in enumerate(pi_hat_list):
            # Calculate rho
            rho = 1.25 * 0.12 * ((1 - np.exp(-50 * pi_hat)) / (1 - np.exp(-50))) + \
                  1.25 * 0.24 * ((1 - (1 - np.exp(-50 * pi_hat))) / (1 - np.exp(-50)))

            # Calculate phi_inverse(pi_hat)
            phi_inverse_pi_hat = norm.ppf(pi_hat)

            # Calculate probability of default (pi(y_T))
            prob_default = norm.cdf((phi_inverse_pi_hat - np.sqrt(rho) * t) / np.sqrt(1 - rho))
            prob_default_list[i][j] = prob_default

    return prob_default_list