import numpy as np

def simulate_data(T):

    # Covariates X_t ~ N([2, 0.5], I_2)
    mean_X = np.array([2, 0.5])
    cov_X = np.identity(2)
    X_t = np.random.multivariate_normal(mean_X, cov_X, T)

    # Calculate the tilde_Y_t | X_t
    beta = np.array([2, -1])
    tilde_Y_t = np.dot(X_t, beta) + np.random.normal(0, 1, T)

    # Calculate the mean and variance of tilde_Y_t
    mean_tilde_Y_t = np.mean(tilde_Y_t)
    var_tilde_Y_t = np.var(tilde_Y_t)

    # Calculate Y_t | X_t
    Y_t = (tilde_Y_t - mean_tilde_Y_t) / np.sqrt(var_tilde_Y_t)

    return Y_t
