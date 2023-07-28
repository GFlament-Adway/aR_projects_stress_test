from simulation import simulate_data
from proba_merton import calculate_probability_of_default
import matplotlib.pyplot as plt
from utils import load_barrier_data
from scipy.stats import norm
import random

if __name__ == "__main__":
    num_samples = 2
    T = 20
    Y_T = simulate_data(T)
    risk_grade_mapping = {
        'A': 0.01,
        'B': 0.05,
        'C': 0.1,
        'D': 1
    }
    barrier = norm.ppf(list(risk_grade_mapping.values()))
    pi_hat_list = random.choices(list(risk_grade_mapping.values()), weights=[0.5, 0.3, 0.2, 0]
                                 , k=num_samples)


