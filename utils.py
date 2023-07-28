import pandas as pd

def load_barrier_data(file_path):
    # Load risk grade data from CSV file and map to numerical barriers
    risk_grade_mapping = {
        'A': 0.01,
        'B': 0.05,
        'C': 0.1,
        'D':1
        # Add more risk grades and their numerical barriers as needed
    }
    data = pd.read_csv(file_path)
    barrier_data = data['Risk Grade'].map(risk_grade_mapping).values
    return barrier_data