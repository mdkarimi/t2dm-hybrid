# generate_synthetic_data.py
# Synthetic Data Generation using CTGAN for T2DM Prediction
# License: MIT (https://opensource.org/licenses/MIT)

import pandas as pd
from ctgan import CTGAN
from scipy.stats import ks_2samp
import numpy as np

# Load NHANES for reference
def load_nhanes():
    nhanes = pd.read_csv('nhanes.csv')  # Replace with NHANES path
    nhanes = nhanes[nhanes['age'].between(35, 45)]
    nhanes.fillna(nhanes.median(), inplace=True)
    return nhanes

# Generate synthetic data
def generate_synthetic_data(n_samples=5000):
    nhanes = load_nhanes()
    ctgan = CTGAN(epochs=100, batch_size=100, generator_dim=(128, 256, 128), discriminator_dim=(256, 128, 64))
    ctgan.fit(nhanes)
    synthetic_data = ctgan.sample(n_samples)
    synthetic_data.to_csv('synthetic_t2dm.csv', index=False)
    return synthetic_data, nhanes

# Validate synthetic data
def validate_synthetic_data(synthetic_data, real_data):
    results = {}
    for col in ['fpg', 'hba1c', 'bmi', 'waist_circ']:
        stat, p = ks_2samp(synthetic_data[col], real_data[col])
        kl_div = np.mean(np.log2(synthetic_data[col].value_counts(normalize=True) / real_data[col].value_counts(normalize=True)))
        results[col] = {'KS_p_value': p, 'KL_divergence': kl_div}
    return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    synthetic_data, real_data = generate_synthetic_data()
    validation_results = validate_synthetic_data(synthetic_data, real_data)
    print("Validation Results:\n", validation_results)
    validation_results.to_csv('validation_results.csv', index=False)