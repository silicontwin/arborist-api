# generate_test_spss.py
import numpy as np
import pandas as pd
import pyreadstat  # Required for saving to SPSS format

# Set random number generator
rng = np.random.default_rng(101)

# Define sample size and number of features
n = 10000
p = 10

# Generate random uniform features
X = rng.uniform(0, 1, (n, p))

# Generate target variable 'y' based on a function of the features
y = X[:, 0] * 100 + X[:, 1] * 2 + rng.normal(0, 1, n)

# Create DataFrame
data = {'feature' + str(i+1): X[:, i] for i in range(p)}
data['y'] = y
df = pd.DataFrame(data)

# Save DataFrame as SPSS file
spss_filename = 'test_data.spss'
pyreadstat.write_sav(df, spss_filename)

print(f"SPSS file '{spss_filename}' has been created.")
