# generate_test_spss.py
import numpy as np
import pandas as pd
import pyreadstat  # Required for saving to SPSS format
import os

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

# Define the filename for the SPSS file
spss_filename = 'test_data.spss'

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Define the file path for saving the SPSS file in the current directory
spss_file_path = os.path.join(current_dir, spss_filename)

# Save the DataFrame to SPSS in the current directory
pyreadstat.write_sav(df, spss_file_path)

print(f"SPSS file '{spss_file_path}' has been created.")
