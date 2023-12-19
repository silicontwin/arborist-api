# generate_test_csv.py
import numpy as np
import pandas as pd
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

# Define the filename for the CSV
csv_filename = 'test_data.csv'

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Define the file path for saving the CSV in the current directory
csv_file_path = os.path.join(current_dir, csv_filename)

# Save the DataFrame to CSV in the current directory
df.to_csv(csv_file_path, index=False)  # index=False to avoid writing row indices

print(f"CSV file '{csv_file_path}' has been created.")
