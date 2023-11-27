# generate_test_spss.py
import pandas as pd
import pyreadstat  # Required for saving to SPSS format

# Generate sample data
n_samples = 1000
data = {
    'feature1': pd.Series(range(n_samples)),
    'feature2': pd.Series(range(n_samples)),
    'y': pd.Series(range(n_samples))
}

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame as SPSS file
spss_filename = 'test_data.spss'
pyreadstat.write_sav(df, spss_filename)

print(f"SPSS file '{spss_filename}' has been created.")
