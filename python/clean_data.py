import pandas as pd
import numpy as np

# Read the malformed CSV
with open('data/junit_data.csv', 'r') as f:
    lines = f.readlines()

# Filter and clean lines with exactly 3 numeric values
clean_data = []
for line in lines:
    values = line.strip().split(',')
    # Try to convert all values to float
    try:
        numeric_values = [float(v) for v in values if v.strip()]
        if len(numeric_values) == 3:
            clean_data.append(numeric_values)
    except ValueError:
        pass

if clean_data:
    df = pd.DataFrame(clean_data)
    print(f"Cleaned data shape: {df.shape}")
    print(df.head(10))
    
    # Save cleaned data
    df.to_csv('data/junit_data_cleaned.csv', header=False, index=False)
    print(f"\nSaved {len(df)} rows to junit_data_cleaned.csv")
else:
    print("No valid 3-column data found")
