import pandas as pd

# Load the CSV without headers
df = pd.read_csv('hand_signs_data.csv', header=None)

# Count number of columns
num_columns = df.shape[1]

# Generate column names
column_names = [f'x{i//2}_{"x" if i % 2 == 0 else "y"}' for i in range(num_columns - 1)]
column_names.append('label')  # Last column is the label

# Assign new headers
df.columns = column_names

# Save to a new CSV
df.to_csv('fixed_hand_signs_data.csv', index=False)

print("✅ CSV fixed and saved as 'fixed_hand_signs_data.csv'")
