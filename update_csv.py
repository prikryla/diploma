import pandas as pd

# Read the CSV file
df = pd.read_csv('modified_file.csv')

# Replace single quotes in the 'column_name' column with an empty string
df['title'] = df['title'].str.replace('"', '')

# Save the modified DataFrame to a new CSV file
df.to_csv('modified_file.csv', index=False)
