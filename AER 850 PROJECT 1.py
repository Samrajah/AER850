
import pandas as pd

# Specify the file path of your CSV file
file_path = "Project 1 Data.csv"

# Read the CSV file and create a DataFrame
df = pd.read_csv(file_path)

# Now, you can perform various data analysis and manipulation operations on the DataFrame.
# For example, you can print the first few rows of the DataFrame using the head() method:
print(df.head())
