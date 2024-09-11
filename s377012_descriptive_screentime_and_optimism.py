import pandas as pd

import matplotlib.pyplot as plt

df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

# First 5 rows of each DataFrame
print("First 5 rows of dataset1:\n", df1.head().to_markdown(index=False))
print("\nFirst 5 rows of dataset2:\n", df2.head().to_markdown(index=False))
print("\nFirst 5 rows of dataset3:\n", df3.head().to_markdown(index=False))

print("\nColumn names and their data types for dataset1:\n")
print(df1.info())

print("\nColumn names and their data types for dataset2:\n")
print(df2.info())

print("\nColumn names and their data types for dataset3:\n")
print(df3.info())

# Merge df2 and df3 on the 'ID' column
merged_df = pd.merge(df2, df3, on='ID')

# Merge merged_df and df1 on the 'ID' column
final_df = pd.merge(merged_df, df1, on='ID')

# First 5 rows of the final_df
print(final_df.head().to_markdown(index=False))

#  Mean for each screen time column
mean_screen_time = final_df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].mean().round(2)

# Standard deviation for each screen time column
std_screen_time = final_df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].std().round(2)

# Quantiles for each screen time column
quantiles_screen_time = final_df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].quantile([0, 0.25, 0.5, 0.75, 1])

print("\nMean Screen Time:\n", mean_screen_time.to_markdown())
print("\nStandard Deviation of Screen Time:\n", std_screen_time.to_markdown())
print("\nQuantiles of Screen Time:\n", quantiles_screen_time.to_markdown())

# Correlation coefficients between `Optm` and screen time columns
correlation_with_optimism = final_df[['Optm', 'C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].corr()['Optm'].round(4)

print("\nCorrelation with Optimism:\n", correlation_with_optimism.to_markdown())

# Histogram for the `Optm` column
plt.figure(figsize=(10, 6))
plt.hist(final_df['Optm'], bins=5, edgecolor='black') 

# Title and labels
plt.title('Distribution of Optimism Levels')
plt.xlabel('Optimism Level (1-5)')
plt.ylabel('Frequency')

#  Plot
plt.show()
