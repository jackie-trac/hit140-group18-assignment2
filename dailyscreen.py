import pandas as pd
import scipy.stats as st
import numpy as np
import statsmodels.stats.weightstats as stm
import math
import matplotlib.pyplot as plt

#Load the datasets in DataFrame
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

#Merge datasets on a common identifier
merged_df = pd.merge(df1, df2, on='ID', how='inner') 
merged_df = pd.merge(merged_df, df3, on='ID', how='inner')

# Calculate total screen time for weekdays and weekends
merged_df['weekday_screen_time'] = (merged_df['C_wk'] + merged_df['G_wk'] + merged_df['S_wk'] + merged_df['T_wk']) * 5
merged_df['weekend_screen_time'] = (merged_df['C_we'] + merged_df['G_we'] + merged_df['S_we'] + merged_df['T_we']) * 2

# Calculate total screen time for the entire week
merged_df['total_screen_time_per_week'] = merged_df['weekday_screen_time'] + merged_df['weekend_screen_time']

# Calculate average daily screen time
merged_df['average_daily_screen_time'] = merged_df['total_screen_time_per_week'] / 7

#Calculate descriptive to define screen time groups
low_threshold = merged_df['average_daily_screen_time'].quantile(0.25)
high_threshold = merged_df['average_daily_screen_time'].quantile(0.75)
average = merged_df['average_daily_screen_time'].mean()
median_value = merged_df['average_daily_screen_time'].median()
min_value = merged_df['average_daily_screen_time'].min()
max_value = merged_df['average_daily_screen_time'].max()
print("Looking at the average daily screen time amongst the sample:")
print("\tOn average: %.2f hours daily" % average)
print("\tMedian value: %.2f hours daily" % median_value)
print("\tAt 25%% IQR: %.2f hours daily" % low_threshold)
print("\tAt 75%% IQR: %.2f hours daily" % high_threshold)
print("\tMin: %.2f hours daily" % min_value)
print("\tMax: %.2f hours daily" % max_value)
print("From these thresholds, we determine the screen time groups as follow:\n\tLow: Less than %.2f hours a day\n\tMedium: From %.2f hours to %.2f hours a day\n\tHigh: More than %.2f hours a day"%(low_threshold,low_threshold,high_threshold,high_threshold))


# Categorize participants into Low, Medium, and High screen time groups
merged_df['screen_time_group'] = pd.cut(merged_df['average_daily_screen_time'], 
                                 bins=[-float('inf'), low_threshold, high_threshold, float('inf')],
                                 labels=['Low', 'Medium', 'High'])

#Gather well-being indicator columns
well_being_columns = ['Optm','Usef','Relx','Intp','Engs','Dealpr','Thcklr','Goodme','Clsep','Conf','Mkmind','Loved','Intthg','Cheer']  

# Create composite well-being score by averaging the indicators horizontally
merged_df['composite_well_being'] = merged_df[well_being_columns].mean(axis=1)

import seaborn as sns

# Create subplots to display histograms for each screen time group
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Define screen time groups
screen_time_groups = ['Low', 'Medium', 'High']

# Plot histograms for each screen time group with KDE
for i, group in enumerate(screen_time_groups):
    subset = merged_df[merged_df['screen_time_group'] == group]['composite_well_being']
    sns.histplot(subset, kde=True, ax=axes[i], color=['green', 'orange', 'red'][i])
    axes[i].set_title(f'{group} Screen Time')
    axes[i].set_xlabel('Composite Well-being Score')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Define a function to calculate confidence interval
def confidence_interval(data, confidence=0.99):
    n = len(data) #Sample size
    x_bar = np.mean(data) #Sample mean
    s = np.std(data, ddof=1) #Sample standard deviation
    std_err = s/ math.sqrt(n)  # Standard error of the mean
    sig_lvl = 1 - confidence #Significant level
    ci_low_z, ci_upp_z = stm._zconfint_generic(x_bar,std_err,alpha=sig_lvl, alternative="two-sided")
    return x_bar, ci_low_z, ci_upp_z

# Calculate confidence intervals for each screen time group
ci_results = merged_df.groupby('screen_time_group',observed=False)['composite_well_being'].apply(confidence_interval).reset_index()

# Unpack the confidence intervals into separate columns
ci_results[['well-being mean', 'lower_bound', 'upper_bound']] = pd.DataFrame(ci_results['composite_well_being'].tolist(), index=ci_results.index)

# Drop the original column that contained the tuples and print the data
ci_results = ci_results.drop(columns=['composite_well_being'])
print("Confidence Intervals for composite well-being score by screen time group:")
print(ci_results)






# Plotting confidence intervals
plt.errorbar(ci_results['screen_time_group'], ci_results['well-being mean'], 
             yerr=[ci_results['well-being mean'] - ci_results['lower_bound'], ci_results['upper_bound'] - ci_results['well-being mean']], 
             fmt='ro', ecolor='blue', capsize=7, capthick=2.5)
plt.xlabel('Screen Time Group')
plt.ylabel('Mean of Composite Well-being Score')
plt.title('Confidence Intervals for Well-being Mean by Screen Time Group')
plt.show()


