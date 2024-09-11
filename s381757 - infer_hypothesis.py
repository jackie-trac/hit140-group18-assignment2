import pandas as pd
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

# read 3 data sets
data1 = pd.read_csv('dataset1.csv')
data2 = pd.read_csv('dataset2.csv')
data3 = pd.read_csv('dataset3.csv')

# We merge 2 data sets:
# Data set 1 & 2
data1_2 = pd.merge(data1, data2, on ='ID', how = 'inner')
# Data set 1 & 3
data1_3 = pd.merge(data1, data3, on ='ID', how = 'inner')

# ANALYSIS: Screen time based on Deprivation

print(""" ---------------------- First Analysis ----------------------
\nWe set the Null hypothesis as:
'Do children/adolescent with deprivation status(less fortunate households) use more screen time than others?'
""")

# assume
# df1 = high deprivation group or less fortunate group in dataset 1 & 2
# df2 = others group in dataset 1 & 2

df1 = data1_2[data1_2["deprived"]==1]
df2 = data1_2[data1_2["deprived"]==0]

#select 8 columns (all screen time calculated) from C_we to T_wk
scr_columns = data1_2.loc[:, 'C_we':]

# Create a data list to append screentime mean of high deprivation and others
scr_data = []

# Run a loop to calculate the mean, standard deviation, total sample size, t-statistic and p-value
for col in scr_columns:
    sample1 = df1[col].to_numpy()
    sample2 = df2[col].to_numpy()

    x_bar1 = st.tmean(sample1)
    s1 = st.tstd(sample1)
    n1 = len(sample1)

    x_bar2 = st.tmean(sample2)
    s2 = st.tstd(sample2)
    n2 = len(sample2)

    t_stats, p_val = st.ttest_ind_from_stats(x_bar1, s1, n1,
                                            x_bar2, s2, n2,
                                            equal_var= False,
                                            alternative = 'greater')
    
    # Create a dataframe to store all datas for comparison and conclusion
    scr_data.append((col, round(x_bar1, 10), round(x_bar2, 10), round(p_val, 100)))
    
result_df = pd.DataFrame(scr_data, columns=['Activity', 'High deprivation mean', 'Others mean', 'P-value'])

print("""\t The results are shown in below dataframe: 
""")
print(result_df)
print("""
Conclusion: \nSince all p-values are less than 0.05, we reject the Null hypothesis and confirm that:
'Children/Adolescent with deprivation status or less fortunate households use more screen time than others'
""")


# ANALYSIS: Education related well-being based on Deprivation
# All calculations are the same so there won't be any comments below

print(""" ---------------------- Second Analysis ----------------------
\nWe set the Null hypothesis as:
'Is there any difference in Education related Well-being of children/adolescent 
between those with deprivation status and others?'

We compare three Indicators: Energy, Problems dealing, Clear thinking
""")

df3 = data1_3[data1_3["deprived"]==1]
df4 = data1_3[data1_3["deprived"]==0]

wb_columns = data1_3[['Engs', 'Dealpr', 'Thcklr']]

wb_data = []

for col2 in wb_columns:
    sample3 = df3[col2].to_numpy()
    sample4 = df4[col2].to_numpy()

    x_bar3 = st.tmean(sample3)
    s3 = st.tstd(sample3)
    n3 = len(sample3)

    x_bar4 = st.tmean(sample4)
    s4 = st.tstd(sample4)
    n4 = len(sample4)

    t_stats2, p_val2 = st.ttest_ind_from_stats(x_bar3, s3, n3,
                                            x_bar4, s4, n4,
                                            equal_var= False,
                                            alternative = 'two-sided')
    
    wb_data.append((col2, round(x_bar3, 10), round(x_bar4, 10), round(p_val2, 3)))

print("""\t The results are shown in below dataframe: 
""")

result_df2 = pd.DataFrame(wb_data, columns=['Well-being', 'High deprivation mean', 'Others mean', 'P-value'])
print("---------------------")
print(result_df2)

print("""
Conclusion:
Regarding 'Energy', since the p-value is more than 0.05, we cannot reject the Null hypothesis.
Therefore, there is no difference in Energy level based on Deprivation status.

Regarding 'Problems dealing' and 'Clear thinking', the p-value for these 2 indicators are less than 0.05, from that we can conclude:
There is a difference between those who are less fortunate and privileged.
""")

# HISTOGRAM 1: Demonstration of Screen-time and Deprivation status
# we have already defined "scr_columns = data1_2.loc[:, 'C_we':]"

fig, axes = plt.subplots(4, 2, figsize=(10, 10))
axes = axes.flatten()
x_ticks = [i * 0.5 for i in range(1, 15)]

# Plot histograms for each screen time category
for i, category in enumerate(scr_columns):
    high_dep1_2 = data1_2[data1_2['deprived'] == 1][category]
    others_dep1_2 = data1_2[data1_2['deprived'] == 0][category]
    
    axes[i].hist(others_dep1_2, bins=15, alpha=0.5, label='Others', color='orange', edgecolor='black')
    axes[i].hist(high_dep1_2, bins=15, alpha=0.5, label='High Deprivation', color='green', edgecolor='black')
    axes[i].set_title(f'Histogram of {category}')
    axes[i].set_xlabel('Screen Time (daily)')
    axes[i].set_ylabel('Frequency')
    axes[i].legend(loc='upper right')
    
    axes[i].set_xticks(x_ticks)

plt.tight_layout()
plt.show()

# HISTOGRAM 2: Graphic demonstration of 3 Well-being and Deprivation status

fig, axes2 = plt.subplots(3, 1, figsize=(5, 10))
axes2 = axes2.flatten()

# Plot histograms for each screen time category
for i, category in enumerate(wb_columns):
    high_dep1_3 = data1_3[data1_3['deprived'] == 1][category]
    others_dep1_3 = data1_3[data1_3['deprived'] == 0][category]
    
    axes2[i].hist(others_dep1_3, bins=5, alpha=1, label='Others', color='blue', edgecolor='black')
    axes2[i].hist(high_dep1_3, bins=5, alpha=1, label='High Deprivation', color='yellow', edgecolor='black')
    axes2[i].set_title(f'Histogram of {category}')
    axes2[i].set_xlabel('Well-being')
    axes2[i].set_ylabel('Frequency')
    axes2[i].legend(loc='upper right')

plt.tight_layout()
plt.show()