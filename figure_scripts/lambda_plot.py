import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data/lambda/results.csv")

# Normalize the data
# df['covered_demand'] = (df['covered_demand'] - df['covered_demand'].min()) / (df['covered_demand'].max() - df['covered_demand'].min())
# df['travel_cost'] = (df['travel_cost'] - df['travel_cost'].min()) / (df['travel_cost'].max() - df['travel_cost'].min())

# Group by lambda and calculate the mean of covered_demand and travel_cost
avg_covered_demand = df.groupby('lambda')['covered_demand'].mean()
avg_travel_cost = df.groupby('lambda')['travel_cost'].mean()
std_covered_demand = df.groupby('lambda')['covered_demand'].std()
std_travel_cost = df.groupby('lambda')['travel_cost'].std()

efficiency = - avg_covered_demand / avg_travel_cost
efficiency = efficiency.fillna(0)  # Handle any NaN values



# Plotting
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plt.subplots_adjust(wspace=0, hspace=0, left=0.18, top=0.95, right=0.95, bottom=0.12)

# plt.plot(avg_covered_demand.index, avg_covered_demand, label='Covered Demand', color='blue')
# plt.plot(avg_travel_cost.index, avg_travel_cost, label='Travel Cost', color='red')

# plt.fill_between(avg_covered_demand.index, avg_covered_demand - std_covered_demand, avg_covered_demand + std_covered_demand, color='blue', alpha=0.1)
# plt.fill_between(avg_travel_cost.index, avg_travel_cost - std_travel_cost, avg_travel_cost + std_travel_cost, color='red', alpha=0.1)


plt.plot(avg_covered_demand.index, efficiency, label='Efficiency', color='green')
plt.fill_between(avg_covered_demand.index, efficiency - std_covered_demand / avg_travel_cost, efficiency + std_covered_demand / avg_travel_cost, color='green', alpha=0.1)

# objective_function = avg_covered_demand + avg_travel_cost
# best_lambda = objective_function.idxmax()
# plt.axvline(x=best_lambda, color='gray', linestyle='-', label=f'Best Lambda: {best_lambda}')
# plt.scatter(best_lambda, objective_function[best_lambda], color='black', label='Best Lambda Point', marker='x', s=50, zorder=5)

plt.xlabel('Lambda')
plt.ylabel('Value')
plt.legend(loc='best', frameon=False)

# plt.ylim(0, 1)

plt.savefig('plots/lambda_plot.png')
