import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

performance_with_moe = [73.44, 73.80]
performance_without_moe = [71.28, 73.12]

# Create a DataFrame for the data
data = {
    'Model Size': ['1B', '7B'] * 2,
    'Performance': performance_with_moe + performance_without_moe,
    'MoE': ['w/ MoE'] * 2 + ['w/o MoE'] * 2
}
df = pd.DataFrame(data)

# Create the plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Model Size', y='Performance', hue='MoE', data=df, palette='viridis')
plt.title('Mustard F1 of Qwen2 with and without MoE', fontsize=22)
plt.xlabel('Model Size', fontsize=22)
plt.ylabel('Mustard F1', fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.ylim(70, 75)
plt.legend(fontsize=16, loc='upper left')

# Display the plot
plt.savefig('performance_with_and_without_moe.pdf')
