import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

performance_with_moe = [73.44, 73.80]
performance_without_moe = [71.28, 73.12]

# Create a DataFrame for the data
data = {
    'Model Size': ['Qwen2-1B-Instruct', 'Qwen2-7B-Instruct'] * 2,
    'Performance': performance_with_moe + performance_without_moe,
    'Model Setting': ['w/ MoE'] * 2 + ['w/o MoE'] * 2
}
df = pd.DataFrame(data)

# Create the plot
plt.figure(figsize=(10, 7))
ax = sns.barplot(x='Model Size', y='Performance', hue='Model Setting', data=df, palette='viridis')

# Set plot titles and labels with bold font
plt.title('Mustard F1 of Different Size of Qwen2', fontsize=22, fontweight='bold')
plt.xlabel('Model Size', fontsize=22, fontweight='bold')
plt.ylabel('Mustard F1', fontsize=22, fontweight='bold')
plt.xticks(fontsize=22, fontweight='bold')
plt.yticks(fontsize=22, fontweight='bold')
plt.ylim(70, 75)
legend = plt.legend(fontsize=18,loc='upper left', title='Model Setting', title_fontsize=18)
plt.setp(legend.get_title(), fontsize='20', fontweight='bold')
plt.setp(legend.get_texts(), fontsize='20', fontweight='bold')

# Add numbers on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=22, fontweight='bold', color='black')

# Save the plot as a PDF file
plt.savefig('performance_with_and_without_moe.pdf')

# Display the plot
plt.show()
