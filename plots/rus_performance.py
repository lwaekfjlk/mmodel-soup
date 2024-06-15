import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Example data for ALBEF and BLIP-2 with R/U/S performance
albef_rus = [0.85, 0.75, 0.65]
blip2_rus = [0.88, 0.78, 0.68]

# Create a DataFrame for the data
data = {
    'Model': ['ALBEF'] * 3 + ['BLIP-2'] * 3,
    'R/U/S': ['R', 'U', 'S'] * 2,
    'Performance': albef_rus + blip2_rus
}
df = pd.DataFrame(data)

# Create the plot
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Model', y='Performance', hue='R/U/S', data=df, palette='Oranges')

# Set plot titles and labels with bold font
plt.title('Comparison of R/U/S Performance of ALBEF and BLIP-2', fontsize=22, fontweight='bold')
plt.xlabel('Model', fontsize=22, fontweight='bold')
plt.ylabel('Performance', fontsize=22, fontweight='bold')
plt.xticks(fontsize=22, fontweight='bold')
plt.yticks(fontsize=22, fontweight='bold')
plt.ylim(0.6, 0.9)
legend = plt.legend(fontsize=18, loc='upper left', title='R/U/S', title_fontsize=18)
plt.setp(legend.get_title(), fontsize='20', fontweight='bold')
plt.setp(legend.get_texts(), fontsize='20', fontweight='bold')

# Add numbers on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=22, fontweight='bold', color='black')

# Save the plot as a PDF file
plt.savefig('albef_blip2_rus_performance.pdf')
