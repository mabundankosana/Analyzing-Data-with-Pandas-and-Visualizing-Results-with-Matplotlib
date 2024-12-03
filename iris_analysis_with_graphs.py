
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display initial data
print("First few rows of the dataset:")
print(df.head())

# Check data structure
print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Describe numerical data
print("\nSummary statistics:")
print(df.describe())

# Grouped data
grouped_data = df.groupby('species').mean()

# Visualizations
sns.set(style="whitegrid")

# Line chart: Trend over indices for sepal length
plt.figure(figsize=(8, 6))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length', color='blue')
plt.title("Trend of Sepal Length")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.savefig("line_chart_sepal_length.png")
print("Line chart saved as 'line_chart_sepal_length.png'")

# Bar chart: Average petal length per species
plt.figure(figsize=(8, 6))
sns.barplot(x=grouped_data.index, y=grouped_data['petal length (cm)'], palette='viridis')
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.savefig("bar_chart_petal_length.png")
print("Bar chart saved as 'bar_chart_petal_length.png'")

# Histogram: Distribution of sepal width
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal width (cm)'], bins=10, kde=True, color='green')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.savefig("histogram_sepal_width.png")
print("Histogram saved as 'histogram_sepal_width.png'")

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='deep')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.savefig("scatter_plot_sepal_vs_petal_length.png")
print("Scatter plot saved as 'scatter_plot_sepal_vs_petal_length.png'")

print("\nAnalysis and visualizations completed. Charts are saved as PNG files in the working directory.")
