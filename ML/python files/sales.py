# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('sales_data_sample.csv', encoding='ISO-8859-1')

selected_data = data.select_dtypes(include=[np.number])  # Select numerical columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)

# Step 2: Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph to find the optimal number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Square (WCSS)')
plt.show()

# Step 3: Fit K-Means with the optimal number of clusters
optimal_clusters = 3  # Replace with the elbow-determined cluster count
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
kmeans.fit(scaled_data)

# Add the cluster labels to the original data
data['Cluster'] = kmeans.labels_

# Step 4: Visualize the Clusters (for 2D or 3D data)

# Example 1: Plot if you have 2 numerical features
plt.figure(figsize=(10, 6))
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=data['Cluster'], cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


# Fit K-Means with the optimal number of clusters
optimal_clusters = 3  # Replace with the number from the elbow method
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
kmeans.fit(scaled_data)

# Add cluster labels to the original data for further analysis
data['Cluster'] = kmeans.labels_

# Optional: Display a few rows of the clustered data
print(data.head())

