import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, homogeneity_completeness_v_measure, adjusted_rand_score
import matplotlib.pyplot as plt

# Load the dataset from CSV
data = pd.read_csv('data.csv')
X = data[['x', 'y']].values
true_labels = data['color'].values

# Apply the EM algorithm for clustering
em = GaussianMixture(n_components=3)
em.fit(X)
y_em = em.predict(X)

# Apply the K-means algorithm for clustering
kmeans = KMeans(n_clusters=3, n_init=10)  # Set n_init explicitly
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Calculate evaluation metrics for EM algorithm
silhouette_em = silhouette_score(X, y_em)
homogeneity_em, completeness_em, _ = homogeneity_completeness_v_measure(true_labels, y_em)
ari_em = adjusted_rand_score(true_labels, y_em)

# Calculate evaluation metrics for K-means algorithm
silhouette_kmeans = silhouette_score(X, y_kmeans)
homogeneity_kmeans, completeness_kmeans, _ = homogeneity_completeness_v_measure(true_labels, y_kmeans)
ari_kmeans = adjusted_rand_score(true_labels, y_kmeans)

# Print the evaluation metrics
print("Evaluation Metrics - EM Algorithm:")
print("Silhouette Score: {:.2f}".format(silhouette_em))
print("Homogeneity Score: {:.2f}".format(homogeneity_em))
print("Completeness Score: {:.2f}".format(completeness_em))
print("Adjusted Rand Index: {:.2f}".format(ari_em))
print()
print("Evaluation Metrics - K-means Algorithm:")
print("Silhouette Score: {:.2f}".format(silhouette_kmeans))
print("Homogeneity Score: {:.2f}".format(homogeneity_kmeans))
print("Completeness Score: {:.2f}".format(completeness_kmeans))
print("Adjusted Rand Index: {:.2f}".format(ari_kmeans))

# Plot the clusters for both algorithms
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_em)
plt.scatter(em.means_[:, 0], em.means_[:, 1], c='red', marker='X', s=100, label='Cluster Centers')
plt.title("EM Algorithm")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=100, label='Cluster Centers')
plt.title("K-means Algorithm")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()

#sammedcjain
#4ni20is095
#ISE 6th sem 