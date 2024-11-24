from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt

# Visualization for Clustering Results
def calculate_nmi(data, clusters, features):
    nmi_scores = {}
    for feature in features:
        nmi_scores[feature] = normalized_mutual_info_score(data[feature], clusters)
    return nmi_scores

# Plotting K-Modes Clusters
def plot_kmodes_clusters(data, clusters, features):
    for cluster in set(clusters):
        cls = data[clusters == cluster]
        for feature in features:
            cls[feature].value_counts().plot(kind='pie', title=f"Cluster {cluster} - {feature}")
            plt.show()

# Plotting Spectral Clustering Clusters
def plot_spectral_clusters(data, clusters, features):
    for cluster in set(clusters):
        cls = data[clusters == cluster]
        for feature in features:
            cls[feature].value_counts().plot(kind='pie', title=f"Cluster {cluster} - {feature}")
            plt.show()