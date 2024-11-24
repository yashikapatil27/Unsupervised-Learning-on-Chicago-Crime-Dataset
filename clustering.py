from kmodes.kmodes import KModes
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import SpectralClustering
from scipy.cluster import hierarchy
from sklearn.metrics.pairwise import cosine_similarity

# K-Modes Clustering
def apply_kmodes(df, n_clusters, init="random", n_init=5):
    kmodes = KModes(n_clusters=n_clusters, init=init, n_init=n_init, verbose=1)
    clusters = kmodes.fit_predict(df)
    return clusters

# Hierarchical Clustering
def hierarchical_clustering(df, threshold=0.1):
    X = CountVectorizer().fit_transform(df)
    X = TfidfTransformer().fit_transform(X).todense()
    Z = hierarchy.linkage(X, "average", metric="cosine")
    clusters = hierarchy.fcluster(Z, threshold, criterion="distance")
    return clusters

# Spectral Clustering
def spectral_clustering(df, n_clusters):
    cosine_sim = cosine_similarity(df)
    spectral_model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=10)
    clusters = spectral_model.fit_predict(cosine_sim)
    return clusters
