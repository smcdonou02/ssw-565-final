import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict
from collections import Counter
import numpy as np

''' Load in CSV file '''
df = pd.read_csv('Code_Check_Ins.csv', encoding='utf-8', encoding_errors='ignore')

df['subject'] = df['subject'].fillna('').str.lower()
df['description'] = df['description'].fillna('').str.lower()
df['combined_subject_description'] = df['subject'] + " " + df['description']

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['combined_subject_description'])

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

df['cluster'].value_counts().plot(kind='bar')
plt.title('Check-In Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel("Number of Check-Ins")
plt.show()

# cluster percentage distribution
cluster_counts = df['cluster'].value_counts()
total_check_ins = len(df)

percentage_distribution = (cluster_counts / total_check_ins) * 100

# display
print("Cluster Percentage Distribution:")
print(percentage_distribution)

# plot distribution
percentage_distribution.plot(kind='bar')
plt.title('Cluster Percentage Distribution')
plt.xlabel('Cluster')
plt.ylabel('Percentage of Check-Ins')
plt.show()

print("Clusters with Low Representation (e.g. <5%): ")
low_rep_clusters = percentage_distribution[percentage_distribution < 5]
print(low_rep_clusters)

for cluster_num in sorted(df['cluster'].unique()):
    print(f"Cluster {cluster_num} Samples:")
    print(df[df['cluster'] == cluster_num]['combined_subject_description'].head(5))
    print("------\n")

cluster_map = defaultdict(list)
for idx, cluster in enumerate(df['cluster']):
    cluster_map[cluster].append(idx)

# Step 2: Precompute top terms for each cluster
top_terms_per_cluster = {}
for cluster_num, indices in cluster_map.items():
    # Efficiently slice and sum sparse matrix rows
    cluster_matrix = X[indices, :].sum(axis=0).A1
    
    # Get top terms
    top_indices = np.argsort(cluster_matrix)[-10:][::-1]
    top_terms = [vectorizer.get_feature_names_out()[i] for i in top_indices if cluster_matrix[i] > 0]
    
    top_terms_per_cluster[cluster_num] = top_terms
    print(f"Cluster {cluster_num} Top Terms: {top_terms}")


all_top_terms = [term for terms in top_terms_per_cluster.values() for term in terms]
term_frequency = Counter(all_top_terms)

print("Overlapping Terms Across Clusters:")
for term, count in term_frequency.items():
    if count > 1:
        print(f"Term '{term}' appears in {count} clusters")

sil_score = silhouette_score(X, df['cluster'])

print(f"Silhouette Score: {sil_score:.2f}")
