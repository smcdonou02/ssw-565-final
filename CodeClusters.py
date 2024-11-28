"""
Author: Stephanie McDonough
Date: 2024-11-27
File: CodeClusters.py
Description: Import code check-ins from OpenDev, clustercheck-ins based on architectural attributes and statistically analyze clusters.
"""

import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict
from collections import Counter
import numpy as np
import os 

# Set up folder to store figures generated.
output_dir = 'figs'
os.makedirs(output_dir, exist_ok=True)

# Build dataframe from CSV file holding 500 code check-in files from OpenDev.
df = pd.read_csv('C:\ssw-565-final\ssw-565-final\OpenDevCheckIns.csv', encoding='utf-8', encoding_errors='ignore')

# Fill in blank data to avoid errors.
df['subject'] = df['subject'].fillna('').str.lower()

# Transform the 'subject' column into a TF-IDF matrix, removing common English stop words (the, and, is, etc).
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['subject'])

# Number of Clusters determined through expirimentation and running program to determine best results.
num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

df['cluster'] = kmeans.fit_predict(X)

# Assigns architectural attributes to clusters based on top terms outputted later in code.
cluster_names = {
    0: 'Performance',
    1: 'Reliability',
    2: 'Testing',
    3: 'Maintainability',
    4: 'Usability',
    5: 'Scalability'
}
# Map the cluster names to each code check-in in the data frame.
df['cluster_name'] = df['cluster'].map(cluster_names)

# Visualize clusters on bar plot, save plot to figs folder.
ax1 = df['cluster_name'].value_counts().plot(kind='bar')
plt.title('Check-In Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel("Number of Check-Ins")
ax1.bar_label(ax1.containers[0], fmt='%.0f')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ClusterValueDistribution.png"), format='png')
plt.close()

# cluster percentage distribution
cluster_counts = df['cluster_name'].value_counts()
total_check_ins = len(df)

percentage_distribution = (cluster_counts / total_check_ins) * 100

# Display percentage distribution in terminal.
print("Cluster Percentage Distribution:")
print(percentage_distribution)

# Plot cluster percentage distribution on bar plot and save in figs.
ax = percentage_distribution.plot(kind='bar', title='Cluster Percentage Distribution', ylabel='Value', xlabel='Cluster')
ax.bar_label(ax.containers[0], fmt='%.1f%%')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ClusterPercentageDistribution.png"), format='png')
plt.close()

# Identify clusters with low representation to analyze challenges in detecting clusters.
print("Clusters with Low Representation (e.g. <5%): ")
low_rep_clusters = percentage_distribution[percentage_distribution < 5]
print(low_rep_clusters)

# Map clusters to indices.
for cluster_num in sorted(df['cluster_name'].unique()):
    print(f"Cluster {cluster_num} Samples:")
    print(df[df['cluster_name'] == cluster_num]['subject'].head(5))
    print("------\n")

cluster_map = defaultdict(list)
for idx, cluster in enumerate(df['cluster_name']):
    cluster_map[cluster].append(idx)

# Calculate top terms for each cluster
top_terms_per_cluster = {}
for cluster_num, indices in cluster_map.items():

    cluster_matrix = X[indices, :].sum(axis=0).A1
    
    # Get top terms
    top_indices = np.argsort(cluster_matrix)[-10:][::-1]
    top_terms = [vectorizer.get_feature_names_out()[i] for i in top_indices if cluster_matrix[i] > 0]
    
    top_terms_per_cluster[cluster_num] = top_terms
    print(f"Cluster {cluster_num} Top Terms: {top_terms}")

all_top_terms = [term for terms in top_terms_per_cluster.values() for term in terms]
term_frequency = Counter(all_top_terms)

# Print terms that overlap in different clusters to analyze challenges in detecting different clusters.
print("Overlapping Terms Across Clusters:")
for term, count in term_frequency.items():
    if count > 1:
        print(f"Term '{term}' appears in {count} clusters")
        
# Generate Word Clouds to visualize top terms in each cluster
for cluster_num in sorted(df['cluster_name'].unique()):
    cluster_text = " ".join(df[df['cluster_name'] == cluster_num]['subject'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Cluster {cluster_num} Word Cloud")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"WordCloud{cluster_num}.png"), format='png')
