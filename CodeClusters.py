# generated csv file of 500 checkins from OpenDev using: ssh -p 29418 smcdonou02@review.opendev.org "gerrit query status:open --format=json" > opendevcheckin.json
# then converted json to csv file and selected which columns I wanted to keep (owner name & subject)
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict
from collections import Counter
import numpy as np
import os 

output_dir = 'figs'
os.makedirs(output_dir, exist_ok=True)

''' Load in CSV file '''
df = pd.read_csv('C:\ssw-565-final\ssw-565-final\OpenDevCheckIns.csv', encoding='utf-8', encoding_errors='ignore')

df['subject'] = df['subject'].fillna('').str.lower()

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['subject'])

num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

cluster_names = {
    0: 'Performance',
    1: 'Reliability',
    2: 'Testing',
    3: 'Maintainability',
    4: 'Usability',
    5: 'Scalability'
}
df['cluster_name'] = df['cluster'].map(cluster_names)

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

# display
print("Cluster Percentage Distribution:")
print(percentage_distribution)

# plot distribution
ax = percentage_distribution.plot(kind='bar', title='Cluster Percentage Distribution', ylabel='Value', xlabel='Cluster')
ax.bar_label(ax.containers[0], fmt='%.1f%%')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ClusterPercentageDistribution.png"), format='png')
plt.show()


print("Clusters with Low Representation (e.g. <5%): ")
low_rep_clusters = percentage_distribution[percentage_distribution < 5]
print(low_rep_clusters)

for cluster_num in sorted(df['cluster_name'].unique()):
    print(f"Cluster {cluster_num} Samples:")
    print(df[df['cluster_name'] == cluster_num]['subject'].head(5))
    print("------\n")

cluster_map = defaultdict(list)
for idx, cluster in enumerate(df['cluster_name']):
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

for cluster_num in sorted(df['cluster_name'].unique()):
    cluster_text = " ".join(df[df['cluster_name'] == cluster_num]['subject'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Cluster {cluster_num} Word Cloud")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"WordCloud{cluster_num}.png"), format='png')
