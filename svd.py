from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ratings_file_path = 'ratings.csv'
movies_file_path = 'movies.csv'

ratings_data = pd.read_csv(ratings_file_path)
movies_data = pd.read_csv(movies_file_path)

user_movie_matrix = ratings_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

#part a - svd with k = 128
svd_128 = TruncatedSVD(n_components=128, random_state=42)
svd_128.fit(user_movie_matrix)

'''
#plot the singular values
singular_values = svd_128.singular_values_
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(singular_values) + 1), singular_values, marker='o')
plt.title('Singular Values for SVD with k = 128')
plt.xlabel('Component Number')
plt.ylabel('Singular Value')
plt.grid()
plt.show()
'''

#part b - sum of the explained variance ratio for k
k_values = [2, 4, 8, 16, 32, 64, 128]
explained_variance_sums = []

for k in k_values:
    svd = TruncatedSVD(n_components=k, random_state=42)
    svd.fit(user_movie_matrix)
    explained_variance_sums.append(svd.explained_variance_ratio_.sum())

#results
print("\nSum of Explained Variance Ratio for Different k:")
for k, var_sum in zip(k_values, explained_variance_sums):
    print(f"k = {k}: {var_sum}")

#part c - apply svd with k = 2 and transform the data
svd_2 = TruncatedSVD(n_components=2, random_state=42)
user_movie_matrix_svd2 = svd_2.fit_transform(user_movie_matrix)

#part d - plot SVD results for k = 2 with users colored by cluster membership
cluster_memberships = pd.Series(user_movie_matrix.index % 10, index=user_movie_matrix.index, name='cluster')

svd2_results = pd.DataFrame(user_movie_matrix_svd2, columns=['Component 1', 'Component 2'])
svd2_results['cluster'] = cluster_memberships.values

#plot the figure for k = 2
plt.figure(figsize=(10, 8))
for cluster in svd2_results['cluster'].unique():
    cluster_data = svd2_results[svd2_results['cluster'] == cluster]
    plt.scatter(cluster_data['Component 1'], cluster_data['Component 2'], label=f'Cluster {cluster}', alpha=0.6)

plt.title('SVD with k = 2: Users Colored by Cluster Membership')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()