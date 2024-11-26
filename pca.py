import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ratings_file_path = 'ratings.csv'
movies_file_path = 'movies.csv'

ratings_data = pd.read_csv(ratings_file_path)
movies_data = pd.read_csv(movies_file_path)

#part a - create the matrix and transpose it
user_movie_matrix = ratings_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
movie_user_matrix = user_movie_matrix.T

#mean center the data
movie_user_matrix_centered = movie_user_matrix - movie_user_matrix.mean(axis=0)

#part b - apply pca with the components
pca = PCA(n_components=2)
movie_user_matrix_pca = pca.fit_transform(movie_user_matrix_centered)

#part c - plot the results
#df for pca results
pca_results = pd.DataFrame(movie_user_matrix_pca, columns = ['PC1', 'PC2'])
pca_results['movieId'] = movie_user_matrix.index
pca_results = pca_results.merge(movies_data, on='movieId', how='left')

pca_results['genre'] = pca_results['genres'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else 'Unknown')

#plot the pca results
plt.figure(figsize=(10, 8))
for genre in pca_results['genre'].unique():
    genre_data = pca_results[pca_results['genre'] == genre]
    plt.scatter(genre_data['PC1'], genre_data['PC2'], label=genre, alpha=0.5)

plt.title('PCA of Movies Colored by Genre')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()

#part d - 80 vs 40 and explanation
explained_variance_ratios = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratios.cumsum()

components_80 = (cumulative_variance >= 0.80).argmax() + 1
components_40 = (cumulative_variance >= 0.40).argmax() + 1
#the results
print(f"Number of components for 80% variance: {components_80}")
print(f"Number of components for 40% variance: {components_40}")
print("Explained Variance Ratios:", explained_variance_ratios)