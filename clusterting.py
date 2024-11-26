import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#part a

file_path = 'ratings.csv'
ratings_data = pd.read_csv(file_path)

movies_file_path = 'movies.csv'
movies_data = pd.read_csv(movies_file_path)

# K values to evaluate
k_values = [2, 4, 8, 16, 32, 64, 128]

user_movie_matrix = ratings_data.pivot(index = 'userId', columns = 'movieId', values = 'rating')
user_movie_matrix = user_movie_matrix.fillna(0)

# calculating the inertia
inertia_values = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(user_movie_matrix)
    inertia_values.append(kmeans.inertia_)

#plot the inertia values
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o')
plt.title('Inertia for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid()
plt.show()


#part b - choose 32 as my k

'''
part c - Cluster the data again with your chosen value of k. 
For each of the resulting clusters, find the top three movies/books that are highest rated (on average) by the users in the cluster.
'''

kmeans_32 = KMeans(n_clusters=32, random_state=42, n_init=10)
user_movie_matrix['cluster'] = kmeans_32.fit_predict(user_movie_matrix)

clusters_top_movies = []
for cluster in range(32):
    cluster_data = user_movie_matrix[user_movie_matrix['cluster'] == cluster]
    mean_ratings = cluster_data.drop(columns = 'cluster').mean(axis = 0)
    # get the top 3
    top_movies = mean_ratings.nlargest(3).index
    top_movies_titles = movies_data[movies_data['movieId'].isin(top_movies)][['title', 'movieId']]
    clusters_top_movies.append((cluster, top_movies_titles))

for cluster, top_movies in clusters_top_movies:
    print(f"Cluster {cluster}:")
    print(top_movies)
    print()