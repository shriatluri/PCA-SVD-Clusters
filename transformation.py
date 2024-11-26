import pandas as pd

file_path = 'ratings.csv'
ratings_data = pd.read_csv(file_path)

movies_file_path = 'movies.csv'
movies_data = pd.read_csv(movies_file_path)

#creating the matrix
user_movie_matrix = ratings_data.pivot(index = 'userId', columns = 'movieId', values = 'rating')
user_movie_matrix = user_movie_matrix.fillna(0)

#finding the top movies as well as their names, count the values
top_movies = ratings_data['movieId'].value_counts().head(3).reset_index()
top_movies.columns = ['movieId', 'rating_count']

#merge to get titles
top_movies_titles = top_movies.merge(movies_data, on = 'movieId', how = 'left')

print('The top 3 movies by ratings:')
print(top_movies_titles[['title', 'rating_count']])

#finding the top 3 users using same methodology
top_users = ratings_data['userId'].value_counts().head(3).reset_index()
top_users.columns = ['userId', 'rating_count']

print("The top 3 users by number of ratings:")
print(top_users)