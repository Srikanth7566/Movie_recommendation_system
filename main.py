import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv('C:/Python/project_6/movies.csv')

k = movies_data.head()

df = pd.DataFrame(movies_data)

selected_features = df[['genres', 'keywords', 'tagline', 'cast', 'director']]
#print(selected_features)

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres']+'' \
 ' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
#print(combined_features)

vector = TfidfVectorizer()
feature_vector = vector.fit_transform(combined_features)
#print(feature_vector)

similarity = cosine_similarity(feature_vector)
#print(similarity)
#print(similarity.shape)

movie_name = input('enter a movie name:')

list_of_all_titles = df['title'].tolist()
#print(list_of_all_titles)

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
#print(find_close_match)

close_match = find_close_match[0]
#print(close_match)

index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]
#print(index_of_movie)

similarity_score = list(enumerate(similarity[index_of_movie]))
#print(similarity_score)

a1 = len(similarity_score)
#print(a1)

sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
#print(sorted_similar_movies)

print('movies suggested for you: \n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if i < 11:
        print(i, '.', title_from_index)
        i += 1



