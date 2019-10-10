from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.externals import joblib


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('movie_dataset.csv')
    df_ = df[['movie_title', 'genres', 'plot_keywords']]
    df_.dropna(inplace=True)
    df_['index'] = range(0, len(df_))
    df_.index = range(0, len(df_))

    # Functions to get movie title from movie index and vice-versa

    def get_title_from_index(index):
        find_title = df_.loc[df_['index'] == index, 'movie_title']
        return find_title[index].replace(u'\xa0', '')

    def get_genres_from_index(index):
        find_genres = df_.loc[df_['index'] == index, 'genres']
        return find_genres[index].replace(u'\xa0', '')

    def get_index_from_title(title):
        movie = title + '\xa0'
        find_index = df_.loc[df_['movie_title'] == movie, 'index']
        return find_index.index[0]
    
    # Using the saved model containing similarity scores
    Movie_Recom_model = open('Movie_Cosine_Scores.pkl', 'rb')
    cosine_sim = joblib.load(Movie_Recom_model)

    if request.method == 'POST':
        message = request.form['message']
        data = message
        movie_user_likes = data
        movie_index = get_index_from_title(movie_user_likes)
        
        # Find out the movies similar to the movie name specified
        similar_movies = list(enumerate(cosine_sim[movie_index]))
        
        # Sort the similar movies in descending order
        sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
        
        # Output top 5 similar movies
        i = 0
        my_prediction = []
        my_prediction.append('Top 5 similar movies to ' + movie_user_likes +
                             '[Genre:' + str(df_['genres'][movie_index]) + ']' + ' are:')
        for element in sorted_similar_movies:
            my_prediction.append(get_title_from_index(
                element[0]) + '........ [Genres: ' + get_genres_from_index(element[0]) + ']')
            i = i+1
            if i > 5:
                break

        
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
