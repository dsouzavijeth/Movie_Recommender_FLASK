# Movie_Recommender_FLASK

This is a Movie Recommendation system which predicts the top 5 movies which are similar to a particular movie specified.
The model is then deployed using FLASK. 

The dataset contains 5043 samples and 28 columns. For our analysis, only columns  'movie_title', 'genres', 'plot_keywords' are considered.

A Content based recommendation systems is used which takes in a movie title as input. It then analyzes the contents (movie title, genre, plot keywords) of the movie to find out other movies which have similar content. Then it ranks similar movies according to their similarity scores and recommends the most relevant top 5 movies.

CountVectorizer is used to represent the texts as vectors (Bag of Words).
We then find the cosine similarity between these vectors to find out how similar they are from each other.

***NOTE***
1. Run the code in the Jupyter Notebook 'Movie_Recommender.ipynb'
2. This saves a model with the name 'Movie_Cosine_Scores.pkl', which will be used while you run the deployment code.
3. Steps for model deployment is explained in 'Steps_for_Model_Deployment.pdf / .docx'
