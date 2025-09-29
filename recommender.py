import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from ast import literal_eval

# --- Helper Functions for Data Cleaning ---

def get_director(crew_data):
    """Extracts the director's name from the 'crew' column."""
    for member in crew_data:
        if member.get('job') == 'Director':
            return member.get('name', '')
    return ''

def get_list_of_names(data, limit=3):
    """Extracts a list of names from columns like 'cast' or 'keywords'."""
    if isinstance(data, list):
        names = [item.get('name', '') for item in data]
        return names[:limit]
    return []

def clean_name(name):
    """Removes spaces from a name to create a single token."""
    return str(name).lower().replace(" ", "")

def create_feature_soup(x):
    """
    Creates a weighted 'soup' of features. The title, director, and cast
    are given higher weights to improve franchise awareness and relevance.
    The plot overview is also included for richer textual context.
    """
    # Heavily weight the title for franchise awareness
    title = clean_name(x['title']) * 4
    director = clean_name(x['director']) * 3
    cast = ' '.join([clean_name(i) for i in x['cast']]) * 2
    keywords = ' '.join([clean_name(i) for i in x['keywords']])
    genres = ' '.join([clean_name(i) for i in x['genres']])
    # Include the overview for better plot-based similarity
    overview = x['overview'] if isinstance(x['overview'], str) else ''

    return f"{title} {director} {cast} {keywords} {genres} {overview}"

# --- Main Data Loading and Preparation Function ---

def load_and_prepare_data(metadata_path, credits_path, keywords_path):
    """
    Loads, cleans, merges, and prepares the movie data from multiple CSV files.
    """
    df_meta = pd.read_csv(metadata_path, low_memory=False)
    df_credits = pd.read_csv(credits_path)
    df_keywords = pd.read_csv(keywords_path)

    # Clean and merge data
    df_meta = df_meta[df_meta['vote_count'] >= 10].copy()
    df_meta['id'] = pd.to_numeric(df_meta['id'], errors='coerce')
    df_meta = df_meta.dropna(subset=['id'])
    df_meta['id'] = df_meta['id'].astype(int)

    df = df_meta.merge(df_credits, on='id')
    df = df.merge(df_keywords, on='id')

    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(literal_eval)

    df['director'] = df['crew'].apply(get_director)
    df['cast'] = df['cast'].apply(lambda x: get_list_of_names(x, limit=3))
    df['keywords'] = df['keywords'].apply(get_list_of_names)
    df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x])
    
    # Ensure overview is a string, filling NaNs
    df['overview'] = df['overview'].fillna('')

    df['soup'] = df.apply(create_feature_soup, axis=1)

    df['title'] = df['title'].astype('str')
    df = df.drop_duplicates(subset='title', keep='first')
    df = df.reset_index(drop=True)

    return df

# --- Model Creation and Recommendation Functions ---

def create_ir_model(df):
    """Creates the TF-IDF vectorizer and cosine similarity matrix."""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title'])
    indices = indices[~indices.index.duplicated(keep='first')]
    return cosine_sim, indices, tfidf, tfidf_matrix

def get_close_match(query, titles):
    """Uses fuzzywuzzy to find the best title match for a user's query."""
    match = process.extractOne(query, titles)
    if match and match[1] > 60:
        return match[0]
    return None

def _calculate_weighted_rating(x, m, C):
    """Helper function to calculate the IMDB weighted rating."""
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

def get_recommendations_by_movie(query, df, indices, cosine_sim):
    """
    Finds a close match for a movie title, gets similar movies, and re-ranks them
    by a weighted rating to ensure quality.
    """
    matched_title = get_close_match(query, indices.index)
    if not matched_title:
        return "No close match found for your query. Please try again.", []

    idx = indices[matched_title]
    if isinstance(idx, (pd.Series, np.ndarray)):
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:26]
    movie_indices = [i[0] for i in sim_scores]

    similar_movies = df.iloc[movie_indices].copy()
    m = df['vote_count'].quantile(0.60)
    C = df['vote_average'].mean()
    similar_movies['score'] = similar_movies.apply(lambda x: _calculate_weighted_rating(x, m, C), axis=1)
    similar_movies = similar_movies.sort_values('score', ascending=False)
    
    return matched_title, similar_movies['title'].head(5).tolist()

def get_recommendations_by_query(query, df, tfidf_vec, tfidf_mat):
    """
    Gets recommendations based on a user's text query (keywords, plot, etc.).
    """
    query_vec = tfidf_vec.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_mat).flatten()
    
    top_indices = sim_scores.argsort()[-25:][::-1]
    
    query_results = df.iloc[top_indices].copy()
    m = df['vote_count'].quantile(0.60)
    C = df['vote_average'].mean()
    query_results['score'] = query_results.apply(lambda x: _calculate_weighted_rating(x, m, C), axis=1)
    query_results = query_results.sort_values('score', ascending=False)

    return query_results['title'].head(5).tolist()

