Advanced IR-Based Movie Recommender System
This project is a sophisticated, content-based movie recommender system built using core Information Retrieval (IR) techniques. It provides high-quality movie recommendations based on user queries, handling typos and prioritizing popular, well-regarded films. The entire application is served through an interactive and user-friendly web interface powered by Streamlit.

Key Features
Hybrid Recommendation Engine: Combines content similarity (TF-IDF & Cosine Similarity) with a quality filter (IMDB Weighted Rating) to ensure recommendations are both relevant and highly rated.

Multiple Recommendation Modes:

Search by Movie Title: Find movies similar to one you already love.

Search by Description: Get recommendations based on a mood, theme, or plot idea (e.g., "sad space movie with aliens").

Robust Typo Handling: Utilizes fuzzy string matching to find the correct movie even if the user makes a spelling mistake.

Franchise Aware: Intelligently recommends sequels and related movies by weighting titles and key personnel (director, cast) in its similarity algorithm.

Interactive Web UI: A clean, multi-tab interface built with Streamlit for an intuitive user experience.

How It Works: The IR Engine Explained
The system is not just a simple search; it's a multi-stage Information Retrieval pipeline designed to understand user intent and rank results by relevance and quality.

1. Data Preprocessing & Feature Engineering
The foundation of the system is a rich "feature soup" created for each movie. This is a single text string that acts as a movie's content signature. It's engineered by:

Loading & Merging: Data from three separate files (movies_metadata.csv, credits.csv, keywords.csv) is cleaned and merged.

Feature Extraction: Key information like the director and top 3 cast members are extracted from complex data columns.

Weighted "Soup" Creation: The most important features are combined into a single string, with weights applied to prioritize certain terms:

Title (x4 weight): For franchise awareness.

Director (x3 weight): A strong indicator of a movie's style.

Cast (x2 weight): Key actors often define a film's genre and appeal.

Keywords, Genres, Overview: Provide thematic and plot context.

2. TF-IDF Vectorization
The text-based "feature soup" is converted into a numerical matrix using the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm. TF-IDF assesses the importance of a word within a movie's description relative to the entire movie collection. Words that are unique to a small set of movies (e.g., "dystopian" or "cyberpunk") are given a higher score than common words (e.g., "love" or "fight"), making them more significant for matching.

3. Cosine Similarity
Once all movies are represented as numerical vectors, Cosine Similarity calculates the similarity between them. It measures the cosine of the angle between two vectors—a score close to 1 indicates very high similarity, while a score near 0 indicates low similarity. This is the core algorithm used to find a list of movies with similar content.

4. Fuzzy String Matching for User Queries
To handle user input errors (e.g., "Batmn," "the dark nite"), the system uses the fuzzywuzzy library. Instead of requiring an exact match, it calculates the Levenshtein distance between the user's query and all movie titles in the database, automatically finding the closest match.

5. Re-ranking with IMDB Weighted Rating
A list of similar movies is not enough; a good recommendation must also be a good movie. The system uses a hybrid approach by re-ranking the initial list of similar movies based on the IMDB Weighted Rating formula. This formula balances a movie's average rating with the number of votes it has received, ensuring that popular, critically-acclaimed films are ranked higher than obscure or poorly-reviewed ones.

Dataset
This project uses The Movies Dataset, a large, metadata-rich collection of over 45,000 movies.

Source: Kaggle

Download Link: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

You will need to create a free Kaggle account to download the dataset. After downloading, extract the zip archive and place movies_metadata.csv, credits.csv, and keywords.csv in the root of the project folder.

Setup and Installation
Follow these steps to set up and run the project locally.

Prerequisites
Python 3.8 or higher

A virtual environment tool (like venv)

1. Create a Project Folder
Create a new folder for the project and place the provided app.py and recommender.py files inside it.

2. Set Up a Virtual Environment
Open your terminal in the project folder and create a virtual environment. This keeps your project's dependencies isolated.

python -m venv venv

3. Activate the Virtual Environment
On Windows:

venv\Scripts\activate

On macOS / Linux:

source venv/bin/activate

Your terminal prompt should now start with (venv).

4. Install Dependencies
Create a requirements.txt file in your project folder with the following content:

streamlit
pandas
scikit-learn
numpy
python-Levenshtein
fuzzywuzzy

Now, install all the required libraries by running:

pip install -r requirements.txt

5. Download and Place the Dataset
Download the dataset from the Kaggle link provided above. Unzip the file and move movies_metadata.csv, credits.csv, and keywords.csv into your main project folder.

How to Run the Application
With the setup complete, you can now launch the web app.

Make sure your virtual environment is still active.

Run the following command in your terminal:

streamlit run app.py

Streamlit will start a local server and provide you with a URL (usually http://localhost:8501). Open this URL in your web browser to use the recommender system!
