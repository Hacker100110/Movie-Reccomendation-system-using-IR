Advanced IR-Based Movie Recommender System
Project Overview
This project is a sophisticated, content-based movie recommender system built using core Information Retrieval (IR) techniques. In an age of overwhelming content choice, this system aims to solve the "paradox of choice" by providing users with relevant, high-quality, and personalized movie recommendations. It intelligently processes user queries, handles common errors like typos, and prioritizes popular, well-regarded films. The entire application is served through an interactive and user-friendly web interface powered by Streamlit.
Key Features
Hybrid Recommendation Engine: The system's core is a hybrid engine that goes beyond simple content matching. It combines two powerful stages: first, it finds a list of movies with the most similar content (using TF-IDF & Cosine Similarity), and second, it re-ranks that list using a quality filter (the IMDB Weighted Rating formula). This ensures that recommendations are not just relevant but also of high quality, preventing suggestions for poorly-rated but similar films.
Multiple Recommendation Modes: To cater to different user needs, the system offers two distinct modes for movie discovery:
Search by Movie Title: For when a user knows a movie they like and wants to find others in the same vein.
Search by Description: An exploratory mode for when a user has a mood, theme, or plot idea in mind (e.g., "sad space movie with aliens," "funny heist movie with a twist") and wants the system to find matching films.
Robust Typo Handling: A key usability feature is its resilience to user error. The system utilizes fuzzy string matching based on the Levenshtein distance. This allows it to correctly identify a movie like "Inglourious Basterds" even if the user types "Inglorious Basterds" or "Inglourious Bastards," making for a forgiving and smooth user experience.
Franchise Aware: Many simple recommenders fail to connect sequels or films within the same universe. This system solves that problem by heavily weighting a movie's title and key personnel (director, cast) in its similarity algorithm. This ensures that a search for "The Dark Knight" will logically and correctly suggest "The Dark Knight Rises" as a top recommendation.
Interactive Web UI: The application is presented through a clean, modern, multi-tab interface built with Streamlit. This makes the powerful backend logic accessible and easy to use for anyone, without needing to interact with code.
How It Works: The IR Engine Explained
The system is not just a simple search tool; it's a multi-stage Information Retrieval pipeline designed to deeply understand a movie's essence, interpret user intent, and rank the final results by both relevance and quality.
1. Data Preprocessing & Feature Engineering
The foundation of the system is a rich "feature soup" created for each movie, which acts as its unique content signature. This process involves several critical steps to handle large, real-world data:
Loading & Merging: Raw data is loaded from three separate files (movies_metadata.csv, credits.csv, keywords.csv), cleaned of invalid entries, and merged into a single, unified DataFrame.
Feature Extraction: Key information, like the director's name and the top 3 cast members, is carefully extracted from complex, stringified JSON columns using safe parsing techniques.
Weighted "Soup" Creation: The most important textual features are combined into a single string for each movie. Crucially, weights are applied to prioritize the terms that are most indicative of a movie's identity:
Title (x4 weight): Given the highest priority for strong franchise awareness.
Director (x3 weight): A director's name is a powerful signal of a film's style and tone.
Cast (x2 weight): The main actors heavily influence a film's genre and appeal.
Keywords, Genres, & Overview: These provide the broad thematic and detailed plot context.
2. TF-IDF Vectorization
The text-based "feature soup" is converted into a numerical matrix using the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm. This is a cornerstone of Information Retrieval. It evaluates the importance of a word by balancing its frequency in a single movie's description against its frequency across all movies. This allows the model to learn that unique, descriptive words (e.g., "dystopian," "cyberpunk," "heist") are much more important for similarity matching than common words (e.g., "love," "life," "world") that appear everywhere.
3. Cosine Similarity
Once all movies are represented as numerical TF-IDF vectors, Cosine Similarity calculates the similarity between every pair. Imagine each movie as a direction in a high-dimensional space. Cosine Similarity measures the angle between these directions. Movies with similar content will "point" in a similar direction, resulting in a small angle and a similarity score close to 1. Dissimilar movies will point in different directions, resulting in a score near 0. This is the core algorithm used to retrieve a list of movies with similar content.
4. Fuzzy String Matching for User Queries
To create a seamless user experience, the system must handle input errors. It uses the fuzzywuzzy library, which implements the Levenshtein distance algorithm. This algorithm calculates the minimum number of single-character edits (insertions, deletions, or substitutions) needed to change one string into another. This allows the system to find the closest match in the database to a user's query, even with typos.
5. Re-ranking with IMDB Weighted Rating
A list of similar movies is not enough; a good recommendation must also be a good movie. The final and most critical step is to re-rank the list of similar movies using the IMDB Weighted Rating formula. This prevents a scenario where a movie that is thematically similar but universally panned gets recommended. The formula elegantly balances a movie's average rating with the number of votes it has received, ensuring that popular, critically-acclaimed films are ranked higher.
Dataset
This project uses The Movies Dataset, a large, metadata-rich collection of over 45,000 movies, making it an excellent resource for building a robust recommender system.
Source: Kaggle
Download Link: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
You will need a free Kaggle account to download the dataset. After downloading, extract the zip archive and place movies_metadata.csv, credits.csv, and keywords.csv in the root of the project folder.
Project Structure
recommender.py: The core of the project. This Python script contains all the backend logic for data processing, feature engineering, and the IR-based recommendation algorithms.
app.py: The frontend of the project. This script uses the Streamlit library to create the interactive web application, handling user input and displaying the recommendations generated by recommender.py.
README.md: This file, providing a comprehensive overview of the project, its methodology, and setup instructions.
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
