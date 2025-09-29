# app.py (v7.0 - Simplified, No API Key Required)

import streamlit as st
from recommender import (
    load_and_prepare_data,
    create_ir_model,
    get_recommendations_by_movie,
    get_recommendations_by_query
)

# --- Page Configuration ---
st.set_page_config(
    page_title="CineRecs IR Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Card for text-based recommendations */
.movie-card-text {
    background-color: #1a1a1a;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
    border: 1px solid #444;
    height: 120px; /* Fixed height for consistency */
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.movie-card-text:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    border: 1px solid #1c83e1;
}

.movie-title-text {
    font-size: 1.1em;
    font-weight: bold;
    color: #ffffff;
    text-align: center;
}

.rank-number-text {
    font-size: 1.5em;
    font-weight: bold;
    color: #1c83e1;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# --- Caching Data Loading and Model Creation ---
@st.cache_resource
def load_model_and_data():
    """Loads data and creates the IR model, cached for performance."""
    with st.spinner("🚀 Launching the Information Retrieval engine... This might take a moment on first run."):
        movies_df = load_and_prepare_data('movies_metadata.csv', 'credits.csv', 'keywords.csv')
        cosine_sim, indices, tfidf_vec, tfidf_mat = create_ir_model(movies_df)
    return movies_df, cosine_sim, indices, tfidf_vec, tfidf_mat

# --- Sidebar ---
with st.sidebar:
    st.title("About CineRecs")
    st.info(
        "This is an advanced movie recommender system based on core "
        "Information Retrieval (IR) principles. It uses a hybrid filtering "
        "approach to provide high-quality recommendations."
    )
    st.subheader("How It Works")
    with st.expander("1. Content-Based Filtering (TF-IDF)"):
        st.markdown(
            "Each movie's content (cast, director, keywords, genres, overview) is converted "
            "into a numerical vector using the **TF-IDF** algorithm. This weights words based on their importance."
        )
    with st.expander("2. Similarity Scoring (Cosine Similarity)"):
        st.markdown(
            "The **Cosine Similarity** algorithm measures the angle between these vectors to find movies "
            "that are most similar in content, regardless of their length or popularity."
        )
    with st.expander("3. Quality Re-ranking (IMDB Weighted Rating)"):
        st.markdown(
            "To ensure recommendations are not just similar but also good, the system re-ranks the most similar movies "
            "using a weighted rating that considers both the average score and the number of votes."
        )
    with st.expander("4. Fault Tolerance (Fuzzy Matching)"):
        st.markdown(
            "The search is typo-tolerant! It uses **fuzzy string matching** to find the closest movie title "
            "even if your spelling isn't perfect."
        )
    st.markdown("---")
    st.write("Built with Streamlit & Scikit-learn.")

# --- Main App ---
st.title("🎬 CineRecs: Your Personal Movie Recommender")
st.markdown("Discover your next favorite film with our intelligent IR-powered engine.")

# Load components
movies_df, cosine_sim, indices, tfidf_vec, tfidf_mat = load_model_and_data()

# --- Create Tabs ---
tab1, tab2 = st.tabs(["**🔍 Recommend by Movie**", "**✍️ Recommend by Keywords**"])

# --- Tab 1: Recommend by Movie Title ---
with tab1:
    st.header("Find Movies Similar to One You Love")
    movie_query = st.text_input(
        "Enter a movie title (e.g., 'Batmn Begins', 'Prates of the Cariban'):",
        key="movie_title_input"
    )

    if st.button("Get Recommendations", key="title_rec_button", type="primary"):
        if movie_query:
            with st.spinner(f"Searching for '{movie_query}' and finding top matches..."):
                matched_title, recommendations = get_recommendations_by_movie(movie_query, movies_df, indices, cosine_sim)

                if recommendations:
                    st.info(f"Showing results for: **{matched_title}** (Closest match to your query)")
                    st.subheader(f"Top 5 High-Quality Recommendations:")

                    cols = st.columns(5)
                    # The recommendations are now just a list of titles
                    for i, (col, movie_title) in enumerate(zip(cols, recommendations), 1):
                        with col:
                            st.markdown(
                                f"""
                                <div class="movie-card-text">
                                    <span class="rank-number-text">{i}</span>
                                    <div class="movie-title-text">{movie_title}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.error(matched_title) # Display "No close match found"
        else:
            st.warning("Please enter a movie title to get started.")

# --- Tab 2: Recommend by Keywords ---
with tab2:
    st.header("Discover Movies Based on Your Vibe")
    user_query = st.text_input(
        "Enter keywords, actors, directors, or a plot idea (e.g., 'dystopian future robots'):",
        key="keyword_input"
    )

    if st.button("Get Recommendations", key="desc_rec_button", type="primary"):
        if user_query:
            with st.spinner(f"Retrieving top 5 movies for '{user_query}'..."):
                recommendations = get_recommendations_by_query(user_query, movies_df, tfidf_vec, tfidf_mat)
                if recommendations:
                    st.subheader(f"Top 5 High-Quality Matches for Your Query:")

                    cols = st.columns(5)
                    # The recommendations are now just a list of titles
                    for i, (col, movie_title) in enumerate(zip(cols, recommendations), 1):
                        with col:
                             st.markdown(
                                f"""
                                <div class="movie-card-text">
                                    <span class="rank-number-text">{i}</span>
                                    <div class="movie-title-text">{movie_title}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.warning("Couldn't find any movies matching that description. Try being more general!")
        else:
            st.warning("Please enter a query to get started.")

