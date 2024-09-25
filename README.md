# Movie Recommendation System

This repository contains a **Movie Recommendation System** built using collaborative filtering techniques. The system predicts a user's potential movie ratings and suggests new movies based on past preferences. This project utilizes a **truncated SVD (Singular Value Decomposition)** for dimensionality reduction, along with **cosine similarity** to enhance recommendation quality.

## Features
- **Collaborative Filtering**: Recommends movies based on the preferences of similar users.
- **User-Item Matrix**: Builds a matrix of user interactions with movies for collaborative filtering.
- **Truncated SVD**: Reduces the dimensions of the user-item matrix for efficient computation.
- **Cosine Similarity**: Calculates similarity between users or items to predict movie ratings.

## Dataset
The dataset is sourced from **the movies dataset**, consisting of user ratings for movies, as well as metadata like movie genres, budget, and more.

### Dataset Source
- [Kaggle Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

## Approach

### 1. **User-Item Matrix**:
   - A matrix of users as rows and movies as columns is constructed using the ratings provided by each user for specific movies.

### 2. **Cosine Similarity**:
   - **Cosine similarity** is calculated between user vectors in the matrix to identify similar users.

### 3. **Truncated SVD**:
   - **SVD** is applied to decompose the matrix and reduce dimensionality, making the recommendation process more computationally efficient. 
   - This reduced matrix is used to generate predictions for unrated movies.

### 4. **Recommendations**:
   - The model predicts ratings for unrated movies and recommends the highest-rated ones.

## Code Usage

1. **Preprocess the data**:
   - Remove unnecessary columns (like `timestamp`) and limit the dataset for testing purposes.
   - Convert the ratings data into a user-item matrix for model training.

    ```python
    ratings_df = ratings.drop('timestamp', axis=1)
    ratings_df = ratings_df[ratings_df['userId'].isin(ratings_df['userId'].unique()[:1000])]  # Limiting to 1000 users for efficiency
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    ```

2. **Apply Truncated SVD**:
   - Use SVD to decompose the user-item matrix and perform dimensionality reduction.

    ```python
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=20)
    user_item_svd = svd.fit_transform(user_item_matrix)
    ```

3. **Cosine Similarity**:
   - Compute cosine similarity between users to find those with similar tastes.

    ```python
    from sklearn.metrics.pairwise import cosine_similarity
    user_similarity = cosine_similarity(user_item_svd)
    ```

4. **Make Recommendations**:
   - For a given user, find movies they haven’t rated and recommend the top N movies with the highest predicted ratings.

    ```python
    def recommend_movies(user_id, user_item_matrix, svd, n_recommendations=5):
        # Predict ratings and return top recommended movies
        user_ratings = user_item_matrix.loc[user_id].values.reshape(1, -1)
        user_ratings_svd = svd.transform(user_ratings)
        user_ratings_pred = svd.inverse_transform(user_ratings_svd)
        unrated_movie_indices = np.where(user_ratings == 0)[1]
        predicted_ratings = user_ratings_pred[0][unrated_movie_indices]
        top_movie_indices = unrated_movie_indices[predicted_ratings.argsort()[::-1][:n_recommendations]]
        return top_movie_indices
    ```

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn


