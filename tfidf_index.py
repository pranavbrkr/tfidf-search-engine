# WARNING: This file is for development and testing purposes only.
# It is NOT intended for use in production environments.


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def perform_tfidf_indexing(path, key_col_name, desc_col_name, vectorizer):
    # Get the data from the DB
    df = pd.read_csv(path)

    # print(df['Description'].isna().sum())

    # Parse the Data
    keys = df[key_col_name].tolist()
    descriptions = df[desc_col_name].tolist()
    
    # Compute TF-IDF
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    # Create a dictionary so we can refer back to the original document
    tfidf_dict = {key: tfidf_matrix[i] for i, key in enumerate(keys)}

    return tfidf_dict, vectorizer

def search_docs(query, tfidf_dict, vectorizer, top_n=1):
    # Transform the query to a TF-IDF vector
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity
    similarities = {}
    for key, doc_vector in tfidf_dict.items():
        similarity = cosine_similarity(query_vector, doc_vector)[0][0]
        similarities[key] = similarity
    
    # Sort documents by similarity and return top N keys
    sorted_keys = sorted(similarities, key=similarities.get, reverse=True)
    return sorted_keys[:top_n]

def main():
    # change me to test the indexing and VERY BASIC searching
    phrase_to_search = "Life is about trying new things and taking risks"

    data_path = 'cleaned_goodreads_data.csv'
    key_col_name = 'Unnamed: 0'
    desc_col_name = 'Description'
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # perform indexing
    tfidf_dictionary, fitted_vectorizer = perform_tfidf_indexing(data_path, key_col_name, desc_col_name, tfidf_vectorizer)
    # perform_tfidf_indexing(data_path, key_col_name, desc_col_name, tfidf_vectorizer)

    # search
    relevant_books = search_docs(phrase_to_search, tfidf_dictionary, fitted_vectorizer, 5)

    for book in relevant_books:
        print(book)



if __name__ == "__main__":
    main()