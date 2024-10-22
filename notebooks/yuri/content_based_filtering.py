import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
import faiss
import json

# Load data
us_df = pd.read_csv('/Users/yoorichoi/Documents/ds_study/Amazon_sales_machine_learning/data/amazon_usa.csv')

# Preprocessing
# Filtering necessary columns that are present in the dataframes
available_columns = [col for col in ['discount_percentage', 'boughtInLastMonth', 'actual_price', 'discounted_price', 'product_name', 'discounted_price_KRW', 'category', 'rating', 'product_link', 'img_link', 'isBestSeller', 'reviews', 'product_id'] if col in us_df.columns]
df = us_df[available_columns]

# Handling missing values
if 'rating' in df.columns:
    df.loc[:, 'rating'] = df['rating'].fillna(df['rating'].mean())

# Content-Based Filtering (using product features with dimensionality reduction)
if 'category' in df.columns and 'isBestSeller' in df.columns:
    # Creating a feature matrix using category and isBestSeller
    df.loc[:, 'isBestSeller'] = df['isBestSeller'].astype(int)
    category_dummies = pd.get_dummies(df['category'], drop_first=True)
    feature_matrix = pd.concat([category_dummies, df[['isBestSeller']]], axis=1)

    # Standardizing the feature matrix
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # Dimensionality reduction using Truncated SVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    feature_matrix_reduced = svd.fit_transform(feature_matrix_scaled)

    # Using FAISS for approximate nearest neighbors
    d = feature_matrix_reduced.shape[1]
    index = faiss.IndexFlatL2(d)  # L2 distance
    faiss.normalize_L2(feature_matrix_reduced.astype('float32'))  # Normalize the vectors
    index.add(feature_matrix_reduced)

    # Example: Get recommendations for a specific product index
    product_index = 0
    distances, indices = index.search(feature_matrix_reduced[product_index].reshape(1, -1), 6)

    # Save recommended products to a JSON file
    recommended_products = [df.iloc[indices.flatten()[i]]['product_id'] for i in range(1, len(indices.flatten()))]
    with open('recommended_products.json', 'w') as f:
        json.dump(recommended_products, f)

# Hybrid Approach
# Use content-based filtering to refine recommendations.
def hybrid_recommendations(product_id, n_recommendations=5):
    if 'product_id' in df.columns and 'category' in df.columns and 'isBestSeller' in df.columns:
        # Content-based filtering - refine recommendations based on product content
        product_index = df[df['product_id'] == product_id].index[0]
        distances, indices = index.search(feature_matrix_reduced[product_index].reshape(1, -1), n_recommendations + 1)
        recommended_products = df.iloc[indices.flatten()[1:]]['product_id'].values

        return list(recommended_products)
    else:
        return []

# Example: Get hybrid recommendations for a product
if 'product_id' in df.columns:
    product_id = df['product_id'].iloc[0]
    hybrid_recs = hybrid_recommendations(product_id)
    # Save hybrid recommendations to a JSON file
    with open('hybrid_recommendations.json', 'w') as f:
        json.dump(hybrid_recs, f)