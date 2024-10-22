"""
특정 제품에 대한 추천 시스템을 구축하고 대시보드를 통해 사용자가 제품 ID를 입력하면 해당 제품과 유사한 제품을 추천하는 기능을 제공

주요 기능:
1. 데이터 로드: 미국의 제품 데이터를 로드하여 사용
2. 전처리: 필요한 열을 필터링하고 결측값을 처리
3. 콘텐츠 기반 필터링: 카테고리 및 베스트셀러 여부와 같은 제품 특성을 사용하여 추천을 위한 특징 행렬을 생성하고, 차원 축소 및 유사 제품 검색을 수행
4. 대시보드 구성: 사용자가 제품 ID를 입력하면 유사한 제품을 추천하는 대시보드를 제공합니다. 추천된 제품의 ID, 이름, 이미지를 표시

사용 라이브러리:
- pandas: 데이터 처리
- scikit-learn: 차원 축소 및 데이터 표준화
- faiss: 근사 최근접 이웃 검색
- dash: 대시보드 구성 및 시각화
- joblib: 캐싱 기능을 통해 특징 행렬을 재사용
"""

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
import faiss
import json
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import joblib
import os

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
cache_file = 'feature_matrix_reduced.pkl'
if os.path.exists(cache_file):
    # Load the cached feature matrix if available
    feature_matrix_reduced = joblib.load(cache_file)
else:
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

        # Cache the reduced feature matrix for future use
        joblib.dump(feature_matrix_reduced, cache_file)

# Using FAISS for approximate nearest neighbors
d = feature_matrix_reduced.shape[1]
index = faiss.IndexFlatL2(d)  # L2 distance
faiss.normalize_L2(feature_matrix_reduced.astype('float32'))  # Normalize the vectors
for i in tqdm(range(feature_matrix_reduced.shape[0]), desc="Adding vectors to FAISS index"):
    index.add(feature_matrix_reduced[i].reshape(1, -1))

# Dashboard setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.layout = html.Div([
    dbc.Container([
        html.H2("Product Recommendation Dashboard", className="text-center mt-4 mb-4"),
        dbc.Row([
            dbc.Col([
                dcc.Input(id='input_product_id', type='text', placeholder='Enter Product ID', className="mb-2"),
                html.Button('Get Recommendations', id='submit_button', n_clicks=0, className="btn btn-primary mb-4"),
            ], width=4),
            dbc.Col([
                html.H4("Recommended Products:", className="mb-4"),
                html.Div(id='recommendations_output')
            ], width=8)
        ])
    ])
])

# Callback for updating recommendations
@app.callback(
    Output('recommendations_output', 'children'),
    Input('submit_button', 'n_clicks'),
    State('input_product_id', 'value')
)
def update_recommendations(n_clicks, product_id):
    if n_clicks > 0 and product_id:
        try:
            product_index = df[df['product_id'] == product_id].index[0]
            distances, indices = index.search(feature_matrix_reduced[product_index].reshape(1, -1), 6)
            recommended_products = []
            for i in range(1, len(indices.flatten())):
                idx = indices.flatten()[i]
                recommended_products.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"Product Name: {df.iloc[idx]['product_name']}", className="card-title"),
                            html.P(f"Product ID: {df.iloc[idx]['product_id']}", className="card-text"),
                            html.P(f"Category: {df.iloc[idx]['category']}", className="card-text"),
                            html.P(f"Rating: {df.iloc[idx]['rating']}", className="card-text"),
                            html.P(f"Reviews: {df.iloc[idx]['reviews']}", className="card-text"),
                            html.P(f"Trending: {'Yes' if df.iloc[idx]['isBestSeller'] == 1 else 'No'}", className="card-text"),
                        ]),
                        dbc.CardImg(src=df.iloc[idx]['img_link'], top=True, style={'width': '150px', 'height': '150px', 'object-fit': 'cover', 'margin': '10px auto'})
                    ], className="mb-4")
                )
            return recommended_products
        except IndexError:
            return html.P("Product ID not found.")
    return html.P("Enter a Product ID and click 'Get Recommendations'.")

# Run the dashboard
if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')
