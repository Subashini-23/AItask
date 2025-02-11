# recommendation.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from django.db import connection
from .models import Order, Product

# Fetch order data from the database
def fetch_order_data():
    query = """
    SELECT customer_id, product_id, COUNT(*) as quantity
    FROM productapp_order
    GROUP BY customer_id, product_id;
    """
    df = pd.read_sql_query(query, connection)
    return df

# Apply K-Means clustering to segment customers
def segment_customers():
    df = fetch_order_data()
    pivot_df = df.pivot(index='customer_id', columns='product_id', values='purchase_count').fillna(0)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    pivot_df['Cluster'] = kmeans.fit_predict(pivot_df)
    
    return pivot_df[['Cluster']]

# Apply Apriori to find product associations
def product_association_rules():
    df = fetch_order_data()
    pivot_df = df.pivot(index='customer_id', columns='product_id', values='purchase_count').fillna(0)
    pivot_df = pivot_df.applymap(lambda x: 1 if x > 0 else 0)
    
    frequent_itemsets = apriori(pivot_df, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    return rules[['antecedents', 'consequents']]

# Recommend products based on past purchases
def recommend_products(customer_id):
    df = fetch_order_data()
    
    past_orders = df[df['customer_id'] == customer_id]['product_id'].tolist()
    recommended = []
    
    rules = product_association_rules()
    
    for antecedent, consequent in zip(rules['antecedents'], rules['consequents']):
        if set(antecedent).issubset(set(past_orders)):
            recommended.extend(list(consequent))
    
    recommended = list(set(recommended) - set(past_orders))
    
    product_names = Product.objects.filter(product_id__in=recommended).values_list('product_name', flat=True)
    
    return list(product_names)
