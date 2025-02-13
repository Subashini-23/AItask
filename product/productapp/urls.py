from django.urls import path
from .views import create_product, create_order ,get_orders
from .views import predict_previous_orders, suggest_remaining_products
from .views import segment_customers, recommend_products , personalized_dashboard ,get_personalized_recommendation
    
    
urlpatterns = [
    path('add_product/', create_product, name='add_product'),
    path('place_order/', create_order, name='place_order'),
    path('get_orders/<str:customer_id>/', get_orders, name='get_orders'),
    path('predict_previous_orders/', predict_previous_orders, name='predict_previous_orders'),
    path('suggest_remaining_products/', suggest_remaining_products, name='suggest_remaining_products'),
    path('customer_segmentation/', segment_customers, name='customer_segmentation'),
    path('product_recommendations/', recommend_products, name='product_recommendations'),
    path('api/dashboard/<str:customer_id>/', personalized_dashboard, name='dashboard'),
]





