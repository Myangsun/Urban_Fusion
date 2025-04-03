"""
Explore the talabat_sample.csv data to understand the food delivery order information
that will be used for constraint evaluation in the site selection benchmark.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# Load the CSV data
print("Loading talabat_sample.csv data...")
df = pd.read_csv('/home/ubuntu/upload/talabat_sample.csv')

# Basic information about the dataset
print(f"\nDataset shape: {df.shape}")
print("\nColumn names:")
for col in df.columns:
    print(f"  - {col}")

# Check for missing values
print("\nMissing values per column:")
missing_values = df.isnull().sum()
for col, count in missing_values.items():
    if count > 0:
        print(f"  - {col}: {count} ({count/len(df)*100:.2f}%)")

# Data types
print("\nData types:")
print(df.dtypes)

# Basic statistics for numerical columns
print("\nBasic statistics for numerical columns:")
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(df[numerical_cols].describe())

# Analyze cuisine types
print("\nCuisine types distribution:")
cuisine_counts = df['main_cuisine'].value_counts().head(10)
print(cuisine_counts)

# Analyze vertical classes
print("\nVertical class distribution:")
vertical_counts = df['vertical_class'].value_counts()
print(vertical_counts)

# Analyze order times
df['order_time'] = pd.to_datetime(df['order_time'])
df['hour'] = df['order_time'].dt.hour
print("\nOrder distribution by hour:")
hour_counts = df['hour'].value_counts().sort_index()
print(hour_counts)

# Analyze order days
df['order_date_utc'] = pd.to_datetime(df['order_date_utc'])
df['day_of_week'] = df['order_date_utc'].dt.day_name()
print("\nOrder distribution by day of week:")
day_counts = df['day_of_week'].value_counts()
print(day_counts)

# Analyze order months
df['month'] = df['order_date_utc'].dt.month
print("\nOrder distribution by month:")
month_counts = df['month'].value_counts().sort_index()
print(month_counts)

# Analyze delivery times
if 'picked_up_at' in df.columns and 'delivered_at' in df.columns:
    # Convert to datetime if they're not already
    df['picked_up_at'] = pd.to_datetime(df['picked_up_at'])
    df['delivered_at'] = pd.to_datetime(df['delivered_at'])
    
    # Calculate delivery time in minutes
    df['delivery_time_minutes'] = (df['delivered_at'] - df['picked_up_at']).dt.total_seconds() / 60
    
    print("\nDelivery time statistics (minutes):")
    print(df['delivery_time_minutes'].describe())

# Analyze order values
print("\nOrder value statistics (AED):")
print(df['gmv_amount_lc'].describe())

# Analyze delivery fees
print("\nDelivery fee statistics (AED):")
print(df['delivery_fee_amount_lc'].describe())

# Analyze service fees
print("\nService fee statistics (AED):")
print(df['service_fee_amount_lc'].describe())

# Analyze basket amounts
print("\nBasket amount statistics (AED):")
print(df['basket_amount_lc'].describe())

# Check for coffee shops
if 'main_cuisine' in df.columns:
    coffee_shops = df[df['main_cuisine'] == 'coffee']
    print(f"\nNumber of coffee shop orders: {len(coffee_shops)}")

# Check for burger restaurants
if 'main_cuisine' in df.columns:
    burger_restaurants = df[df['main_cuisine'] == 'burgers']
    print(f"\nNumber of burger restaurant orders: {len(burger_restaurants)}")

# Check for breakfast restaurants (assumption: cafes and breakfast-related cuisines)
breakfast_cuisines = ['cafe', 'breakfast', 'coffee']
breakfast_restaurants = df[df['main_cuisine'].isin(breakfast_cuisines)]
print(f"\nNumber of potential breakfast restaurant orders: {len(breakfast_restaurants)}")

# Analyze customer locations
if 'customer_selected_location' in df.columns:
    # Extract coordinates from POINT format
    df['customer_location'] = df['customer_selected_location'].str.extract(r'POINT\((.*?)\)')
    
    # Count unique customer locations
    unique_customers = df['customer_location'].nunique()
    print(f"\nNumber of unique customer locations: {unique_customers}")

# Analyze vendor locations
if 'vendor_location' in df.columns:
    # Extract coordinates from POINT format
    df['vendor_coords'] = df['vendor_location'].str.extract(r'POINT\((.*?)\)')
    
    # Count unique vendor locations
    unique_vendors = df['vendor_coords'].nunique()
    print(f"\nNumber of unique vendor locations: {unique_vendors}")

# Summary of findings relevant to benchmark constraints
print("\n=== SUMMARY OF FINDINGS RELEVANT TO BENCHMARK CONSTRAINTS ===")
print("1. Geospatial Data:")
print(f"   - Unique customer locations: {unique_customers}")
print(f"   - Unique vendor locations: {unique_vendors}")
print("   - Data includes coordinates for both customers and vendors")

print("\n2. Temporal Data:")
print("   - Order timestamps available for analysis of:")
print("     * Time of day (breakfast, evening, late-night)")
print("     * Day of week (weekends)")
print("     * Month (summer months)")

print("\n3. Economic Data:")
print(f"   - Order values range from {df['gmv_amount_lc'].min():.2f} to {df['gmv_amount_lc'].max():.2f} AED")
print(f"   - Delivery fees range from {df['delivery_fee_amount_lc'].min():.2f} to {df['delivery_fee_amount_lc'].max():.2f} AED")
print(f"   - Service fees range from {df['service_fee_amount_lc'].min():.2f} to {df['service_fee_amount_lc'].max():.2f} AED")
print(f"   - Basket amounts range from {df['basket_amount_lc'].min():.2f} to {df['basket_amount_lc'].max():.2f} AED")

print("\n4. Market Data:")
print(f"   - Various cuisine types available ({df['main_cuisine'].nunique()} unique types)")
print(f"   - Coffee shops: {len(coffee_shops)} orders")
print(f"   - Burger restaurants: {len(burger_restaurants)} orders")
print(f"   - Potential breakfast restaurants: {len(breakfast_restaurants)} orders")

print("\n5. Data Challenges:")
missing_cols = missing_values[missing_values > 0].index.tolist()
if missing_cols:
    print(f"   - Missing values in columns: {', '.join(missing_cols)}")
else:
    print("   - No significant missing values detected")
print("   - Need to implement spatial calculations for radius-based constraints")
print("   - Need to extract and process timestamps for temporal constraints")
