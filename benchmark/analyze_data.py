import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import folium
from datetime import datetime
import json

# Load the GeoJSON file with Dubai sectors
dubai_sectors = gpd.read_file('/home/ubuntu/upload/dubai.geojson')

# Print basic information about the sectors
print("Dubai Sectors Information:")
print(f"Number of sectors: {len(dubai_sectors)}")
print(f"Columns: {dubai_sectors.columns.tolist()}")
print("\nSector distribution:")
sector_counts = dubai_sectors['Sector'].value_counts()
print(sector_counts)

# Find Sector 3 specifically (mentioned in the benchmark questions)
sector_3 = dubai_sectors[dubai_sectors['Sector'] == '3']
print(f"\nSector 3 communities: {sector_3['CNAME_E'].tolist()}")
print(f"Number of communities in Sector 3: {len(sector_3)}")

# Load the Talabat sample data
talabat_data = pd.read_csv('/home/ubuntu/upload/talabat_sample.csv')

# Print basic information about the Talabat data
print("\nTalabat Data Information:")
print(f"Number of orders: {len(talabat_data)}")
print(f"Columns: {talabat_data.columns.tolist()}")
print(f"Date range: {talabat_data['order_date_utc'].min()} to {talabat_data['order_date_utc'].max()}")

# Convert vendor and customer locations to GeoDataFrames
def extract_point_coordinates(point_str):
    """Extract coordinates from POINT(lon lat) string format"""
    if pd.isna(point_str):
        return None
    # Extract numbers from the string
    coords = point_str.replace('POINT(', '').replace(')', '').split()
    if len(coords) == 2:
        try:
            return Point(float(coords[0]), float(coords[1]))
        except ValueError:
            return None
    return None

# Create GeoDataFrames for vendors and customer locations
talabat_data['vendor_geometry'] = talabat_data['vendor_location'].apply(extract_point_coordinates)
talabat_data['customer_geometry'] = talabat_data['customer_selected_location'].apply(extract_point_coordinates)

# Filter out rows with invalid geometries
valid_vendor_data = talabat_data[talabat_data['vendor_geometry'].notnull()]
valid_customer_data = talabat_data[talabat_data['customer_geometry'].notnull()]

# Create GeoDataFrames
vendors_gdf = gpd.GeoDataFrame(valid_vendor_data, geometry='vendor_geometry', crs="EPSG:4326")
customers_gdf = gpd.GeoDataFrame(valid_customer_data, geometry='customer_geometry', crs="EPSG:4326")

# Print information about valid geometries
print(f"\nValid vendor locations: {len(vendors_gdf)} out of {len(talabat_data)}")
print(f"Valid customer locations: {len(customers_gdf)} out of {len(talabat_data)}")

# Identify vendors and customers in Sector 3
vendors_in_sector3 = gpd.sjoin(vendors_gdf, sector_3, predicate='within')
customers_in_sector3 = gpd.sjoin(customers_gdf, sector_3, predicate='within')

print(f"\nVendors in Sector 3: {len(vendors_in_sector3)}")
print(f"Customers in Sector 3: {len(customers_in_sector3)}")

# Analyze restaurant types in Sector 3
print("\nRestaurant types in Sector 3:")
restaurant_types = vendors_in_sector3['main_cuisine'].value_counts()
print(restaurant_types)

# Count coffee shops in Sector 3
coffee_shops = vendors_in_sector3[vendors_in_sector3['main_cuisine'] == 'coffee']
print(f"\nNumber of coffee shops in Sector 3: {len(coffee_shops)}")

# Save the results to files for further analysis
dubai_sectors.to_file('/home/ubuntu/site_selection_benchmark/dubai_sectors.geojson', driver='GeoJSON')
sector_3.to_file('/home/ubuntu/site_selection_benchmark/sector_3.geojson', driver='GeoJSON')

# Save vendor and customer data as CSV with WKT geometry
vendors_gdf.to_csv('/home/ubuntu/site_selection_benchmark/vendors.csv')
customers_gdf.to_csv('/home/ubuntu/site_selection_benchmark/customers.csv')
vendors_in_sector3.to_csv('/home/ubuntu/site_selection_benchmark/vendors_sector3.csv')
customers_in_sector3.to_csv('/home/ubuntu/site_selection_benchmark/customers_sector3.csv')

# Create a simple map visualization of Sector 3 with vendors
m = folium.Map(location=[25.24, 55.30], zoom_start=12)

# Add Sector 3 boundary
for idx, row in sector_3.iterrows():
    folium.GeoJson(row['geometry'], 
                  style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'fillOpacity': 0.1}).add_to(m)

# Add vendors in Sector 3 (fixed to use vendor_geometry)
for idx, row in vendors_in_sector3.iterrows():
    popup_text = f"Name: {row['vendor_name']}<br>Cuisine: {row['main_cuisine']}"
    folium.Marker(
        location=[row['vendor_geometry'].y, row['vendor_geometry'].x],
        popup=popup_text,
        icon=folium.Icon(color='red', icon='cutlery', prefix='fa')
    ).add_to(m)

# Save the map
m.save('/home/ubuntu/site_selection_benchmark/sector3_map.html')

print("\nAnalysis complete. Files saved for further processing.")
